"""
Generator for PrefEval Implicit Preference - Persona-driven Conversation (MCQ evaluation).

Reads local per-topic JSON files from:
  data/PrefEval/implicit_preference/persona-driven/{topic}.json   (persona + question)
  data/PrefEval/mcq_options/{topic}.json                          (4 classification options)

Produces:
  source_input.jsonl  -- SourceExample: conversation messages per entry
  target.jsonl        -- InferenceInputExample: question + shuffled MCQ options

Data contract:
  - Index i in persona-driven corresponds to index i in mcq_options.
  - classification_task_options[0] is always the correct (preference-following) answer.
  - Options are shuffled with a fixed-seed RNG for reproducibility.

No collection phase needed; generate_targets() raises NotImplementedError.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from src.data_processing.generators.base import BaseGenerator
from src.data_processing.schemas import InferenceInputExample, SourceExample
from src.utils import print_example

# Fixed seed ensures identical shuffles across runs and machines.
_SHUFFLE_SEED = 42


class PrefEvalGenerator(BaseGenerator):
    """
    PrefEval Implicit Persona + MCQ generator.

    source_file = path to the persona-driven directory
                  (e.g. data/PrefEval/implicit_preference/persona-driven)

    mcq_options_dir is derived automatically:
        <source_file>/../../mcq_options  →  data/PrefEval/mcq_options
    """

    def __init__(self, dataset_name: str = "prefeval-mcq"):
        super().__init__(dataset_name)

    # ── Public API ───────────────────────────────────────────────────────────

    def generate_sources(
        self,
        source_file: str,
        target_file: Optional[str] = None,
        output_data_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        if output_data_dir is None:
            raise ValueError("output_data_dir is required for PrefEvalGenerator")

        persona_dir = source_file
        if not os.path.isdir(persona_dir):
            raise NotADirectoryError(
                f"source_file must be the persona-driven directory: {persona_dir}"
            )

        # Derive MCQ options dir: ../../mcq_options relative to persona_dir
        mcq_options_dir = os.path.normpath(
            os.path.join(persona_dir, "..", "..", "mcq_options")
        )
        if not os.path.isdir(mcq_options_dir):
            raise NotADirectoryError(
                f"MCQ options directory not found at {mcq_options_dir}. "
                "Expected data/PrefEval/mcq_options/ as a sibling of implicit_preference/."
            )

        os.makedirs(output_data_dir, exist_ok=True)

        rng = random.Random(_SHUFFLE_SEED)
        sources: List[SourceExample] = []
        targets: List[InferenceInputExample] = []
        global_id = 0
        stats: Dict[str, int] = {
            "topics": 0,
            "entries": 0,
            "skipped_no_mcq": 0,
            "skipped_null_conv": 0,
        }

        for topic_filename in sorted(
            f for f in os.listdir(persona_dir) if f.endswith(".json")
        ):
            topic = topic_filename[:-5]  # strip .json suffix
            persona_path = os.path.join(persona_dir, topic_filename)
            mcq_path = os.path.join(mcq_options_dir, topic_filename)

            if not os.path.exists(mcq_path):
                print(
                    f"[PrefEvalGenerator] MCQ options file not found for topic "
                    f"'{topic}' ({mcq_path}), skipping topic."
                )
                continue

            with open(persona_path, "r", encoding="utf-8") as f:
                persona_data: List[Dict[str, Any]] = json.load(f)
            with open(mcq_path, "r", encoding="utf-8") as f:
                mcq_data: List[Dict[str, Any]] = json.load(f)

            if len(persona_data) != len(mcq_data):
                print(
                    f"[PrefEvalGenerator] Length mismatch for topic '{topic}': "
                    f"{len(persona_data)} persona entries vs {len(mcq_data)} mcq entries. "
                    "Using min(len)."
                )

            stats["topics"] += 1

            for entry_idx, (persona_entry, mcq_entry) in enumerate(
                zip(persona_data, mcq_data)
            ):
                mcq_options: List[str] = mcq_entry.get("classification_task_options", [])
                if not mcq_options or len(mcq_options) < 2:
                    stats["skipped_no_mcq"] += 1
                    continue

                conversation: Dict[str, Any] = persona_entry.get("conversation", {})
                conv_msgs = self._parse_conversation(conversation)
                if not conv_msgs:
                    stats["skipped_null_conv"] += 1
                    continue

                shuffled, correct_idx = self._shuffle_options_rng(rng, mcq_options)

                preference: str = persona_entry.get("preference", "")
                question: str = persona_entry.get("question", "")
                explanation: str = persona_entry.get("explanation", "")
                persona: str = persona_entry.get("persona", "")

                source = SourceExample(
                    source=self.dataset_name,
                    id=global_id,
                    task_main=f"PrefEval persona-driven: topic={topic} entry={entry_idx}",
                    messages=conv_msgs,
                    target_ids=[global_id],
                    memory="",
                    metadata={"topic": topic, "entry_idx": entry_idx},
                )
                target = InferenceInputExample(
                    source=self.dataset_name,
                    id=global_id,
                    task_main=question,
                    metadata={
                        "preference": preference,
                        "persona": persona,
                        "topic": topic,
                        "explanation": explanation,
                        "shuffled_options": shuffled,
                        "correct_idx": correct_idx,
                    },
                )

                sources.append(source)
                targets.append(target)
                global_id += 1
                stats["entries"] += 1

        print(
            f"[PrefEvalGenerator] "
            f"topics={stats['topics']}, "
            f"entries={stats['entries']}, "
            f"skipped_no_mcq={stats['skipped_no_mcq']}, "
            f"skipped_null_conv={stats['skipped_null_conv']}"
        )
        print(
            f"[PrefEvalGenerator] "
            f"sources={len(sources)}, targets={len(targets)}"
        )

        if sources:
            print_example("First Source Example", sources[0].to_dict())
        if targets:
            print_example("First Target Example", targets[0].to_dict())

        self.save_outputs(sources, targets, output_data_dir)

    def generate_targets(
        self,
        output_data_dir: Optional[str] = None,
        max_targets: int = -1,
        **kwargs,
    ) -> None:
        raise NotImplementedError(
            "PrefEvalGenerator does not support a collection phase. "
            "Call generate_sources() with the persona-driven directory directly."
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_conversation(conversation: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert PrefEval persona-driven conversation dict to a messages list.

        Input:
            {"0": {"user": "...", "assistant": "..."}, "1": {...}, ...}
            (some turns may be None — those are skipped)

        Output:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        msgs: List[Dict[str, str]] = []
        for key in sorted(conversation.keys(), key=lambda k: int(k)):
            turn = conversation[key]
            if turn is None:
                continue
            user_text: str = turn.get("user") or ""
            assistant_text: str = turn.get("assistant") or ""
            if user_text:
                msgs.append({"role": "user", "content": user_text})
            if assistant_text:
                msgs.append({"role": "assistant", "content": assistant_text})
        return msgs

    @staticmethod
    def _shuffle_options_rng(
        rng: random.Random, options: List[str]
    ) -> Tuple[List[str], str]:
        """
        Shuffle `options` using `rng`.  options[0] is always the correct answer.
        Returns (shuffled_list, correct_letter) where correct_letter ∈ {"A","B","C","D"}.
        """
        shuffled = options[:]
        rng.shuffle(shuffled)
        correct_letter = "ABCD"[shuffled.index(options[0])]
        return shuffled, correct_letter
