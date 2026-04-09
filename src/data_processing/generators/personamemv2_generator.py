"""
Generator for PersonaMem-v2 (chunk-memory version).

Reads local CSV (val_text.csv) and produces:
- source_input.jsonl (chunked conversation history, only chunks with >=1 matched MCQ)
- target.jsonl (MCQ QA pairs)
"""

from __future__ import annotations

import ast
import collections
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from src.data_processing.schemas import SourceExample, InferenceInputExample
from src.data_processing.generators.base import BaseGenerator
from src.utils import print_example


class PersonaMemV2Generator(BaseGenerator):
    """
    PersonaMem-v2 generator.

    Processing flow (conversation-first):
      1. Filter rows to MCQ only (presence of `incorrect_answers`).
      2. Group MCQ rows by `chat_history_32k_link` (one conversation per group).
      3. For each conversation: load chat history once, build token-bounded chunks
         (each chunk starts with a user turn, ends with an assistant turn, <=5000 tokens).
      4. For each chunk: find all MCQ rows whose `related_conversation_snippet`
         messages are all present in the chunk (non-contiguous).
         - No matches  → discard chunk.
         - >=1 match   → emit SourceExample; merge matched QA ids into target_ids.
      5. Guarantee: len(sources) <= len(targets).
    """

    def __init__(self, dataset_name: str = "personamem-v2"):
        super().__init__(dataset_name)

    def generate_sources(
        self,
        source_file: str,
        target_file: Optional[str] = None,
        output_data_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        if output_data_dir is None:
            raise ValueError("output_data_dir is required for PersonaMemV2Generator")

        try:
            import tiktoken  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "tiktoken is required. Install via `pip install tiktoken`."
            ) from exc

        encoder = tiktoken.get_encoding("cl100k_base")

        # ------------------------------------------------------------------ #
        # Helper functions                                                     #
        # ------------------------------------------------------------------ #

        def count_tokens(msg: dict) -> int:
            return len(encoder.encode(f"{msg.get('role', '')}: {msg.get('content', '')}"))

        def normalize_messages(messages: Any) -> List[dict]:
            if not isinstance(messages, list):
                return []
            return [
                {"role": m.get("role", ""), "content": m.get("content", "")}
                for m in messages
                if m.get("role", "") != "system"
            ]

        def maybe_json(val: Any) -> Any:
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except Exception:
                    return val
            return val

        def load_history(link: Optional[str], base_dir: str) -> List[dict]:
            if not link:
                return []
            candidates = [
                os.path.normpath(os.path.join(base_dir, link)),
                os.path.normpath(os.path.join(os.path.dirname(base_dir), link)),
                os.path.normpath(
                    os.path.join(os.path.dirname(os.path.dirname(base_dir)), link)
                ),
            ]
            path = next((p for p in candidates if os.path.exists(p)), None)
            if not path:
                return []
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "chat_history" in data:
                    return data["chat_history"]
            except Exception:
                pass
            return []

        def build_chunks(
            messages: List[dict], max_tokens: int = 5000
        ) -> List[Tuple[List[dict], int]]:
            """
            Split `messages` into non-overlapping chunks where:
              - each chunk starts with a 'user' turn
              - each chunk ends with an 'assistant' turn
              - total tokens per chunk <= max_tokens

            Returns list of (chunk_messages, chunk_token_count).
            """
            chunks: List[Tuple[List[dict], int]] = []
            i = 0
            n = len(messages)
            while i < n:
                # Advance to next 'user' turn
                while i < n and messages[i].get("role") != "user":
                    i += 1
                if i >= n:
                    break
                start = i
                token_sum = 0
                last_assistant = -1
                j = i
                while j < n:
                    t = count_tokens(messages[j])
                    if token_sum + t > max_tokens:
                        break
                    token_sum += t
                    if messages[j].get("role") == "assistant":
                        last_assistant = j
                    j += 1
                if last_assistant < start:
                    # No assistant turn fits within the token budget from this start
                    i = start + 1
                    continue
                chunk_msgs = messages[start : last_assistant + 1]
                chunk_tokens = sum(count_tokens(m) for m in chunk_msgs)
                chunks.append((chunk_msgs, chunk_tokens))
                i = last_assistant + 1
            return chunks

        def snippet_in_chunk(snippet: List[dict], chunk_msgs: List[dict]) -> bool:
            """Return True iff every message in snippet appears in chunk_msgs (role+content)."""
            for sm in snippet:
                if not any(
                    sm["role"] == cm["role"] and sm["content"] == cm["content"]
                    for cm in chunk_msgs
                ):
                    return False
            return True

        def parse_user_query(raw: str) -> str:
            """Extract content from a dict-string like {'role': 'user', 'content': '...'}."""
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, dict):
                    return parsed.get("content", raw)
            except Exception:
                pass
            return raw

        def normalize_choices(row: Dict[str, Any]) -> Optional[Any]:
            incorrect = maybe_json(row.get("incorrect_answers")) or []
            correct = row.get("correct_answer")
            choices = list(incorrect)
            if correct is not None and correct not in choices:
                choices.append(correct)
            return choices if choices else None

        # ------------------------------------------------------------------ #
        # Load CSV and filter MCQ rows                                        #
        # ------------------------------------------------------------------ #

        if not source_file or not os.path.exists(source_file):
            raise FileNotFoundError(f"source_file not found: {source_file}")
        base_dir = os.path.dirname(source_file)

        with open(source_file, "r", encoding="utf-8") as f:
            all_rows = list(csv.DictReader(f))
        print(f"[PersonaMemV2Generator] Loaded {len(all_rows)} rows from {source_file}")

        mcq_rows = [r for r in all_rows if r.get("incorrect_answers", "").strip()]
        print(f"[PersonaMemV2Generator] MCQ rows: {len(mcq_rows)}")

        # ------------------------------------------------------------------ #
        # Group MCQ rows by conversation (chat_history_32k_link)             #
        # ------------------------------------------------------------------ #

        conv_groups: Dict[str, List[dict]] = collections.defaultdict(list)
        for row in mcq_rows:
            key = row.get("chat_history_32k_link") or row.get("persona_id", "")
            conv_groups[key].append(row)

        os.makedirs(output_data_dir, exist_ok=True)

        sources: List[SourceExample] = []
        targets: List[InferenceInputExample] = []
        chunk_global_id = 0
        qa_global_id = 0

        stats = {
            "conversations": 0,
            "chunks_total": 0,
            "chunks_kept": 0,
            "chunks_dropped": 0,
            "qa_mapped": 0,
        }

        # ------------------------------------------------------------------ #
        # Conversation-first processing                                       #
        # ------------------------------------------------------------------ #

        for conv_idx, (conv_key, conv_rows) in enumerate(conv_groups.items()):
            stats["conversations"] += 1

            # Load history from the first row (all rows in the group share the same history)
            first_row = conv_rows[0]
            history_raw = maybe_json(first_row.get("chat_history_32k", []))
            history = normalize_messages(history_raw)
            if not history:
                history = normalize_messages(load_history(conv_key, base_dir))
            if not history:
                continue

            chunks = build_chunks(history, max_tokens=5000)
            stats["chunks_total"] += len(chunks)

            # Pre-parse snippets for every QA row in this conversation
            qa_snippets: List[Tuple[dict, List[dict]]] = []
            for row in conv_rows:
                snippet_raw = maybe_json(row.get("related_conversation_snippet", []))
                snippet = normalize_messages(snippet_raw)
                qa_snippets.append((row, snippet))

            # For each chunk, find all matching QA rows
            for chunk_idx, (chunk_msgs, chunk_tokens) in enumerate(chunks):
                matched_rows = [
                    (row, snippet)
                    for row, snippet in qa_snippets
                    if snippet and snippet_in_chunk(snippet, chunk_msgs)
                ]

                if not matched_rows:
                    stats["chunks_dropped"] += 1
                    continue

                stats["chunks_kept"] += 1
                matched_qa_ids: List[int] = []

                for row, snippet in matched_rows:
                    choices = normalize_choices(row)
                    user_query = parse_user_query(row.get("user_query", ""))
                    correct_answer = row.get("correct_answer", "")

                    target = InferenceInputExample(
                        source="personamem-v2",
                        id=qa_global_id,
                        task_main=user_query,
                        metadata={
                            "user_query": user_query,
                            "correct_answer": correct_answer,
                            "choices": choices,
                            "related_conversation_snippet": snippet,
                        },
                    )
                    targets.append(target)
                    matched_qa_ids.append(qa_global_id)
                    qa_global_id += 1

                source = SourceExample(
                    source="personamem-v2",
                    id=chunk_global_id,
                    task_main=f"PersonaMem-v2 conv{conv_idx} chunk{chunk_idx}",
                    messages=chunk_msgs,
                    target_ids=matched_qa_ids,
                    memory="",
                    metadata={
                        "conversation_key": conv_key,
                        "conv_idx": conv_idx,
                        "chunk_idx": chunk_idx,
                        "chunk_token_len": chunk_tokens,
                        "num_turns": len(chunk_msgs),
                    },
                )
                sources.append(source)
                stats["qa_mapped"] += len(matched_qa_ids)
                chunk_global_id += 1

        # ------------------------------------------------------------------ #
        # Output                                                              #
        # ------------------------------------------------------------------ #

        print(
            f"[PersonaMemV2Generator] "
            f"conversations={stats['conversations']}, "
            f"chunks_total={stats['chunks_total']}, "
            f"chunks_kept={stats['chunks_kept']}, "
            f"chunks_dropped={stats['chunks_dropped']}, "
            f"qa_mapped={stats['qa_mapped']}"
        )
        print(
            f"[PersonaMemV2Generator] "
            f"sources={len(sources)}, targets={len(targets)} "
            f"(sources <= targets: {len(sources) <= len(targets)})"
        )

        if sources:
            print_example("First Source Example", sources[0].to_dict())
        if targets:
            print_example("First Target Example", targets[0].to_dict())

        self.save_outputs(sources, targets, output_data_dir)
