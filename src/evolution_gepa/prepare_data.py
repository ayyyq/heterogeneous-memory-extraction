"""
Data preparation for GEPA memory extraction optimization.

Prepares mixed dataset from multiple sources for training.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import random
from pathlib import Path
from typing import Optional

from src.data_processing.schemas import SourceExample, InferenceInputExample
from src.data_processing.loader import DataLoader


def prepare_gepa_data(
    datasets: list[str],
    source_file_paths: list[str],
    target_file_paths: list[str],
    num_sources_per_dataset: Optional[int] = None,
    val_size: float = 0.5,
    seed: int = 42,
    target_selection: str = "first",  # 'first' or 'random'
    split_file_path: str | None = None,
    refresh: bool = False,
    max_train_per_dataset: int = 100,
) -> tuple[list[dict], list[dict]]:
    """
    Prepare training and validation data for GEPA optimization.

    There are two layers of splitting:

    1. **Global per-dataset train/test split (persisted to a single JSON file)**:
       - For each dataset, half of the available `SourceExample`s (capped at 100)
         are selected as the **per-dataset training pool** (`train_source_ids`).
       - The remaining sources implicitly form the in-distribution **test pool**.
       - This information is stored in a single *global* split file:

         Global split file structure (JSON):

         ```json
         {
           "seed": int,
           "max_train_per_dataset": int,
           "train_ratio_per_dataset": float,
           "datasets": [
             {
               "name": str,
               "source_file_path": str,
               "target_file_path": str,
               "train_source_ids": [str, ...]
             },
             ...
           ]
         }
         ```

    2. **GEPA-level train/val split (in-memory only, stratified by dataset)**:
       - All source-target pairs whose `source.id` is in the per-dataset
         `train_source_ids` are aggregated across datasets.
       - Splitting is done **within each dataset first**, then merged:
         * if `val_size < 1`, each dataset uses that proportion for validation;
         * if `val_size >= 1`, validation count is allocated across datasets
           proportionally using the largest remainder method.
       - The merged `trainset`/`valset` are shuffled again before returning.

    Control flow:
        - If `split_file_path` is provided, `refresh` is False, and the file exists:
            * the global split file is loaded and its dataset entries (including
              `train_source_ids`) are used as the authoritative configuration.
        - Otherwise (no file, missing `split_file_path`, or `refresh` is True):
            * `datasets`, `source_file_paths`, and `target_file_paths` must be
              provided and have the same non-zero length;
            * a new global split config is generated and written to a JSON file.
              If `split_file_path` is None, a default is derived from the first
              `source_file_path` as:
                  Path(source_file_paths[0]).parent / "gepa_global_split.json".

    Args:
        datasets: Dataset names (ignored when loading from an existing split file).
        source_file_paths: Path to the source file for each dataset.
        target_file_paths: Path to the target file for each dataset.
        num_sources_per_dataset: Number of sources to sample per dataset before
                                 defining the per-dataset split (generation mode
                                 only; ignored when loading an existing split).
        val_size: When < 1, validation proportion; when >= 1, validation sample
                  count. Remaining data goes to training.
        seed: Random seed for reproducibility.
        target_selection: How to select target for each source ('first' or 'random').
        split_file_path: Optional path to the global split file. If None and a new
                         split is generated, defaults to
                         Path(source_file_paths[0]).parent / "gepa_global_split.json".
        refresh: If True, always regenerate the global split file even if it exists.

    Returns:
        Tuple of (trainset, valset) where each item is a MemoryExtractionDataInst.
    """
    random.seed(seed)
    MAX_TRAIN_PER_DATASET = max_train_per_dataset
    TRAIN_RATIO_PER_DATASET = 0.5

    # Determine whether to load an existing global split or generate a new one.
    split_path: Path | None = Path(split_file_path) if split_file_path is not None else None
    use_existing_split = (
        split_path is not None and split_path.is_file() and not refresh
    )

    dataset_entries: list[dict] = []
    generated_split_datasets: list[dict] = []

    if use_existing_split:
        with open(split_path, "r", encoding="utf-8") as f:
            try:
                split_cfg = json.load(f)
            except Exception as e:
                raise ValueError(
                    f"[PrepareData] Failed to parse split file {split_path}: {e}"
                ) from e

        if not isinstance(split_cfg, dict) or "datasets" not in split_cfg:
            raise ValueError(
                f"[PrepareData] Invalid split file format at {split_path}: "
                "expected a dict with key 'datasets'."
            )

        datasets_cfg = split_cfg.get("datasets")
        if not isinstance(datasets_cfg, list) or not datasets_cfg:
            raise ValueError(
                f"[PrepareData] Invalid split file {split_path}: "
                "field 'datasets' must be a non-empty list."
            )

        dataset_entries = datasets_cfg
        print(
            f"[PrepareData] Loading existing global split from {split_path} "
            f"for {len(dataset_entries)} datasets..."
        )
    else:
        if not (datasets and source_file_paths and target_file_paths):
            raise ValueError(
                "[PrepareData] When no existing split file is available or refresh=True, "
                "`datasets`, `source_file_paths` and `target_file_paths` must be provided "
                "and non-empty."
            )
        if not (
            len(datasets) == len(source_file_paths) == len(target_file_paths)
        ):
            raise ValueError(
                "[PrepareData] datasets, source_file_paths and target_file_paths "
                "must have the same length.\n"
                f"  datasets     = {datasets}\n"
                f"  source_files = {source_file_paths}\n"
                f"  target_files = {target_file_paths}"
            )

        dataset_entries = [
            {
                "name": ds,
                "source_file_path": src,
                "target_file_path": tgt,
            }
            for ds, src, tgt in zip(datasets, source_file_paths, target_file_paths)
        ]

        if split_path is None:
            # Default global split location derived from first source file.
            first_source = source_file_paths[0]
            split_path = Path(first_source).parent / "gepa_global_split.json"

        print(
            f"[PrepareData] Generating new global split for {len(dataset_entries)} "
            f"datasets; will write to {split_path}"
        )

    pairs_by_dataset: dict[str, list[dict]] = {}

    for entry in dataset_entries:
        dataset_name = entry.get("name")
        source_file_path = entry.get("source_file_path")
        target_file_path = entry.get("target_file_path")
        if not isinstance(dataset_name, str) or not isinstance(source_file_path, str) or not isinstance(
            target_file_path, str
        ):
            raise ValueError(
                "[PrepareData] Each dataset entry in split configuration must contain "
                "'name', 'source_file_path', and 'target_file_path' as strings."
            )

        # Check if files exist
        if not os.path.exists(source_file_path):
            print(
                f"\033[93m[PrepareData] WARNING: Source file not found for dataset {dataset_name}: "
                f"{source_file_path} — skipping this dataset.\033[0m"
            )
            continue
        if not os.path.exists(target_file_path):
            print(
                f"\033[93m[PrepareData] WARNING: Target file not found for dataset {dataset_name}: "
                f"{target_file_path} — skipping this dataset.\033[0m"
            )
            continue

        # Load sources
        sources_dict = DataLoader().load_sources(source_file_path)
        sources = list(sources_dict.values())

        # In generation mode, we can optionally truncate before defining the split.
        if num_sources_per_dataset is not None and not use_existing_split:
            sources = sources[:num_sources_per_dataset]
        elif num_sources_per_dataset is not None and use_existing_split:
            print(
                "[PrepareData] Warning: num_sources_per_dataset is ignored when "
                "loading from an existing global split file."
            )

        if not sources:
            print(
                f"\033[93m[PrepareData] WARNING: No sources available for dataset {dataset_name} "
                f"(source_file_path={source_file_path}) — skipping.\033[0m"
            )
            continue

        # Load targets
        targets_dict: dict[str, InferenceInputExample] = {}
        with open(target_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    target = InferenceInputExample.from_dict(json.loads(line))
                    targets_dict[target.id] = target

        all_source_ids = [s.id for s in sources]
        source_id_set = set(all_source_ids)

        # Determine / load per-dataset training source IDs.
        if use_existing_split:
            train_source_ids = entry.get("train_source_ids")
            if not isinstance(train_source_ids, list) or not train_source_ids:
                raise ValueError(
                    f"[PrepareData] Split file {split_path} dataset '{dataset_name}' "
                    "has missing or empty 'train_source_ids'."
                )
        else:
            n = len(all_source_ids)
            desired = min(n // 2, MAX_TRAIN_PER_DATASET)
            if desired == 0 and n > 0:
                desired = min(n, MAX_TRAIN_PER_DATASET)

            if desired == 0:
                raise ValueError(
                    f"[PrepareData] Unable to select training sources for dataset "
                    f"{dataset_name}: no usable sources."
                )

            train_source_ids = random.sample(all_source_ids, desired)

            generated_split_datasets.append(
                {
                    "name": dataset_name,
                    "source_file_path": str(source_file_path),
                    "target_file_path": str(target_file_path),
                    "train_source_ids": sorted(train_source_ids),
                }
            )

        # Strict validation: ensure all train_source_ids exist in the loaded sources.
        missing_ids = [sid for sid in train_source_ids if sid not in source_id_set]
        if missing_ids:
            raise ValueError(
                f"[PrepareData] Split mismatch for dataset {dataset_name}: "
                f"{len(missing_ids)} train_source_ids from "
                f"{split_path if split_path is not None else 'in-memory config'} "
                "not found in sources loaded from "
                f"{source_file_path}."
            )

        train_ids = set(train_source_ids)
        num_train_sources = len(train_ids)
        num_total_sources = len(all_source_ids)
        num_test_sources = max(num_total_sources - num_train_sources, 0)
        print(
            f"[PrepareData] Dataset {dataset_name}: total_sources={num_total_sources}, "
            f"train_sources={num_train_sources}, test_sources={num_test_sources}"
        )

        dataset_pairs: list[dict] = []

        # Create source-target pairs, restricted to the per-dataset training pool.
        for source in sources:
            if source.id not in train_ids:
                continue
            if not source.target_ids:
                continue

            # Select one target per source
            if target_selection == "first":
                target_id = source.target_ids[0]
            elif target_selection == "random":
                target_id = random.choice(source.target_ids)
            else:
                raise ValueError(f"Unknown target_selection: {target_selection}")

            if target_id not in targets_dict:
                print(
                    f"[PrepareData] Warning: target_id {target_id} not found for source {source.id}"
                )
                continue

            target = targets_dict[target_id]

            dataset_pairs.append(
                {
                    "source": source,
                    "target": target,
                    "source_dataset": dataset_name,
                }
            )

        pairs_by_dataset[dataset_name] = dataset_pairs

        print(
            f"[PrepareData] Loaded {len(dataset_pairs)} "
            f"pairs from {dataset_name}"
        )

    # If we were in generation mode, persist the global split configuration.
    if not use_existing_split and split_path is not None:
        global_split_payload = {
            "seed": seed,
            "max_train_per_dataset": MAX_TRAIN_PER_DATASET,
            "train_ratio_per_dataset": TRAIN_RATIO_PER_DATASET,
            "datasets": generated_split_datasets,
        }
        try:
            split_path.parent.mkdir(parents=True, exist_ok=True)
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(global_split_payload, f, indent=2)
            print(f"[PrepareData] Wrote global split file to {split_path}")
        except Exception as e:
            raise ValueError(
                f"[PrepareData] Failed to write global split file {split_path}: {e}"
            ) from e
    
    total_pairs = sum(len(v) for v in pairs_by_dataset.values())
    print(f"[PrepareData] Total pairs: {total_pairs}")
    print("[PrepareData] GEPA split mode: stratified_by_dataset")

    # Stratified split by dataset.
    trainset: list[dict] = []
    valset: list[dict] = []

    # Shuffle each dataset bucket before slicing.
    for ds_pairs in pairs_by_dataset.values():
        random.shuffle(ds_pairs)

    if val_size < 1:
        for dataset_name, ds_pairs in pairs_by_dataset.items():
            n_ds = len(ds_pairs)
            val_k = int(val_size * n_ds)
            split_idx = n_ds - val_k
            trainset.extend(ds_pairs[:split_idx])
            valset.extend(ds_pairs[split_idx:])
    else:
        target_val_total = min(int(val_size), total_pairs)
        dataset_sizes = {
            dataset_name: len(ds_pairs) for dataset_name, ds_pairs in pairs_by_dataset.items()
        }

        raw_and_base: dict[str, tuple[float, int]] = {}
        for dataset_name, n_ds in dataset_sizes.items():
            if total_pairs == 0:
                raw = 0.0
            else:
                raw = target_val_total * n_ds / total_pairs
            base = int(raw)
            raw_and_base[dataset_name] = (raw, base)

        val_alloc = {dataset_name: base for dataset_name, (_, base) in raw_and_base.items()}
        remaining = target_val_total - sum(val_alloc.values())

        # Largest remainder method with stable tiebreak by dataset name.
        remainders = sorted(
            [
                (dataset_name, raw_and_base[dataset_name][0] - raw_and_base[dataset_name][1])
                for dataset_name in raw_and_base
            ],
            key=lambda x: (-x[1], x[0]),
        )
        for i in range(remaining):
            if not remainders:
                break
            ds_name = remainders[i % len(remainders)][0]
            val_alloc[ds_name] += 1

        print(
            f"[PrepareData] Fixed-count val allocation: target_val_total={target_val_total}"
        )
        for dataset_name in sorted(dataset_sizes):
            n_ds = dataset_sizes[dataset_name]
            val_k_ds = min(val_alloc.get(dataset_name, 0), n_ds)
            train_k_ds = n_ds - val_k_ds
            print(
                f"[PrepareData]  - {dataset_name}: total={n_ds}, val={val_k_ds}, train={train_k_ds}"
            )

        for dataset_name, ds_pairs in pairs_by_dataset.items():
            n_ds = len(ds_pairs)
            val_k_ds = min(val_alloc.get(dataset_name, 0), n_ds)
            split_idx = n_ds - val_k_ds
            trainset.extend(ds_pairs[:split_idx])
            valset.extend(ds_pairs[split_idx:])

    # Shuffle merged splits to avoid dataset block ordering.
    random.shuffle(trainset)
    random.shuffle(valset)
    
    print(f"[PrepareData] Split: {len(trainset)} train, {len(valset)} val")
    
    # Print distribution
    train_dist = {}
    val_dist = {}
    for pair in trainset:
        ds = pair['source_dataset']
        train_dist[ds] = train_dist.get(ds, 0) + 1
    for pair in valset:
        ds = pair['source_dataset']
        val_dist[ds] = val_dist.get(ds, 0) + 1
    
    print(f"[PrepareData] Train distribution: {train_dist}")
    print(f"[PrepareData] Val distribution: {val_dist}")
    
    return trainset, valset


def save_prepared_data(
    trainset: list[dict],
    valset: list[dict],
    output_dir: str
):
    """
    Save prepared datasets to files for inspection.
    
    Args:
        trainset: Training set
        valset: Validation set
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metadata only (sources/targets are heavy)
    train_meta = [
        {
            'source_id': p['source'].id,
            'target_id': p['target'].id,
            'source_dataset': p['source_dataset']
        }
        for p in trainset
    ]
    val_meta = [
        {
            'source_id': p['source'].id,
            'target_id': p['target'].id,
            'source_dataset': p['source_dataset']
        }
        for p in valset
    ]
    
    with open(os.path.join(output_dir, 'train_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(train_meta, f, indent=2)
    
    with open(os.path.join(output_dir, 'val_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(val_meta, f, indent=2)
    
    print(f"[PrepareData] Saved metadata to {output_dir}")


if __name__ == "__main__":

    datasets = ["membench-high", "membench-low", "personamem-v2-val", "prefeval-mcq", "alfworld", "fever", "pddl", "babyai", "scienceworld", "GameOf24", "AIME_2024", "AIME_2025", "AIME_2020_2024", "MMLU_Pro_Physics", "MMLU_Pro_Engineering", "MathEquationBalancer", "bigcodebench"]
    source_file_paths = [f"data_collection/train/{ds}/openai/Qwen3-32B/source_input.jsonl" for ds in datasets]
    target_file_paths = [f"data_collection/train/{ds}/openai/Qwen3-32B/target.jsonl" for ds in datasets]
    trainset, valset = prepare_gepa_data(
        datasets, source_file_paths, target_file_paths,
        split_file_path='data_collection/train/global_split_per20.json',
        refresh=True,
        max_train_per_dataset=20
    )
