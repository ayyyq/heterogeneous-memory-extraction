"""
Mapping strategies for linking SourceExample and InferenceInputExample.

当前目标：
- 将原本散落在各个 Generator 内的映射逻辑（尤其是基于相似度的映射）抽离为纯函数工具，
  方便在不同数据集 / pipeline 中复用。

注意：
- 本模块只负责在内存中修改 `sources` 的 `target_ids` 字段，不负责落盘。
"""

from __future__ import annotations

from typing import List, Dict, Callable, Optional

import numpy as np

from .schemas import SourceExample, InferenceInputExample


def map_by_similarity(
    sources: List[SourceExample],
    targets: List[InferenceInputExample],
    embedding_func: Optional[Callable[[str], list[float]]] = None,
    source_embeddings: Optional[Dict[int, np.ndarray]] = None,
    target_embeddings: Optional[Dict[int, np.ndarray]] = None,
    max_targets_per_source: int = 5,
    threshold: float = 0.0,
) -> None:
    """
    基于任务语义相似度的 one-to-many 映射。

    支持两种用法（二选一）：
    - 模式 1（GMemory）：仅传 embedding_func，内部用 source.task_main / target.task_main 现场编码再算相似度。
    - 模式 2（DC 等）：同时传 source_embeddings 与 target_embeddings（key 为 source.id / target.id），
      直接使用预计算向量算相似度，不调用 embedding_func。

    公共逻辑：对每个 source 与所有 target 算余弦相似度，按相似度降序取 top-k、过滤 threshold，写回 source.target_ids。

    Args:
        sources: 源示例列表（会就地修改其 target_ids）
        targets: 推理输入示例列表
        embedding_func: 文本 -> 向量的函数（模式 1 必填）；支持 callable 或带 embed_query 的对象
        source_embeddings: 预计算的 source 向量，key 为 source.id（模式 2 与 target_embeddings 同时提供时使用）
        target_embeddings: 预计算的 target 向量，key 为 target.id（模式 2 与 source_embeddings 同时提供时使用）
        max_targets_per_source: 每个 source 允许映射的最多 target 数量
        threshold: 相似度阈值，只有 similarity >= threshold 的映射才会被保留（默认 0.0，即保留所有）
    """
    if not sources or not targets:
        for s in sources:
            s.target_ids = []
        return

    use_precomputed = source_embeddings is not None and target_embeddings is not None
    if use_precomputed:
        # 模式 2：直接使用预计算 embedding，不再计算
        pass
    else:
        if embedding_func is None:
            raise ValueError(
                "[mapping_strategies.map_by_similarity] Either provide embedding_func, "
                "or both source_embeddings and target_embeddings."
            )
        # 模式 1：用 embedding_func 从 task_main 计算 embedding
        def _get_embedding(text: str) -> list[float]:
            if hasattr(embedding_func, "embed_query"):
                return embedding_func.embed_query(text)
            return embedding_func(text)

        print("[mapping_strategies] Computing embeddings for sources...")
        source_embeddings = {}
        for source in sources:
            if not source.task_main:
                continue
            source_embeddings[source.id] = np.array(_get_embedding(source.task_main), dtype=float)
        print("[mapping_strategies] Computing embeddings for targets...")
        target_embeddings = {}
        for target in targets:
            if not target.task_main:
                continue
            target_embeddings[target.id] = np.array(_get_embedding(target.task_main), dtype=float)

    if not source_embeddings or not target_embeddings:
        print("[mapping_strategies] No valid embeddings for sources or targets, clearing target_ids.")
        for s in sources:
            s.target_ids = []
        return

    # 为每个 source 选择最相似的 targets
    target_ids = list(target_embeddings.keys())
    for source in sources:
        if source.id not in source_embeddings:
            source.target_ids = []
            continue

        src_emb = source_embeddings[source.id]

        sims: list[tuple[int, float]] = []
        for tid in target_ids:
            if tid == source.id:
                # 跳过自己
                continue
            tgt_emb = target_embeddings[tid]
            denom = np.linalg.norm(src_emb) * np.linalg.norm(tgt_emb) + 1e-10
            sim = float(np.dot(src_emb, tgt_emb) / denom)
            sims.append((tid, sim))

        # 相似度降序排序，过滤掉低于 threshold 的，取 top-k
        sims.sort(key=lambda x: x[1], reverse=True)
        filtered_sims = [(tid, sim) for tid, sim in sims if sim >= threshold]
        top_ids = [tid for tid, _ in filtered_sims[: max_targets_per_source]]
        source.target_ids = top_ids

    total_mappings = sum(len(s.target_ids) for s in sources)
    print(
        f"[mapping_strategies] Finished similarity-based mapping for "
        f"{len(sources)} sources and {len(target_embeddings)} targets "
        f"(threshold={threshold}, total_mappings={total_mappings})."
    )

