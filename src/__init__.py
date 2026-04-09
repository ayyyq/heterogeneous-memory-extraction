"""
MyMem - Memory Extraction and Injection Module

This module provides functionality to:
1. Extract memory from source trajectories
2. Inject memory into target task prompts
3. Run single agent inference with memory augmentation
4. Evaluate per-source accuracy
"""

"""
Legacy src package exports (kept for backward compatibility where possible).

NOTE: MappingEntry and old src.data APIs have been removed in favor of
`src.data_processing.schemas` and pipeline-specific loaders/generators.
"""

from src.data_processing.schemas import SourceExample, InferenceInputExample
from src.data_processing.loader import DataLoader

__all__ = [
    "SourceExample",
    "InferenceInputExample",
    "DataLoader",
]

