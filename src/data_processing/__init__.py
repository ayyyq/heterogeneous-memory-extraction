"""
Data module for MyMem

Provides data schemas, loaders, and generators for source, target, and mapping files.
"""

from .schemas import SourceExample, InferenceInputExample
from .loader import DataLoader
from .generators.base import BaseGenerator
from .generators.factory import create_generator, detect_generator_type

__all__ = [
    "SourceExample",
    "InferenceInputExample",
    "DataLoader",
    "BaseGenerator",
    "create_generator",
    "detect_generator_type",
]

