"""
Data loader for MyMem

Handles loading and saving of source, target, and mapping files.
"""

import json
import os
from typing import Optional, List

from .schemas import SourceExample, InferenceInputExample


class DataLoader:
    """
    Loader for source, target, and mapping JSONL files.
    """
    
    def __init__(
        self,
        source_file: Optional[str] = None,
        target_file: Optional[str] = None,
    ):
        self.source_file = source_file
        self.target_file = target_file
        
        self._sources: dict[int, SourceExample] = {}
        self._targets: dict[int, InferenceInputExample] = {}
    
    def load_sources(self, file_path: Optional[str] = None) -> dict[int, SourceExample]:
        """Load source examples from JSONL file."""
        path = file_path or self.source_file
        if not path:
            raise ValueError("No source file specified")
        
        self._sources = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    example = SourceExample.from_dict(data)
                    self._sources[example.id] = example
        
        print(f"[DataLoader] Loaded {len(self._sources)} source examples from {path}")
        return self._sources
    
    def load_targets(self, file_path: Optional[str] = None) -> dict[int, InferenceInputExample]:
        """Load target examples from JSONL file."""
        path = file_path or self.target_file
        if not path:
            raise ValueError("No target file specified")
        
        self._targets = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    example = InferenceInputExample.from_dict(data)
                    self._targets[example.id] = example
        
        print(f"[DataLoader] Loaded {len(self._targets)} target examples from {path}")
        return self._targets
    
    def load_all(self) -> tuple[dict[int, SourceExample], dict[int, InferenceInputExample]]:
        """Load all data files."""
        sources = self.load_sources() if self.source_file else {}
        targets = self.load_targets() if self.target_file else {}
        return sources, targets
    
    def save_sources(self, sources: list[SourceExample], file_path: str):
        """Save source examples to JSONL file."""
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for source in sources:
                f.write(json.dumps(source.to_dict(), ensure_ascii=False) + '\n')
        print(f"[DataLoader] Saved {len(sources)} source examples to {file_path}")
    
    def save_targets(self, targets: list[InferenceInputExample], file_path: str):
        """Save target examples to JSONL file."""
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for target in targets:
                f.write(json.dumps(target.to_dict(), ensure_ascii=False) + '\n')
        print(f"[DataLoader] Saved {len(targets)} target examples to {file_path}")
    
    @property
    def sources(self) -> dict[int, SourceExample]:
        return self._sources
    
    @property
    def targets(self) -> dict[int, InferenceInputExample]:
        return self._targets
    
    def get_source(self, source_id: int) -> Optional[SourceExample]:
        return self._sources.get(source_id)
    
    def get_target(self, target_id: int) -> Optional[InferenceInputExample]:
        return self._targets.get(target_id)
