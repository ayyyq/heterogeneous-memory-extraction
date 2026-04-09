"""
Data schemas for MyMem

Defines the structure of source, target, and evaluation result data.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SourceExample:
    """
    A source example containing trajectory data for memory extraction.
    
    Attributes:
        source: Dataset name (e.g., 'alfworld')
        id: Unique identifier for this source example
        messages: Multi-turn conversation messages
        metadata: Additional information (task_main, task_description, label, num_turns, etc.)
    """
    source: str
    id: int
    task_main: str
    messages: list[dict]  # [{'role': 'system'/'user'/'assistant', 'content': '...'}]
    target_ids: list[int] = field(default_factory=list)
    memory: str = ""
    metadata: dict = field(default_factory=dict)

    def get_conversation_text(self) -> str:
        """
        Convert messages to a readable conversation text.
        """
        lines = []
        for msg in self.messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if role == 'system':
                lines.append(f"<system>{content}</system>")
            elif role == 'user':
                lines.append(f"<user>{content}</user>")
            elif role == 'assistant':
                lines.append(f"<assistant>{content}</assistant>")
            else:
                raise ValueError(f'Invalid role: {role}')
        return '\n'.join(lines)
    
    def get_trajectory_text(self) -> str:
        """Convert messages to a readable trajectory text."""
        lines = []
        for msg in self.messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if role == 'system':
                continue  # Skip system message in trajectory
            elif role == 'assistant':
                lines.append(f"Action: {content}")
            elif role == 'user':
                lines.append(f"Observation: {content}")
        return '\n'.join(lines)
    
    def to_dict(self) -> dict:
        return {
            'source': self.source,
            'id': self.id,
            'task_main': self.task_main,
            'messages': self.messages,
            'target_ids': self.target_ids,
            'memory': self.memory,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SourceExample':
        return cls(
            source=data.get('source', 'unknown'),
            id=data.get('id', 0),
            task_main=data.get('task_main', ''),
            messages=data.get('messages', []),
            target_ids=data.get('target_ids', []),
            memory=data.get('memory', ''),
            metadata=data.get('metadata', {})
        )


@dataclass
class InferenceInputExample:
    """
    极简化的Target定义，适用于所有数据集。
    
    核心字段:
        source: 数据集名称（与SourceExample保持一致，如'alfworld'/'fever'/'longmemeval'）
        id: 唯一标识符（必须与SourceExample.target_ids对应）
        metadata: 所有数据集特定信息（完全自由的dict）
    
    设计原则:
        1. id必须保证与SourceExample.target_ids一一对应
        2. 不同数据集差别过大，全部信息放入metadata
        3. 序列化时metadata内容平铺到顶层，便于访问
    
    """
    source: str
    id: int
    task_main: str
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """序列化时平铺metadata到顶层"""
        result = {
            'source': self.source,
            'id': self.id,
            'task_main': self.task_main,
        }
        result.update(self.metadata)  # 平铺所有metadata字段
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> 'InferenceInputExample':
        """反序列化时自动识别metadata"""
        source = data.get('source', 'unknown')
        id_val = data.get('id', 0)
        task_main = data.get('task_main', '')
        
        # 除了source、id和task_main，其他全部放入metadata
        metadata = {
            k: v for k, v in data.items() 
            if k not in ['source', 'id', 'task_main']
        }
        
        return cls(
            source=source,
            id=id_val,
            task_main=task_main,
            metadata=metadata
        )
    
    def get(self, key: str, default=None):
        """便捷访问metadata字段"""
        return self.metadata.get(key, default)

    def to_task_config(self) -> dict:
        """
        转换为 task config 格式（供 GMemory tasks.envs 与 inference adapter 使用）。
        使用 metadata 副本，并确保含 'task' 字段。
        """
        config = dict(self.metadata)
        config.setdefault(
            "task",
            self.task_main
        )
        return config


@dataclass
class InferenceOutputExample:
    """
    Unified inference output example.
    Contains source, id, task_main, metadata, messages and label; serialization is self-contained.
    """

    source: str
    id: int
    task_main: str
    messages: list[dict] = field(default_factory=list)
    label: Optional[bool] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {
            "source": self.source,
            "id": self.id,
            "task_main": self.task_main,
            "messages": self.messages,
            "label": self.label,
        }
        result.update(self.metadata)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "InferenceOutputExample":
        source = data.get("source", "unknown")
        id_val = data.get("id", 0)
        task_main = data.get("task_main", "")
        messages = data.get("messages", [])
        label = data.get("label")
        metadata = {
            k: v for k, v in data.items()
            if k not in ["source", "id", "task_main", "messages", "label"]
        }
        return cls(
            source=source,
            id=id_val,
            task_main=task_main,
            messages=messages,
            label=label,
            metadata=metadata,
        )

    def get_conversation_text(self) -> str:
        """
        Convert messages to a readable conversation text.
        """
        lines = []
        for msg in self.messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if role == 'system':
                lines.append(f"<system>{content}</system>")
            elif role == 'user':
                lines.append(f"<user>{content}</user>")
            elif role == 'assistant':
                lines.append(f"<assistant>{content}</assistant>")
            else:
                raise ValueError(f'Invalid role: {role}')
        return '\n'.join(lines)