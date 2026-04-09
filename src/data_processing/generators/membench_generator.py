"""
Generator for MemBench data format.

Handles: FirstAgentDataHighLevel_multiple_0.json / FirstAgentDataLowLevel_multiple_0.json
         → source_input.jsonl + target.jsonl
保留原字段命名（question而非goal）。
"""

import os
import json
from typing import Optional, List, Tuple
from src.data_processing.schemas import SourceExample, InferenceInputExample
from src.data_processing.generators.base import BaseGenerator
from src.utils import print_example

MEMBENCH_DATASETS = {"membench-high", "membench-low"}


class MemBenchGenerator(BaseGenerator):
    """
    Generate source.jsonl and target.jsonl from MemBench data.
    
    Input formats:
    - HighLevel: {"highlevel": {"movie": [...], "book": [...]}}
    - LowLevel: {"simple": {"roles": [...], ...}}
    
    Each sample contains:
    - tid: task id
    - message_list: conversation turns [{"user": "...", "agent": "..."}]
    - QA: question, answer, choices, ground_truth, etc.
    """
    
    def generate_sources(
        self,
        source_file: str,
        target_file: Optional[str] = None,
        output_data_dir: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Generate sources and targets from MemBench data. 数据自带 one-to-one 映射。
        """
        # Load MemBench data
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"[MemBenchGenerator] Loaded data from {source_file}")
        
        sources = []
        targets = []
        sample_idx = 0
        
        # Detect data type and process
        if 'highlevel' in data:
            # HighLevel data: movie, book, etc.
            print(f"[MemBenchGenerator] Processing HighLevel data")
            for category, samples in data['highlevel'].items():
                print(f"[MemBenchGenerator] Processing category '{category}': {len(samples)} samples")
                for sample in samples:
                    source, target = self._process_sample(sample, sample_idx, category, 'highlevel')
                    if source and target:
                        sources.append(source)
                        targets.append(target)
                        sample_idx += 1
        
        elif 'simple' in data:
            # LowLevel data: roles, etc.
            print(f"[MemBenchGenerator] Processing LowLevel (simple) data")
            for category, samples in data['simple'].items():
                print(f"[MemBenchGenerator] Processing category '{category}': {len(samples)} samples")
                for sample in samples:
                    source, target = self._process_sample(sample, sample_idx, category, 'simple')
                    if source and target:
                        sources.append(source)
                        targets.append(target)
                        sample_idx += 1
        
        else:
            raise ValueError(f"Unknown MemBench data format. Expected 'highlevel' or 'simple' keys, got: {list(data.keys())}")
        
        if sources:
            print_example("First Source Example", sources[0].to_dict())
        if targets:
            print_example("First Target Example", targets[0].to_dict())
        
        print(f"[MemBenchGenerator] Generated {len(sources)} source examples and {len(targets)} target examples")
        
        # Save to files if output_data_dir is specified（同时保存sources和targets）
        if output_data_dir:
            self.save_outputs(sources, targets, output_data_dir)
    
    def _process_sample(
        self,
        sample: dict,
        sample_idx: int,
        category: str,
        data_level: str
    ) -> tuple[Optional[SourceExample], Optional[InferenceInputExample]]:
        """
        Process a single MemBench sample.
        
        Args:
            sample: Sample dict with tid, message_list, QA
            sample_idx: Global sample index
            category: Category name (movie, book, roles, etc.)
            data_level: 'highlevel' or 'simple'
        
        Returns:
            Tuple of (SourceExample, InferenceInputExample) or (None, None) if invalid
        """
        try:
            tid = sample.get('tid')
            message_list = sample.get('message_list', [])
            qa_data = sample.get('QA', {})
            
            if not message_list or not qa_data:
                print(f"[Warning] Sample tid={tid} missing message_list or QA. Skipping.")
                return None, None
            
            # Convert message_list to standard messages format
            messages = []
            for turn in message_list:
                # Add user message
                if 'user' in turn:
                    messages.append({
                        'role': 'user',
                        'content': turn['user']
                    })
                # Add agent message
                if 'agent' in turn:
                    messages.append({
                        'role': 'assistant',
                        'content': turn['agent']
                    })
            
            # Extract QA information
            qid = qa_data.get('qid')
            question = qa_data.get('question', '')
            answer = qa_data.get('answer', '')
            choices = qa_data.get('choices', {})
            ground_truth = qa_data.get('ground_truth', '')
            target_step_id = qa_data.get('target_step_id', [])
            time = qa_data.get('time', '')
            
            # Create SourceExample
            source = SourceExample(
                source='membench',
                id=sample_idx,
                task_main=question,  # task_main 使用 question
                messages=messages,
                target_ids=[sample_idx],  # Self-referential
                memory="",  # Will be filled during extraction
                metadata={
                    'data_level': data_level,  # highlevel or simple
                    'category': category,  # movie, book, roles, etc.
                    'tid': tid,
                    'qid': qid,
                    'target_step_id': target_step_id,
                    'time': time,
                }
            )
            
            # Create InferenceInputExample（保留原字段名question）
            target = InferenceInputExample(
                source='membench',
                id=sample_idx,
                task_main=question,  # task_main 使用 question
                metadata={
                    'data_level': data_level,
                    'category': category,
                    'tid': tid,
                    'qid': qid,
                    'question': question,
                    'answer': answer,
                    'target_step_id': target_step_id,
                    'choices': choices,
                    'ground_truth': ground_truth,
                    'time': time,
                }
            )
            
            return source, target
        
        except Exception as e:
            print(f"[Error] Failed to process sample: {e}")
            return None, None
    