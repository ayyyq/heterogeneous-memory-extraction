"""
Generator for LongMemEval data format.

Handles: longmemeval_oracle.json → source_input.jsonl + target.jsonl
保留原字段命名（question而非goal）。
"""

import os
import json
from typing import Optional, List, Tuple
from src.data_processing.schemas import SourceExample, InferenceInputExample
from src.data_processing.generators.base import BaseGenerator
from src.utils import print_example


class LongMemEvalGenerator(BaseGenerator):
    """
    Generate source.jsonl and target.jsonl from LongMemEval data.
    
    Input format:
    - longmemeval_oracle.json: List of examples with question, answer, haystack_sessions, etc.
    
    Filter criteria:
    - Only keep examples where len(haystack_session_ids) == 1
    """
    
    def generate_sources(
        self,
        source_file: str,
        target_file: Optional[str] = None,
        output_data_dir: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Generate sources and targets from LongMemEval data. 数据自带 one-to-one 映射。
        """
        # Load LongMemEval data
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"[LongMemEvalGenerator] Loaded {len(data)} total examples")
        
        # Filter: only keep examples where len(haystack_session_ids) == 1
        filtered_data = [
            example for example in data 
            if len(example.get('haystack_session_ids', [])) == 1
        ]
        
        print(f"[LongMemEvalGenerator] Filtered to {len(filtered_data)} examples with single session")
        
        sources = []
        targets = []
        
        for idx, example in enumerate(filtered_data):
            # Extract fields
            question_id = example.get('question_id')
            question_type = example.get('question_type')
            question = example.get('question')
            answer = example.get('answer')
            question_date = example.get('question_date')
            haystack_dates = example.get('haystack_dates', [])
            haystack_session_ids = example.get('haystack_session_ids', [])
            haystack_sessions = example.get('haystack_sessions', [])
            
            # Get the single session
            if not haystack_sessions or len(haystack_sessions) != 1:
                # print(f"[Warning] Example {idx} has {len(haystack_sessions)} sessions, expected 1. Skipping.")
                continue
            
            session = haystack_sessions[0]  # The single conversation
            
            # Find which turn has the answer
            has_answer_turns = []
            for turn_idx, turn in enumerate(session):
                if turn.get('has_answer', False):
                    has_answer_turns.append(turn_idx)
            
            # Create SourceExample
            # id: use filtered index
            # messages: the single conversation
            # target_ids: [id] (self-referential: source i -> target i)
            # metadata: store all other information
            source = SourceExample(
                source='longmemeval',
                id=idx,
                task_main=question,  # task_main 使用 question
                messages=session,  # The single conversation
                target_ids=[idx],  # Self-referential
                memory="",  # Will be filled during extraction
                metadata={
                    'question_id': question_id,
                    'question_type': question_type,
                    'question_date': question_date,
                    'haystack_dates': haystack_dates,
                    'haystack_session_ids': haystack_session_ids,
                    'has_answer_turns': has_answer_turns,
                }
            )
            sources.append(source)
            
            # Create InferenceInputExample（保留原字段名question）
            # id: same as source id
            # metadata: 保留原字段命名
            target = InferenceInputExample(
                source='longmemeval',
                id=idx,
                task_main=question,  # task_main 使用 question
                metadata={
                    'question_id': question_id,
                    'question_type': question_type,
                    'question': question,
                    'answer': answer,
                    'has_answer_turns': has_answer_turns,
                }
            )
            targets.append(target)
        
        print(f"[LongMemEvalGenerator] Generated {len(sources)} source examples and {len(targets)} target examples")
        
        # Print first raw source, first source, and first target for debugging
        if filtered_data:
            print_example("First Raw Source Example", filtered_data[0])
        if sources:
            print_example("First Source Example", sources[0].to_dict())
        if targets:
            print_example("First Target Example", targets[0].to_dict())
        
        # Save to files if output_data_dir is specified（同时保存sources和targets）
        if output_data_dir:
            self.save_outputs(sources, targets, output_data_dir)
