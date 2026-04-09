"""
evolution_ace: ACE-based evolution of memory extraction prompts.

Uses ACE's three-role architecture (Generator/Extractor, Reflector, Curator)
to evolve the memory extraction playbook from source-target evaluation feedback.
"""

from .ace_system import ACE
from .data_processor import MemoryExtractionDataProcessor
from .playbook import seed_playbook_from_factual_experiential, render_playbook_to_system_prompt

__all__ = [
    "ACE",
    "MemoryExtractionDataProcessor",
    "seed_playbook_from_factual_experiential",
    "render_playbook_to_system_prompt",
]
