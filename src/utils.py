"""
Utility functions for MyMem
"""

import json
import os
import sys
from typing import List, Dict, Any

from src.data_processing.schemas import SourceExample, InferenceInputExample


def format_example_value(value, indent_level: int = 0) -> str:
    """
    Format a value (dict/list/scalar) for pretty-printing in debug output.

    This is shared between generators, extractors, and runners so that
    examples look consistent everywhere.
    """
    indent = "  " * indent_level

    # Lists: show at most 2 items, pretty-printed
    if isinstance(value, list):
        if len(value) == 0:
            return "[]"
        items_to_show = value[:2]
        formatted_items: list[str] = []
        for item in items_to_show:
            formatted_item = format_example_value(item, indent_level + 1)
            # If item is a dict (multi-line), add proper indentation
            if isinstance(item, dict) and "\n" in formatted_item:
                indented_lines: list[str] = []
                for line in formatted_item.split("\n"):
                    indented_lines.append(f"{indent}  {line}")
                formatted_items.append("\n".join(indented_lines))
            else:
                formatted_items.append(formatted_item)

        result_lines = ["["]
        for i, formatted_item in enumerate(formatted_items):
            line = f"{indent}  {formatted_item}"
            if i < len(formatted_items) - 1:
                line += ","
            result_lines.append(line)
        if len(value) > 2:
            result_lines.append(f"{indent}  ... (total {len(value)} items)")
        result_lines.append(f"{indent}]")
        return "\n".join(result_lines)

    # Dicts: show at most 5 keys, pretty-printed
    if isinstance(value, dict):
        if len(value) == 0:
            return "{}"
        lines = ["{"]
        items_shown = 0
        for k, v in value.items():
            if items_shown >= 5:
                break
            formatted_v = format_example_value(v, indent_level + 1)
            if "\n" in formatted_v:
                lines.append(f"{indent}  '{k}':")
                for line in formatted_v.split("\n"):
                    lines.append(f"{indent}    {line}")
            else:
                lines.append(f"{indent}  '{k}': {formatted_v}")
            items_shown += 1
        if len(value) > 5:
            lines.append(f"{indent}  ... (total {len(value)} keys)")
        lines.append(f"{indent}}}")
        return "\n".join(lines)

    # Scalars: use repr for strings (with truncation), str for others
    if isinstance(value, str):
        if len(value) > 200:
            return repr(value[:200] + "...")
        return repr(value)
    return repr(value)


def print_example(title: str, example_dict: Dict[str, Any]) -> None:
    """
    Pretty-print a single example (e.g., SourceExample.to_dict()).

    Used for debugging in generators, extractors, and the main runner.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for key, value in example_dict.items():
        formatted_value = format_example_value(value)
        print(f"{key}: {formatted_value}")
    print("=" * 60 + "\n")


def print_messages(messages: List[dict]) -> None:
    """
    Print messages in the form [role]: content.
    Each message is a dict with 'role' and 'content' keys.
    """
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        print(f"[{role.upper()}]: {content}")

def filter_successful_sources(
    sources: List[SourceExample],
    success_only: bool = True,
) -> List[SourceExample]:
    """
    Filter sources by success label.
    
    Args:
        sources: List of source examples
        success_only: If True, keep only successful ones
        
    Returns:
        Filtered list of sources
    """
    if success_only:
        return [s for s in sources if s.label]
    return sources


def group_by_env_name(
    items: List[Any],
    get_env_name_func,
) -> Dict[str, List[Any]]:
    """
    Group items by environment name.
    
    Args:
        items: List of items to group
        get_env_name_func: Function to extract env_name from item
        
    Returns:
        Dict mapping env_name to list of items
    """
    groups: Dict[str, List[Any]] = {}
    for item in items:
        env_name = get_env_name_func(item) or 'unknown'
        if env_name not in groups:
            groups[env_name] = []
        groups[env_name].append(item)
    return groups


# CLI for utility functions
if __name__ == '__main__':
    # Suppress RuntimeWarning: if src.utils was already imported, reload it
    import importlib
    if 'src.utils' in sys.modules:
        importlib.reload(sys.modules['src.utils'])
    
    import argparse
    
    parser = argparse.ArgumentParser(description='MyMem Utilities')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Convert to source.jsonl (unified interface)
    convert_source_parser = subparsers.add_parser('convert-source',
                                                   help='Convert trajectories to source.jsonl and mapping.jsonl')
    convert_source_parser.add_argument('--trajectories', required=True,
                                       help='Input trajectories.json')
    convert_source_parser.add_argument('--output-dir', type=str, required=True,
                                       help='Output directory (will create source.jsonl and mapping.jsonl)')
    convert_source_parser.add_argument('--dataset', default='alfworld',
                                       help='Dataset name')
    convert_source_parser.add_argument('--generator', default='auto',
                                       choices=['collector', 'direct', 'auto'],
                                       help='Generator type')
    convert_source_parser.add_argument('--clusters', type=str, default=None,
                                       help='Path to clusters.json (optional)')
    convert_source_parser.add_argument('--no-clusters', action='store_true',
                                       help='Ignore clusters even if provided')
    convert_source_parser.add_argument('--cluster-method', default='finch',
                                       choices=['finch', 'type'],
                                       help='Cluster method (if using clusters)')
    convert_source_parser.add_argument('--success-only', action='store_true',
                                       help='Only include successful trajectories')
    convert_source_parser.add_argument('--max-targets', type=int, default=20,
                                       help='Maximum number of targets per source (default: 20)')
    convert_source_parser.add_argument('--max-sources', type=int, default=-1,
                                       help='Maximum number of sources to sample (if > 0, randomly sample m sources; default: -1 means no limit)')
    
    # Convert collector output (legacy, for backward compatibility)
    convert_parser = subparsers.add_parser('convert-collector',
                                           help='[Legacy] Convert collector output to source.jsonl and mapping.jsonl')
    convert_parser.add_argument('--input', required=True, help='Input trajectories.json')
    convert_parser.add_argument('--output-dir', required=True, help='Output directory')
    convert_parser.add_argument('--dataset', default='alfworld', help='Dataset name')
    convert_parser.add_argument('--clusters', type=str, default=None,
                                help='Path to clusters.json')
    convert_parser.add_argument('--max-targets', type=int, default=20,
                                help='Maximum number of targets per source')
    
    # Convert alfworld tasks
    alfworld_parser = subparsers.add_parser('convert-alfworld',
                                            help='Convert alfworld tasks to target.jsonl')
    alfworld_parser.add_argument('--input', required=True, help='Input alfworld_tasks.json')
    alfworld_parser.add_argument('--output', required=True, help='Output target.jsonl')
    alfworld_parser.add_argument('--limit', type=int, default=-1,
                                 help='Limit number of tasks (-1 for all)')
    
    args = parser.parse_args()
    
    if args.command == 'convert-source':
        # Delay import to avoid module import warnings
        import sys
        if 'src.data_processing.generators.factory' not in sys.modules:
            from src.data_processing.generators.factory import create_generator
        else:
            from src.data_processing.generators import factory
            create_generator = factory.create_generator
        
        generator = create_generator(args.generator, dataset_name=args.dataset)
        sources = generator.generate(
            trajectories_file=args.trajectories,
            clusters_file=args.clusters,
            output_data_dir=args.output_dir,
            use_clusters=not args.no_clusters and args.clusters is not None,
            cluster_method=args.cluster_method,
            filter_success_only=args.success_only,
            max_targets_per_source=args.max_targets,
            max_sources=args.max_sources,
        )
        num_mappings = sum(len(s.target_ids) for s in sources)
        print(f"\n✓ Generated {len(sources)} sources with {num_mappings} total target mappings")
        
    # elif args.command == 'convert-collector':
    #     sources, mappings = convert_collector_output_to_source(
    #         args.input, args.output_dir, args.dataset,
    #         clusters_file=args.clusters,
    #         use_clusters=args.clusters is not None,
    #         max_targets_per_source=args.max_targets,
    #     )
    #     print(f"\n✓ Generated {len(sources)} sources and {len(mappings)} mappings")
        
    # elif args.command == 'convert-alfworld':
    #     convert_alfworld_tasks_to_target(args.input, args.output, limit=args.limit)
        
    else:
        parser.print_help()

