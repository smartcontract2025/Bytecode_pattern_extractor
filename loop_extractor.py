import os
import json
import re
import torch
import random
import dgl
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import traceback


def parse_cfg(cfg_data):
    """
    Parse CFG string to extract basic blocks and edge information
    :param cfg_data: Text content of the CFG
    :return: blocks (dict mapping block IDs to instruction strings),
             edges (list of tuples (src, dst, color))
    :raises: ValueError if input data is invalid
    """
    blocks = {}
    edges = []

    if not cfg_data or not isinstance(cfg_data, str):
        raise ValueError("Invalid CFG data: must be non-empty string")

    try:
        # Regular expression to match nodes (basic blocks)
        node_pattern = re.compile(r'block_([a-zA-Z0-9]+) \[label="([^"]+)"\]')
        for match in node_pattern.finditer(cfg_data):
            try:
                block_id = match.group(1)
                if not block_id:
                    continue
                instructions = match.group(2).replace('\\l', '\n').strip()
                blocks[block_id] = instructions
            except IndexError:
                print(f"[WARNING] Malformed node pattern in CFG: {match.group(0)}")
                continue

        # Regular expression to match edges
        edge_pattern = re.compile(r'block_([a-zA-Z0-9]+) -> block_([a-zA-Z0-9]+) \[color=([a-zA-Z]+)\]')
        for match in edge_pattern.finditer(cfg_data):
            try:
                src = match.group(1)
                dst = match.group(2)
                color = match.group(3)
                if src and dst:  # Only add valid edges
                    edges.append((src, dst, color))
            except IndexError:
                print(f"[WARNING] Malformed edge pattern in CFG: {match.group(0)}")
                continue

    except re.error as e:
        print(f"[REGEX ERROR] Failed to compile regex pattern: {e}")
        raise
    except Exception as e:
        print(f"[PARSE ERROR] Unexpected error during CFG parsing: {e}")
        traceback.print_exc()
        raise

    if not blocks:
        raise ValueError("No valid basic blocks found in CFG data")

    return blocks, edges


def select_target_nodes_infinite_loop(blocks, edges):
    """
    Select potentially suspicious blocks related to infinite loops:
    - loopStatement: Jumps to predecessors or ancestor blocks
    - loopCondition: Always-true conditions or loops without state modification
    - selfInvocation: Unprotected recursive calls

    Returns: List of indices of suspicious blocks (indices in blocks.keys())
    """
    try:
        if not blocks or not isinstance(blocks, dict):
            return []

        block_ids = list(blocks.keys())
        if not block_ids:
            return []

        idx_of = {bid: i for i, bid in enumerate(block_ids)}

        # Build CFG adjacency lists with validation
        successors = {bid: [] for bid in block_ids}
        predecessors = {bid: [] for bid in block_ids}

        for edge in edges:
            try:
                src, dst, _ = edge
                if src in successors and dst in predecessors:  # Validate block IDs
                    successors[src].append(dst)
                    predecessors[dst].append(src)
            except (ValueError, KeyError) as e:
                print(f"[EDGE PROCESSING ERROR] Invalid edge {edge}: {e}")
                continue

        # Recursively compute all ancestors (all predecessors)
        def compute_all_ancestors():
            all_ancestors = {bid: set() for bid in block_ids}

            for node in block_ids:
                try:
                    visited = set()
                    stack = predecessors.get(node, [])[:]  # Handle missing keys
                    while stack:
                        pred = stack.pop()
                        if pred not in visited:
                            visited.add(pred)
                            stack.extend(predecessors.get(pred, []))
                    all_ancestors[node] = visited
                except Exception as e:
                    print(f"[ANCESTOR COMPUTATION ERROR] For node {node}: {e}")
                    continue
            return all_ancestors

        all_ancestors = compute_all_ancestors()

        def is_backward_jump(src, dst):
            """Check if jump goes backward (dst is ancestor of src)"""
            try:
                return dst == src or dst in all_ancestors.get(src, set())
            except Exception as e:
                print(f"[BACKWARD JUMP CHECK ERROR] For src={src}, dst={dst}: {e}")
                return False

        def has_unprotected_call(instr):
            """Check for unconditional calls (CALL/DELEGATECALL/STATICCALL without preceding JUMPI)"""
            try:
                if not instr:
                    return False

                lines = instr.splitlines()
                for i, line in enumerate(lines):
                    try:
                        tokens = line.strip().split()
                        if any(op in tokens for op in ['CALL', 'DELEGATECALL', 'STATICCALL']):
                            # Check for conditional jump before call
                            if not any('JUMPI' in l for l in lines[:i]):
                                return True
                    except Exception as e:
                        print(f"[LINE PROCESSING ERROR] In line {i}: {e}")
                        continue
                return False
            except Exception as e:
                print(f"[CALL CHECK ERROR] Failed to analyze calls: {e}")
                return False

        def is_constant_condition_loop(instr):
            """Check for loops with always-true conditions (JUMPI + constant condition + no state changes)"""
            try:
                if not instr:
                    return False

                lines = instr.splitlines()
                has_jumpi = any('JUMPI' in line for line in lines)
                modifies_state = any(op in instr for op in ['SSTORE', 'CALL', 'DELEGATECALL', 'SELFDESTRUCT'])
                loads_storage = any('SLOAD' in line for line in lines)
                likely_const_cond = any(op in instr for op in ['EQ', 'ISZERO', 'LT', 'GT']) and not loads_storage
                return has_jumpi and not modifies_state and likely_const_cond
            except Exception as e:
                print(f"[CONDITION CHECK ERROR] Failed to analyze conditions: {e}")
                return False

        selected = set()

        for src in block_ids:
            try:
                # Check successors for backward jumps
                for dst in successors.get(src, []):
                    if is_backward_jump(src, dst):
                        selected.add(idx_of[src])

                # Check instruction patterns
                instr = blocks.get(src, "")
                if is_constant_condition_loop(instr):
                    selected.add(idx_of[src])
                if has_unprotected_call(instr):
                    selected.add(idx_of[src])
            except Exception as e:
                print(f"[BLOCK PROCESSING ERROR] For block {src}: {e}")
                continue

        # Fallback: Select first block if no suspicious blocks found
        if not selected and block_ids:
            selected.add(0)

        return sorted(selected)

    except KeyError as ke:
        print(f"[KEY ERROR] Missing block ID in mapping: {ke}")
        return []
    except Exception as e:
        print(f"[TARGET SELECTION ERROR] Unexpected error: {e}")
        traceback.print_exc()
        return [0] if block_ids else []


def save_graph_data(blocks, edges, output_path, target_node_selector=select_target_nodes_infinite_loop):
    """
    Save graph structure data to JSON file
    :param blocks: Mapping of block IDs to instruction content
    :param edges: List of edge information
    :param output_path: Output file path
    :param target_node_selector: Function to select target nodes (default uses infinite loop detection)
    :raises: IOError if file operations fail
    """
    try:
        if not blocks or not edges:
            raise ValueError("Empty blocks or edges provided")

        block_ids = list(blocks.keys())
        if not block_ids:
            raise ValueError("No blocks available for processing")

        # Create block mapping with validation
        block_map = {}
        try:
            block_map = {block_id: idx for idx, block_id in enumerate(block_ids)}
        except Exception as e:
            raise ValueError(f"Failed to create block mapping: {e}")

        # Convert edge information to index format with validation
        edges_indices = []
        for src, dst, _ in edges:
            try:
                if src in block_map and dst in block_map:
                    edges_indices.append((block_map[src], block_map[dst]))
                else:
                    print(f"[EDGE VALIDATION WARNING] Invalid edge - src: {src}, dst: {dst}")
            except Exception as e:
                print(f"[EDGE PROCESSING ERROR] For edge ({src}, {dst}): {e}")

        # Select target nodes with error handling
        try:
            target_nodes = target_node_selector(blocks, edges)
            if not isinstance(target_nodes, list):
                raise TypeError("Target node selector should return a list")
        except Exception as e:
            print(f"[TARGET NODE SELECTION ERROR] {e}")
            target_nodes = []

        # Build graph data dictionary
        graph_data = {
            "nodes": block_ids,
            "edges": edges_indices,
            "target_nodes": target_nodes
        }

        # Validate data consistency
        if len(graph_data["nodes"]) != len(blocks):
            print(f"[DATA CONSISTENCY WARNING] File: {output_path}")
            print(f"Node count mismatch: {len(graph_data['nodes'])} vs {len(blocks)}")

        # Save as JSON file with proper error handling
        try:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
        except IOError as e:
            raise IOError(f"Failed to write to {output_path}: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error while saving {output_path}: {e}")

    except Exception as e:
        print(f"[GRAPH SAVE ERROR] Failed to save graph data: {e}")
        traceback.print_exc()
        raise


def process_all_cfg_files(input_dir, output_dir):
    """
    Batch process all CFG files in directory to generate graph structure JSONs
    :param input_dir: Input directory containing CFG files
    :param output_dir: Output directory for JSON files
    :raises: FileNotFoundError if directories are invalid
    """
    try:
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        os.makedirs(output_dir, exist_ok=True)

        # Get file list with validation
        try:
            file_list = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
            if not file_list:
                print("[WARNING] No .txt files found in input directory")
                return
        except Exception as e:
            raise IOError(f"Failed to read input directory: {e}")

        target_counts = []
        node_counts = []
        processed_files = 0
        error_files = 0

        for filename in tqdm(file_list, desc="Processing CFG files"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".txt", ".json"))

            try:
                # Read input file with error handling
                try:
                    with open(input_path, 'r', encoding='utf-8') as f:
                        cfg_data = f.read()
                except UnicodeDecodeError:
                    with open(input_path, 'r', encoding='latin-1') as f:
                        cfg_data = f.read()
                except Exception as e:
                    raise IOError(f"Failed to read {input_path}: {e}")

                # Process CFG data
                blocks, edges = parse_cfg(cfg_data)
                target_nodes = select_target_nodes_infinite_loop(blocks, edges)

                # Track statistics
                target_counts.append(len(target_nodes))
                node_counts.append(len(blocks))

                # Save graph data
                save_graph_data(blocks, edges, output_path)
                processed_files += 1

            except Exception as e:
                error_files += 1
                tqdm.write(f"[ERROR] Failed to process {filename}: {e}")
                traceback.print_exc()
                continue

        # Print summary statistics
        if target_counts and node_counts:
            try:
                avg_targets = np.mean(target_counts)
                avg_nodes = np.mean(node_counts)
                print(f"\nProcessing complete. Statistics:")
                print(f"- Processed files: {processed_files}")
                print(f"- Files with errors: {error_files}")
                print(f"- Average target nodes per contract: {avg_targets:.2f}")
                print(f"- Average nodes per contract: {avg_nodes:.2f}")
            except Exception as e:
                print(f"[STATS ERROR] Failed to calculate statistics: {e}")
        else:
            print("No valid contracts were processed.")

    except Exception as e:
        print(f"[BATCH PROCESSING ERROR] Fatal error: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        # Configuration with fallback values
        input_folder = "test_input"  # Input directory containing CFG text files
        output_folder = "test_output"  # Output directory for JSON graph files

        # Validate paths before processing
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

        print(f"Starting CFG processing...")
        print(f"Input directory: {os.path.abspath(input_folder)}")
        print(f"Output directory: {os.path.abspath(output_folder)}")

        process_all_cfg_files(input_folder, output_folder)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"[MAIN ERROR] Fatal error in main execution: {e}")
        traceback.print_exc()