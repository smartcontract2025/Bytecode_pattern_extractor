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
from typing import Dict, List, Tuple, Set, Optional


def parse_cfg(cfg_data: str) -> Tuple[Dict[str, str], List[Tuple[str, str, str]]]:
    """
    Parse CFG string to extract basic blocks and edge information
    :param cfg_data: Text content of the CFG
    :return: Tuple containing:
             - blocks (dict mapping block IDs to instruction strings)
             - edges (list of tuples (src, dst, color))
    :raises: ValueError if input data is invalid or parsing fails
    """
    blocks: Dict[str, str] = {}
    edges: List[Tuple[str, str, str]] = []

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
            except IndexError as ie:
                print(f"[WARNING] Malformed node pattern in CFG (skipping): {match.group(0) if match else 'None'}")
                continue
            except Exception as e:
                print(f"[NODE PARSE ERROR] Error processing node: {e}")
                continue

        # Regular expression to match edges
        edge_pattern = re.compile(r'block_([a-zA-Z0-9]+) -> block_([a-zA-Z0-9]+) \[color=([a-zA-Z]+)\]')
        for match in edge_pattern.finditer(cfg_data):
            try:
                src = match.group(1)
                dst = match.group(2)
                color = match.group(3)
                if src and dst and color:  # Only add valid edges
                    edges.append((src, dst, color))
            except IndexError as ie:
                print(f"[WARNING] Malformed edge pattern in CFG (skipping): {match.group(0) if match else 'None'}")
                continue
            except Exception as e:
                print(f"[EDGE PARSE ERROR] Error processing edge: {e}")
                continue

    except re.error as re_err:
        raise ValueError(f"Regex compilation error: {re_err}")
    except Exception as e:
        raise ValueError(f"Unexpected error during CFG parsing: {e}")

    if not blocks:
        raise ValueError("No valid basic blocks found in CFG data")

    return blocks, edges


def select_reentrancy_nodes(blocks: Dict[str, str], edges: List[Tuple[str, str, str]]) -> List[int]:
    """
    Extract target basic block indices for reentrancy vulnerability (three sub-patterns):
      1. callValueInvocation: Blocks containing 'CALLVALUE' instruction
      2. balanceDeduction: Direct successors of CALLVALUE blocks containing 'SSTORE'
      3. enoughBalance: Direct predecessors of CALLVALUE blocks containing comparison
                       instructions ('LT', 'GT', 'EQ', 'ISZERO') or 'JUMPI'

    :param blocks: Dictionary { block_id: instruction_string }
    :param edges: List of tuples [(src_block_id, dst_block_id, color), ...]
    :return: List of indices of these target blocks in blocks.keys() list
             (deduplicated and sorted)
    :raises: TypeError if input types are invalid
    """
    try:
        # Input validation
        if not isinstance(blocks, dict) or not isinstance(edges, list):
            raise TypeError("blocks must be dict and edges must be list")

        # 1. Basic block list and index mapping
        block_ids = list(blocks.keys())
        if not block_ids:
            return []

        idx_of = {bid: i for i, bid in enumerate(block_ids)}

        # 2. Build successor and predecessor adjacency lists with validation
        successors: Dict[str, List[str]] = {bid: [] for bid in block_ids}
        predecessors: Dict[str, List[str]] = {bid: [] for bid in block_ids}

        for edge in edges:
            try:
                src, dst, _ = edge
                if src in successors and dst in predecessors:
                    successors[src].append(dst)
                    predecessors[dst].append(src)
            except (ValueError, TypeError) as ve:
                print(f"[EDGE VALIDATION ERROR] Invalid edge format (skipping): {edge}")
                continue
            except KeyError as ke:
                print(f"[EDGE KEY ERROR] Invalid block ID in edge (skipping): {ke}")
                continue

        # 3. Helper functions for matching sub-patterns

        def contains_callvalue(instr: str) -> bool:
            """Check if instruction contains CALLVALUE"""
            try:
                return bool(re.search(r"\bCALLVALUE\b", instr)) if instr else False
            except re.error as re_err:
                print(f"[REGEX ERROR] in contains_callvalue: {re_err}")
                return False
            except Exception as e:
                print(f"[CALLVALUE CHECK ERROR]: {e}")
                return False

        def successor_has_sstore(succ_block: str) -> bool:
            """Check if successor block contains SSTORE"""
            try:
                instr = blocks.get(succ_block, "")
                return bool(re.search(r"\bSSTORE\b", instr)) if instr else False
            except re.error as re_err:
                print(f"[REGEX ERROR] in successor_has_sstore: {re_err}")
                return False
            except Exception as e:
                print(f"[SSTORE CHECK ERROR]: {e}")
                return False

        def predecessor_has_balance_check(pred_block: str) -> bool:
            """Check if predecessor has balance check instructions"""
            try:
                instr = blocks.get(pred_block, "")
                return bool(re.search(r"\b(LT|GT|EQ|ISZERO|JUMPI)\b", instr)) if instr else False
            except re.error as re_err:
                print(f"[REGEX ERROR] in predecessor_has_balance_check: {re_err}")
                return False
            except Exception as e:
                print(f"[BALANCE CHECK ERROR]: {e}")
                return False

        # 4. Find all blocks containing CALLVALUE
        callvalue_blocks: List[str] = []
        for bid, instr in blocks.items():
            try:
                if contains_callvalue(instr):
                    callvalue_blocks.append(bid)
            except Exception as e:
                print(f"[CALLVALUE BLOCK IDENTIFICATION ERROR] for block {bid}: {e}")
                continue

        # 5. Mark target blocks according to sub-patterns
        selected: Set[int] = set()

        # Sub-pattern 1: All blocks with CALLVALUE
        for bid in callvalue_blocks:
            try:
                selected.add(idx_of[bid])
            except KeyError as ke:
                print(f"[INDEX MAPPING ERROR] for block {bid}: {ke}")
                continue

        # Sub-patterns 2 & 3: Check successors and predecessors of CALLVALUE blocks
        for bid in callvalue_blocks:
            try:
                # Sub-pattern 2: Check direct successors
                for succ in successors.get(bid, []):
                    try:
                        if successor_has_sstore(succ):
                            selected.add(idx_of[succ])
                    except Exception as e:
                        print(f"[SUCCESSOR PROCESSING ERROR] for block {bid}->{succ}: {e}")
                        continue

                # Sub-pattern 3: Check direct predecessors
                for pred in predecessors.get(bid, []):
                    try:
                        if predecessor_has_balance_check(pred):
                            selected.add(idx_of[pred])
                    except Exception as e:
                        print(f"[PREDECESSOR PROCESSING ERROR] for block {pred}->{bid}: {e}")
                        continue
            except Exception as e:
                print(f"[NEIGHBOR PROCESSING ERROR] for block {bid}: {e}")
                continue

        # 6. Fallback: Select first block if no matches found
        if not selected and block_ids:
            selected.add(0)

        return sorted(selected)

    except TypeError as te:
        raise TypeError(f"Type error in select_reentrancy_nodes: {te}")
    except Exception as e:
        print(f"[CRITICAL ERROR] in select_reentrancy_nodes: {e}")
        traceback.print_exc()
        return [0] if block_ids else []


def save_graph_data(
        blocks: Dict[str, str],
        edges: List[Tuple[str, str, str]],
        output_path: str,
        target_node_selector=select_reentrancy_nodes
) -> None:
    """
    Save graph structure data to JSON file
    :param blocks: Mapping of block IDs to instruction content
    :param edges: List of edge information tuples
    :param output_path: Output file path
    :param target_node_selector: Function to select target nodes (default: reentrancy detection)
    :raises: IOError if file operations fail, ValueError for invalid data
    """
    try:
        # Input validation
        if not blocks or not isinstance(blocks, dict):
            raise ValueError("Invalid blocks: must be non-empty dictionary")
        if not edges or not isinstance(edges, list):
            raise ValueError("Invalid edges: must be non-empty list")
        if not output_path or not isinstance(output_path, str):
            raise ValueError("Invalid output path")

        block_ids = list(blocks.keys())
        if not block_ids:
            raise ValueError("No blocks available for processing")

        # Create block mapping with validation
        block_map: Dict[str, int] = {}
        try:
            block_map = {block_id: idx for idx, block_id in enumerate(block_ids)}
        except Exception as e:
            raise ValueError(f"Failed to create block mapping: {e}")

        # Convert edge information to index format with validation
        edges_indices: List[Tuple[int, int]] = []
        edge_errors = 0
        for edge in edges:
            try:
                src, dst, _ = edge
                if src in block_map and dst in block_map:
                    edges_indices.append((block_map[src], block_map[dst]))
                else:
                    edge_errors += 1
                    print(f"[EDGE VALIDATION WARNING] Invalid edge - src: {src}, dst: {dst}")
            except Exception as e:
                edge_errors += 1
                print(f"[EDGE PROCESSING ERROR] For edge {edge}: {e}")

        if edge_errors > 0:
            print(f"[WARNING] {edge_errors} edges had errors during processing")

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
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
        except IOError as ioe:
            raise IOError(f"Failed to write to {output_path}: {ioe}")
        except json.JSONEncodeError as je:
            raise ValueError(f"JSON encoding error for {output_path}: {je}")
        except Exception as e:
            raise Exception(f"Unexpected error while saving {output_path}: {e}")

    except Exception as e:
        print(f"[GRAPH SAVE ERROR] Failed to save graph data: {e}")
        traceback.print_exc()
        raise


def process_all_cfg_files(input_dir: str, output_dir: str) -> None:
    """
    Batch process all CFG files in directory to generate graph structure JSONs
    :param input_dir: Input directory containing CFG files
    :param output_dir: Output directory for JSON files
    :raises: FileNotFoundError if directories are invalid
    """
    try:
        # Validate input directory
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)

        # Get file list with validation
        try:
            file_list = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
            if not file_list:
                print("[WARNING] No .txt files found in input directory")
                return
        except PermissionError as pe:
            raise PermissionError(f"Permission denied reading {input_dir}: {pe}")
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
                    try:
                        with open(input_path, 'r', encoding='latin-1') as f:
                            cfg_data = f.read()
                    except Exception as e:
                        raise IOError(f"Failed to read {input_path} with latin-1 encoding: {e}")
                except Exception as e:
                    raise IOError(f"Failed to read {input_path}: {e}")

                # Process CFG data
                blocks, edges = parse_cfg(cfg_data)
                target_nodes = select_reentrancy_nodes(blocks, edges)

                # Track statistics
                target_counts.append(len(target_nodes))
                node_counts.append(len(blocks))

                # Save graph data
                save_graph_data(blocks, edges, output_path)
                processed_files += 1

            except Exception as e:
                error_files += 1
                tqdm.write(f"[ERROR] Failed to process {filename}: {str(e)[:200]}")  # Truncate long error messages
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
                if error_files > 0:
                    print(f"[WARNING] {error_files} files had errors during processing")
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
    except PermissionError as pe:
        print(f"\nPermission denied: {pe}")
    except Exception as e:
        print(f"\n[MAIN ERROR] Fatal error in main execution: {e}")
        traceback.print_exc()
        exit(1)