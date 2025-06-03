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


def select_target_nodes_advanced(blocks: Dict[str, str], edges: List[Tuple[str, str, str]]) -> List[int]:
    """
    Advanced target node selection based on timestamp patterns:
    1) Find all blocks containing TIMESTAMP or BLOCKHASH (Pattern 1)
    2) For each such block, perform BFS traversal of all its successors:
       - For each successor block, check Pattern 2 and Pattern 3 regardless of parent selection
       - If matching assign_ops or contamination_ops, mark the block as target
    3) If no Pattern 1 matches found, mark the first block as fallback

    Returns list of indices of these target blocks in blocks list
    """
    try:
        # Input validation
        if not isinstance(blocks, dict) or not isinstance(edges, list):
            raise TypeError("blocks must be dict and edges must be list")

        # Operation patterns to detect
        timestamp_ops = {"TIMESTAMP", "BLOCKHASH"}
        assign_ops = {
            "SSTORE", "MSTORE", "CALL", "DELEGATECALL", "STATICCALL",
            "LOG0", "LOG1", "LOG2", "LOG3", "LOG4",
            "CREATE", "CREATE2", "RETURN", "REVERT",
            "PUSH1", "PUSH2", "PUSH3", "PUSH4", "PUSH32",
            "SWAP1", "DUP1", "DUP2"
        }
        contamination_ops = {
            "GT", "EQ", "LT", "ISZERO", "JUMPI",
            "AND", "OR", "XOR", "NOT", "SHL", "SHR",
            "ADD", "SUB", "MUL", "DIV", "MOD", "EXP",
            "CALLVALUE", "BALANCE", "EXTCODESIZE"
        }

        # Create block list and index mapping
        block_ids = list(blocks.keys())
        if not block_ids:
            return []

        idx_of = {bid: i for i, bid in enumerate(block_ids)}

        # Build successor adjacency list with validation
        successors: Dict[str, List[str]] = {bid: [] for bid in block_ids}
        for edge in edges:
            try:
                src, dst, _ = edge
                if src in successors and dst in idx_of:  # Validate block IDs
                    successors[src].append(dst)
            except (ValueError, TypeError) as ve:
                print(f"[EDGE VALIDATION ERROR] Invalid edge format (skipping): {edge}")
                continue
            except KeyError as ke:
                print(f"[EDGE KEY ERROR] Invalid block ID in edge (skipping): {ke}")
                continue

        selected: Set[int] = set()

        # Find all blocks containing timestamp operations (Pattern 1)
        timestamp_blocks: List[str] = []
        for bid, instr in blocks.items():
            try:
                if any(op in instr for op in timestamp_ops):
                    timestamp_blocks.append(bid)
                    selected.add(idx_of[bid])  # Mark Pattern 1 blocks
            except Exception as e:
                print(f"[TIMESTAMP BLOCK IDENTIFICATION ERROR] for block {bid}: {e}")
                continue

        # BFS traversal for each timestamp block
        for start in timestamp_blocks:
            try:
                queue = [start]
                seen = set(queue)

                while queue:
                    cur = queue.pop(0)
                    for neighbor in successors.get(cur, []):
                        try:
                            # Add to BFS queue if not seen
                            if neighbor not in seen:
                                seen.add(neighbor)
                                queue.append(neighbor)

                            # Check for Pattern 2/3 in each successor
                            instr = blocks.get(neighbor, "")
                            try:
                                ops = set(re.findall(r"\b([A-Z]+[0-9]*)\b", instr))
                                if (assign_ops & ops) or (contamination_ops & ops):
                                    selected.add(idx_of[neighbor])
                            except re.error as re_err:
                                print(f"[REGEX ERROR] in operation extraction for block {neighbor}: {re_err}")
                                continue
                        except Exception as e:
                            print(f"[NEIGHBOR PROCESSING ERROR] for block {neighbor}: {e}")
                            continue
            except Exception as e:
                print(f"[BFS TRAVERSAL ERROR] for start block {start}: {e}")
                continue

        # Fallback: Select first block if no matches found
        if not selected and block_ids:
            selected.add(0)

        return sorted(selected)

    except TypeError as te:
        raise TypeError(f"Type error in select_target_nodes_advanced: {te}")
    except Exception as e:
        print(f"[CRITICAL ERROR] in select_target_nodes_advanced: {e}")
        traceback.print_exc()
        return [0] if block_ids else []


def save_graph_data(
        blocks: Dict[str, str],
        edges: List[Tuple[str, str, str]],
        output_path: str,
        target_node_selector=select_target_nodes_advanced
) -> None:
    """
    Save graph structure data to JSON file
    :param blocks: Mapping of block IDs to instruction content
    :param edges: List of edge information tuples
    :param output_path: Output file path
    :param target_node_selector: Function to select target nodes (default: timestamp pattern detection)
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
                target_nodes = select_target_nodes_advanced(blocks, edges)

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
        input_folder = "opcode"  # Input directory containing CFG text files
        output_folder = "opcode_graph_bert_cls"  # Output directory for JSON graph files

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