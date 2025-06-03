# Smart Contract Bytecode Graph Constructor and expert pattern extrctor

A tool for constructing graphs from smart contract bytecode (opcodes) and annotating nodes using specially designed expert patterns. The designed expert rules follow and extend the source code based approaches from:

1. [**IJCAI 2021**] [Smart Contract Vulnerability Detection: From Pure Neural Network to Interpretable Graph Feature and Expert Pattern Fusion](https://arxiv.org/abs/2106.09282)  
2. [**TKDE 2021**] [Combining Graph Neural Networks with Expert Knowledge for Smart Contract Vulnerability Detection](https://arxiv.org/abs/2107.11598)

## Features

- **Bytecode-to-Graph Conversion**: Constructs control flow graphs (CFGs) directly from Ethereum bytecode
- **Expert Pattern Annotation**: Implements specialized vulnerability detection patterns for:
  - Timestamp dependency
  - Reentrancy vulnerabilities
  - Infinite loops


## Installation

```bash
git clone https://github.com/yourusername/smart-contract-graph-constructor.git
cd smart-contract-graph-constructor
pip install -r requirements.txt
```

## Expert Pattern Implementation

Our implementation extends the expert patterns from the cited papers to work directly with bytecode:

Timestamp Dependency Pattern
Detects blocks containing TIMESTAMP or BLOCKHASH opcodes and their data-dependent successors

Reentrancy Pattern
Identifies dangerous call sequences following the checks-effects-interactions pattern

Infinite Loop Pattern
Detects potential unbounded loops through jump analysis

Output Format
Generated graphs follow this JSON structure:

{

  "nodes": ["block_1", "block_2", ...],
  
  "edges": [[0, 1], [1, 2], ...],
  
  "target_nodes": [3, 5, ...]
  
}

更多细节请参阅我们的论文原文和代码...
