# PyGO: KataGo Neural Network in PyTorch

**Scope:**  
This repo contains a modular PyTorch implementation of the *neural network model* used in KataGo.  
- **What’s implemented:** Neural network architecture only (`model.py`), including all trunk, residual, and head structures per KataGo’s design.  
- **What’s *not* implemented:** Monte Carlo Tree Search (MCTS), Go-specific feature extraction, GTP engine, distributed training, or C++ performance optimizations.

---

## Purpose

This repository defines a modular, research-ready neural network architecture for Go, based on KataGo.  
The long-term objective is to enable:
- **Self-play training** pipelines and performance benchmarking  
- **Integration** with GTP-compatible engines and Monte Carlo Tree Search  
- **Experimental extensions** (e.g., opponent modeling, human-like play, uncertainty estimation, interpretability research)

Agents should not only refactor and document code, but also:
1. **Diagnose architectural gaps** and trade-offs  
2. **Propose experimental modifications** to improve performance or add new capabilities  
3. **Evaluate output semantics** against design intent and success metrics (e.g., win-rate, score accuracy)  
4. **Prepare the model** for integration into broader gameplay and training systems

---

## Environment Setup

Python 3.8+ and PyTorch are required.  
Recommended for local and agent use:

```bash
pip install torch
# (optional) pip install pytest black flake8
````

---

## Directory Layout

* **`model.py`** – Core model: `InitialBlock`, residual blocks, policy/value heads, config class.
* **`README.md` / `AGENTS.md`** – Project overview, objectives, and agent guidelines.
* **`tests/`** (optional) – Unit tests (to be generated or expanded by agents).

---

## Quick Example

```python
from model import KataGoModel, ModelConfig
import torch

# ModelConfig is a frozen dataclass describing architecture parameters
config = ModelConfig()
model = KataGoModel(config)
batch = torch.randn(2, config.in_channels, config.board_size, config.board_size)
mask  = torch.ones(2, config.board_size, config.board_size)

outputs = model(batch, board_mask=mask)
# outputs -> dict(policy_logits, game_outcome_logits, score_mean, score_stdev, ownership_map)
```

---


