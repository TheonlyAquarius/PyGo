# AGENTS.md

## Objectives for Agents

1. Work primarily in `model.py` and `tests/`.  
2. Align any architectural changes with self-play training and inference goals.  
3. Add tests for policy/value heads, masking logic, and forward/backward passes.  
4. Document performance metrics and evaluation procedures.

## Guidelines for Software Agents

When engaging as a software engineering agent (e.g., Codex):

* **Understand the Goal:**
  Extend beyond formatting and linting—reason about how this model drives self-play or real-time Go analysis.

* **Key Tasks:**

  1. **Refactor & Document** for clarity and testability
  2. **Identify & Propose** architectural improvements or new research directions
  3. **Generate & Validate** unit tests covering edge cases and expected performance metrics
  4. **Bridge to Gameplay:** Sketch or scaffold modules for feature extraction, MCTS wrappers, and GTP integration

* **Verification:**
  Use synthetic data, mask tests, and simple win/loss simulations to validate that changes preserve or improve expected behaviors.

---

## Setup Scripts (for Agent Environments)

```bash
pip install torch pytest black flake8
# Add any additional dependencies here
```
