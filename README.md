# attention-from-scratch

A minimal, from-scratch exploration of the attention mechanism (scaled dot-product attention) using NumPy.

This repository is intentionally small and pedagogical. It aims to help you _see_ and _reason about_ attention without training loops, autograd, or deep-learning frameworks.

## What you'll find

- `notebooks/` — step-by-step notebook-style guides (in markdown) you can paste into Jupyter.
- `src/attention.py` — clean NumPy implementation with helper utilities.
- `src/softmax.py` — numerically stable softmax.
- `experiments/manual_qkv_cases.py` — runnable script that walks through the core experiments and prints matrices.
- `notes/conceptual_takeaways.md` — short, interview-ready takeaways.

## Requirements

- Python 3.8+
- NumPy
- Jupyter (optional, for notebooks)

Install:

```bash
python -m venv .venv
source .venv/bin/activate # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```
