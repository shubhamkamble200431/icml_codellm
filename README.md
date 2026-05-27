# CODE_LLM — Mechanistic Interpretability Pipeline for Qwen2.5-Coder on MBPP+

## Quick Start

```bash
# 0. Install dependencies
pip install -r requirements.txt

# 1. Generate activations (MBPP+ × 2 Qwen models × 5 runs)
python scripts/contrastive_set_gen.py

# 2. Probe analysis — select best steering layers
python scripts/probing.py

# 3. Symmetry score — compute V_rec, V_cor, and their alignment
python scripts/symmetry.py

# 4. Activation steering — measure pass@1 improvement/degradation
python scripts/steering.py
```

---

## Scripts

| Script | Step | Description |
|--------|------|-------------|
| `scripts/contrastive_set_gen.py` | 1 | Runs 5 generations per MBPP+ problem, saves HDF5 activations + splits |
| `scripts/probing.py` | 2 | Probe analysis, bootstrapped directions, top-N layer selection |
| `scripts/symmetry.py` | 3 | Builds steering dataset (Group A/B), computes V_rec/V_cor, symmetry score |
| `scripts/steering.py` | 4 | Steers model with pre-computed directions, reports Δpass@1 |

---

## Models

| Key | HuggingFace ID | Params |
|-----|---------------|--------|
| `qwen-coder-1.5b-instruct` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.5B |
| `qwen-coder-7b-instruct` | `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B |

---

## Output Structure

All outputs go to `mbppplus/<model_key>_<timestamp>/`:

```
mbppplus/
  qwen-coder-1.5b-instruct_20260426_185842/
    all/                          # all problem activation dirs
      Mbpp_103/
        right/run1/layer_16.h5   # hidden state at token before EOS
        wrong/run2/layer_16.h5
    split.json                   # 70/30 stratified contrastive split
    contrastive_stats.txt        # per-problem verdict counts
    summary.json                 # overall run summary
    probing/
      probing_results.json       # per-layer CV + test metrics
    probing_5fold_cv/
      probing_results.json       # 5-fold CV over all contrastive problems
    steering_dataset.json        # Group A (Hard) + Group B (Easy)
    steering_vectors.json        # V_rec, V_cor, cosine similarity
    symmetry_results.json        # symmetry score + interpretation
    probing/
      probe_analysis.json        # probe full results
      top_layers.json            # selected top-N layers
      directions/layer_XX.npy   # unit-norm steering directions
    steering/
      pipeline2_results.json    # full steering results
      steering_report.txt
      steering_report.xlsx
  all_symmetry_results.json      # combined symmetry across all models
```

## Hardware Requirements

| Model | VRAM | Notes |
|-------|------|-------|
| Qwen2.5-Coder-1.5B | ~4 GB | batch_size=4 |
| Qwen2.5-Coder-7B | ~16 GB | batch_size=1 |

CPU fallback is supported but very slow for generation.

---

## CLI Reference

### Step 1: contrastive_set_gen.py
```bash
python scripts/contrastive_set_gen.py [--models qwen-coder-1.5b-instruct] [--skip-probing] [--recompute-split]
```

### Step 2: probing.py
```bash
python scripts/probing.py [--model qwen-coder-1.5b-instruct] [--top-n 5] [--gap-thresh 0.15]
```

### Step 3: symmetry.py
```bash
python scripts/symmetry.py [--model qwen-coder-1.5b-instruct]
```

### Step 4: steering.py
```bash
python scripts/steering.py [--model qwen-coder-1.5b-instruct] [--alphas-fwd 0.5 1.0 2.0 5.0 10.0] [--device cuda]
```
