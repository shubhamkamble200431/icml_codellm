"""
CODE_LLM symmetry.py  —  Step 3: Symmetry score between recovery and corruption directions
==========================================================================================
Reads activations + probing results from ../mbppplus/ and computes:

  1. Steering dataset  — Group A (Hard/Recovery: n_passed 1-2)
                         Group B (Easy/Corruption: n_passed 3-4)
  2. Steering vectors  — V_rec = mean(Right_A) - mean(Wrong_A)
                         V_cor = mean(Wrong_B) - mean(Right_B)
  3. Symmetry score    — dot(norm(V_rec), -norm(V_cor))
                         +1 = perfectly symmetric, -1 = anti-symmetric

Reads:
  ../mbppplus/<model>_<ts>/split.json              — contrastive test IDs
  ../mbppplus/<model>_<ts>/contrastive_stats.txt   — per-problem n_passed
  ../mbppplus/<model>_<ts>/probing_5fold_cv/       — target layer (best CV AUC)
  ../mbppplus/<model>_<ts>/all/<Mbpp_ID>/right|wrong/runN/layer_NN.h5

Writes per model:
  ../mbppplus/<model>_<ts>/steering_dataset.json
  ../mbppplus/<model>_<ts>/steering_vectors.json
  ../mbppplus/<model>_<ts>/symmetry_results.json

Writes combined:
  ../mbppplus/all_symmetry_results.json

Usage
-----
  python symmetry.py                               # process all Qwen runs
  python symmetry.py --model qwen-coder-1.5b-instruct
  python symmetry.py --model-dir /full/path/to/model_dir
"""

import argparse
import json
import os
import re
import sys
import numpy as np
import h5py
from pathlib import Path

MBPPPLUS_ROOT = Path(__file__).parent.parent / "mbppplus"

QWEN_MODEL_KEYS = [
    "qwen-coder-1.5b-instruct",
    "qwen-coder-7b-instruct",
]



def _infer_model_key(dir_name: str) -> str | None:
    for mk in QWEN_MODEL_KEYS:
        if dir_name.startswith(mk + "_") or dir_name == mk:
            return mk
    return None


def find_model_dirs(root: Path, model_key: str | None = None) -> list[tuple[str, Path]]:
    """Return [(model_key, model_dir), ...] for completed activation runs."""
    results = []
    if not root.exists():
        return results
    best: dict[str, Path] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d / "split.json").exists():
            continue
        mk = _infer_model_key(d.name)
        if mk is None:
            continue
        best[mk] = d
    for mk, d in best.items():
        if model_key is None or mk == model_key:
            results.append((mk, d))
    return results



def _parse_contrastive_stats(stats_path: Path) -> dict:
    """Parse contrastive_stats.txt → {task_id: n_passed}."""
    result: dict = {}
    in_breakdown = False
    with open(stats_path) as f:
        for line in f:
            s = line.strip()
            if "PER-PROMPT BREAKDOWN" in s:
                in_breakdown = True
                continue
            if in_breakdown and (s.startswith("---") or s.startswith("task_id")):
                continue
            if in_breakdown and (s == "" or s.startswith("====")):
                in_breakdown = False
                continue
            if in_breakdown:
                parts = s.split()
                if len(parts) >= 2:
                    try:
                        result[parts[0]] = int(parts[1])
                    except ValueError:
                        pass
    return result


def _best_layer_from_probing(model_dir: Path) -> int | None:
    """Return layer idx with highest CV AUC from 5-fold CV probing, or fall back to regular probing."""
    cv_path = model_dir / "probing_5fold_cv" / "probing_results.json"
    if cv_path.exists():
        with open(cv_path) as f:
            rows = json.load(f)
        if rows:
            best = max(rows, key=lambda r: r.get("cv_auc_mean", 0))
            return best["layer_idx"]

    probe_path = model_dir / "probing" / "probing_results.json"
    if probe_path.exists():
        with open(probe_path) as f:
            rows = json.load(f)
        if rows:
            best = max(rows, key=lambda r: r.get("cv_auc_mean") or r.get("cv_bal_acc_mean") or 0)
            return best["layer_idx"]

    return None


def build_steering_dataset(model_dir: Path) -> dict:
    """Build steering dataset (Group A + Group B) for one model dir."""
    with open(model_dir / "split.json") as f:
        sp = json.load(f)
    contrastive_test_ids = sp["test"]["contrastive_ids"]

    n_passed_map = _parse_contrastive_stats(model_dir / "contrastive_stats.txt")

    target_layer = _best_layer_from_probing(model_dir)

    group_a, group_b = [], []
    excluded_0, excluded_5 = [], []
    for pid in sorted(contrastive_test_ids):
        n = n_passed_map.get(pid)
        if n is None:
            print(f"  WARNING: {pid} not found in contrastive_stats.txt")
            continue
        if n == 0:
            excluded_0.append(pid)
        elif n == 5:
            excluded_5.append(pid)
        elif n in (1, 2):
            group_a.append(pid)
        elif n in (3, 4):
            group_b.append(pid)

    dataset = {
        "model_dir":    str(model_dir),
        "target_layers": [target_layer] if target_layer is not None else [],
        "groups": {
            "A": {
                "label":       "Hard/Recovery",
                "description": "n_passed ∈ {1, 2} — good candidates for testing recovery steering",
                "criteria":    "n_passed == 1 or n_passed == 2",
                "problem_ids": group_a,
            },
            "B": {
                "label":       "Easy/Corruption",
                "description": "n_passed ∈ {3, 4} — good candidates for testing corruption steering",
                "criteria":    "n_passed == 3 or n_passed == 4",
                "problem_ids": group_b,
            },
        },
        "excluded": {"n_passed_0": excluded_0, "n_passed_5": excluded_5},
        "stats": {
            "total_contrastive_test_size": len(contrastive_test_ids),
            "group_A_size":  len(group_a),
            "group_B_size":  len(group_b),
            "excluded_size": len(excluded_0) + len(excluded_5),
        },
    }
    out = model_dir / "steering_dataset.json"
    with open(out, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"  Steering dataset → {out}")
    print(f"    Group A (Hard/Recovery):   {len(group_a)}")
    print(f"    Group B (Easy/Corruption): {len(group_b)}")
    print(f"    Target layer:              {target_layer}")
    return dataset



def _load_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["activation"][:].astype(np.float64)


def _collect_activations(model_dir: Path, problem_ids: list, layer_idx: int, side: str) -> list:
    """Collect hidden-state vectors for one side (right/wrong) at one layer."""
    layer_name = f"layer_{layer_idx:02d}"
    acts = []
    for pid in sorted(problem_ids):
        folder   = pid.replace("/", "_")
        side_dir = model_dir / "all" / folder / side
        if not side_dir.is_dir():
            continue
        for run_name in sorted(os.listdir(side_dir)):
            h5 = side_dir / run_name / f"{layer_name}.h5"
            if h5.exists():
                try:
                    acts.append(_load_h5(h5))
                except Exception as e:
                    print(f"  [WARN] H5 load error {h5}: {e}")
    return acts


def compute_steering_vectors(model_dir: Path) -> dict:
    """Compute V_rec and V_cor from Group A / Group B activations."""
    with open(model_dir / "steering_dataset.json") as f:
        ds = json.load(f)

    if not ds["target_layers"]:
        raise RuntimeError("No target layer in steering_dataset.json — run probing.py first.")

    layer_idx   = ds["target_layers"][0]
    group_a_ids = ds["groups"]["A"]["problem_ids"]
    group_b_ids = ds["groups"]["B"]["problem_ids"]

    print(f"\n  Layer {layer_idx}  |  Group A: {len(group_a_ids)}  Group B: {len(group_b_ids)}")

    right_a = _collect_activations(model_dir, group_a_ids, layer_idx, "right")
    wrong_a = _collect_activations(model_dir, group_a_ids, layer_idx, "wrong")
    right_b = _collect_activations(model_dir, group_b_ids, layer_idx, "right")
    wrong_b = _collect_activations(model_dir, group_b_ids, layer_idx, "wrong")

    if len(right_a) == 0 or len(wrong_a) == 0:
        raise RuntimeError("Group A has no right or wrong activations.")
    if len(right_b) == 0 or len(wrong_b) == 0:
        raise RuntimeError("Group B has no right or wrong activations.")

    print(f"    A — right runs: {len(right_a)}  wrong runs: {len(wrong_a)}")
    print(f"    B — right runs: {len(right_b)}  wrong runs: {len(wrong_b)}")

    mean_right_a = np.mean(np.stack(right_a), axis=0)
    mean_wrong_a = np.mean(np.stack(wrong_a), axis=0)
    V_rec        = mean_right_a - mean_wrong_a
    V_rec_norm   = V_rec / np.linalg.norm(V_rec)

    mean_wrong_b = np.mean(np.stack(wrong_b), axis=0)
    mean_right_b = np.mean(np.stack(right_b), axis=0)
    V_cor        = mean_wrong_b - mean_right_b
    V_cor_norm   = V_cor / np.linalg.norm(V_cor)

    cos_sim = float(np.dot(V_rec_norm, V_cor_norm))
    hidden_size = int(V_rec.shape[0])

    print(f"    cos(V_rec, V_cor): {cos_sim:+.4f}")

    result = {
        "layer":      layer_idx,
        "hidden_size": hidden_size,
        "V_rec": {
            "description":  "mean(Right_A) - mean(Wrong_A)  [Recovery direction]",
            "normalised":   V_rec_norm.tolist(),
            "l2_norm":      float(np.linalg.norm(V_rec)),
            "n_right_runs": len(right_a),
            "n_wrong_runs": len(wrong_a),
        },
        "V_cor": {
            "description":  "mean(Wrong_B) - mean(Right_B)  [Corruption direction]",
            "normalised":   V_cor_norm.tolist(),
            "l2_norm":      float(np.linalg.norm(V_cor)),
            "n_right_runs": len(right_b),
            "n_wrong_runs": len(wrong_b),
        },
        "diagnostics": {"cosine_V_rec_V_cor": cos_sim},
    }

    out = model_dir / "steering_vectors.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Steering vectors  → {out}")
    return result



def compute_symmetry(model_dir: Path, display_name: str) -> dict:
    """Compute symmetry_score = dot(V_rec_norm, -V_cor_norm) for one model."""
    with open(model_dir / "steering_vectors.json") as f:
        sv = json.load(f)

    V_rec_norm = np.array(sv["V_rec"]["normalised"], dtype=np.float64)
    V_cor_norm = np.array(sv["V_cor"]["normalised"], dtype=np.float64)

    symmetry_score = float(np.dot(V_rec_norm, -V_cor_norm))

    target_layer = sv["layer"]
    auroc = cv_bal_acc = None
    cv_path = model_dir / "probing_5fold_cv" / "probing_results.json"
    if cv_path.exists():
        with open(cv_path) as f:
            for p in json.load(f):
                if p["layer_idx"] == target_layer:
                    auroc      = p.get("cv_auc_mean")
                    cv_bal_acc = p.get("cv_bal_acc_mean")
                    break

    result = {
        "model":           display_name,
        "model_dir":       str(model_dir),
        "layer":           target_layer,
        "symmetry_score":  symmetry_score,
        "auroc":           auroc,
        "cv_bal_acc":      cv_bal_acc,
        "V_rec_l2_norm":   sv["V_rec"]["l2_norm"],
        "V_cor_l2_norm":   sv["V_cor"]["l2_norm"],
        "cos_V_rec_V_cor": sv["diagnostics"]["cosine_V_rec_V_cor"],
        "interpretation": (
            "STRONG"  if symmetry_score > 0.7 else
            "PARTIAL" if symmetry_score > 0.3 else
            "WEAK"    if symmetry_score >= 0  else
            "ANTI"
        ),
    }

    out = model_dir / "symmetry_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    return result



def run_model(model_key: str, model_dir: Path) -> dict | None:
    display = {
        "qwen-coder-1.5b-instruct": "Qwen2.5-Coder-1.5B",
        "qwen-coder-7b-instruct":   "Qwen2.5-Coder-7B",
    }.get(model_key, model_key)

    print(f"\n{'='*70}")
    print(f"  {display}  ←  {model_dir.name}")
    print(f"{'='*70}")

    print("\n  [1/3] Building steering dataset …")
    try:
        build_steering_dataset(model_dir)
    except Exception as e:
        print(f"  [ERROR] steering dataset: {e}")
        return None

    print("\n  [2/3] Computing steering vectors …")
    try:
        compute_steering_vectors(model_dir)
    except Exception as e:
        print(f"  [ERROR] steering vectors: {e}")
        return None

    print("\n  [3/3] Computing symmetry score …")
    try:
        result = compute_symmetry(model_dir, display)
    except Exception as e:
        print(f"  [ERROR] symmetry: {e}")
        return None

    score = result["symmetry_score"]
    print(f"\n  ✓  symmetry_score = {score:+.4f}  [{result['interpretation']}]"
          + (f"  AUROC={result['auroc']:.3f}" if result["auroc"] else ""))
    print(f"  Symmetry results  → {model_dir / 'symmetry_results.json'}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="CODE_LLM symmetry — compute V_rec, V_cor and symmetry score for Qwen on MBPP+"
    )
    parser.add_argument("--model",     default=None, choices=QWEN_MODEL_KEYS,
                        help="Run only for this model key.")
    parser.add_argument("--model-dir", default=None,
                        help="Path to a specific model activation dir (overrides auto-discovery).")
    parser.add_argument("--root",      default=str(MBPPPLUS_ROOT),
                        help="Root dir to scan for model dirs (default: ../mbppplus).")
    args = parser.parse_args()

    if args.model_dir:
        model_dir = Path(args.model_dir)
        mk        = _infer_model_key(model_dir.name) or "unknown"
        pairs     = [(mk, model_dir)]
    else:
        root  = Path(args.root)
        pairs = find_model_dirs(root, model_key=args.model)
        if not pairs:
            print(f"[ERROR] No completed activation runs found in {root}")
            print("Run: python contrastive_set_gen.py  (and then python probing.py)")
            sys.exit(1)

    all_results = []

    for mk, model_dir in pairs:
        result = run_model(mk, model_dir)
        if result:
            all_results.append(result)

    if not all_results:
        print("\n[ERROR] No symmetry results computed.")
        sys.exit(1)

    print(f"\n\n{'='*72}")
    print(f"  SYMMETRY SUMMARY  —  MBPP+  ({len(all_results)} model(s))")
    print(f"{'='*72}")
    print(f"  {'Model':<28}  {'Layer':>5}  {'Symmetry':>9}  {'AUROC':>7}  Band")
    print(f"  {'-'*28}  {'-'*5}  {'-'*9}  {'-'*7}  ----")
    sorted_res = sorted(all_results, key=lambda r: r["symmetry_score"], reverse=True)
    for r in sorted_res:
        auc_s = f"{r['auroc']:.3f}" if r["auroc"] is not None else "   N/A"
        print(f"  {r['model']:<28}  {r['layer']:>5}  "
              f"{r['symmetry_score']:>+9.4f}  {auc_s:>7}  {r['interpretation']}")

    mean_score = float(np.mean([r["symmetry_score"] for r in all_results]))
    print(f"\n  Mean symmetry score: {mean_score:+.4f}")

    combined = {
        "models":  all_results,
        "summary": {
            "most_symmetric":    sorted_res[0]["model"],
            "least_symmetric":   sorted_res[-1]["model"],
            "mean_symmetry_score": mean_score,
            "ranking": [{"model": r["model"], "symmetry_score": r["symmetry_score"],
                         "band": r["interpretation"]} for r in sorted_res],
            "interpretation": {
                "above_0.7":  "Strong symmetry: recovery and corruption share one correctness axis",
                "0.3_to_0.7": "Partial symmetry: related but not mirror-image",
                "below_0.3":  "Weak/no symmetry: different geometry for fixing vs breaking",
                "negative":   "Anti-symmetric: fixing and breaking pull in opposing directions",
            },
        },
    }
    combined_path = Path(args.root) / "all_symmetry_results.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Combined results  → {combined_path}")


if __name__ == "__main__":
    main()
