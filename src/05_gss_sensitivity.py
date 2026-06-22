from __future__ import annotations
import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
def load_model_metrics(ablation_path: str) -> Dict[str, Dict[str, float]]:
    with open(ablation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    agg = data["aggregate"]
    out = {}
    for name, m in agg.items():
        out[name] = {
            "cost_mape": m["cost_mape"]["mean"],
            "lat_mean": m["latency_mean_ms"]["mean"],
            "lat_stdev": m["latency_stdev_ms"]["mean"],
        }
    return out
def gss(metrics: Dict[str, float], mape_norm: float, lat_norm: float,
        stab_norm: float, w: Tuple[float, float, float]) -> float:
    s_acc = max(0.0, 1.0 - metrics["cost_mape"] / mape_norm)
    s_lat = max(0.0, 1.0 - metrics["lat_mean"] / lat_norm)
    s_stab = max(0.0, 1.0 - metrics["lat_stdev"] / stab_norm)
    return w[0] * s_acc + w[1] * s_lat + w[2] * s_stab
def rank_models(model_metrics: Dict[str, Dict[str, float]], mape_norm, lat_norm,
                stab_norm, w) -> List[str]:
    scored = [(name, gss(m, mape_norm, lat_norm, stab_norm, w))
              for name, m in model_metrics.items()]
    scored.sort(key=lambda x: -x[1])
    return [name for name, _ in scored]
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", default="data/ablation_ci.json")
    ap.add_argument("--out_json", default="data/gss_sensitivity.json")
    ap.add_argument("--out_fig", default="results/figures/fig_gss_sensitivity.png")
    args = ap.parse_args()
    mm = load_model_metrics(args.ablation)
    models = list(mm.keys())
    print(f"[*] Models: {models}")
    mape_norms = [40, 50, 60, 70, 80]
    lat_norms = [300, 500, 700]
    stab_norms = [50, 100, 150]
    weight_schemes = {
        "equal": (1/3, 1/3, 1/3),
        "accuracy_heavy": (0.50, 0.25, 0.25),
        "latency_heavy": (0.25, 0.50, 0.25),
        "stability_heavy": (0.25, 0.25, 0.50),
    }
    norm_top1 = Counter()
    norm_configs = list(itertools.product(mape_norms, lat_norms, stab_norms))
    for mn, ln, sn in norm_configs:
        top = rank_models(mm, mn, ln, sn, weight_schemes["equal"])[0]
        norm_top1[top] += 1
    n_norm = len(norm_configs)
    weight_top1 = {}
    weight_ranks = {}
    for wname, w in weight_schemes.items():
        ranked = rank_models(mm, 60, 500, 100, w)
        weight_top1[wname] = ranked[0]
        weight_ranks[wname] = ranked
    all_top1 = Counter()
    all_configs = 0
    for mn, ln, sn in norm_configs:
        for wname, w in weight_schemes.items():
            top = rank_models(mm, mn, ln, sn, w)[0]
            all_top1[top] += 1
            all_configs += 1
    baseline_rank = rank_models(mm, 60, 500, 100, weight_schemes["equal"])
    print("\n" + "=" * 64)
    print("  GSS Sensitivity Analysis")
    print("=" * 64)
    print(f"  Baseline (60/500/100, equal) ranking:")
    for i, m in enumerate(baseline_rank, 1):
        print(f"    {i}. {m}")
    print(f"\n  [1] Normalization sweep ({n_norm} configs, equal weights)")
    for m, c in norm_top1.most_common():
        print(f"    Top-1 as {m:<14}: {c}/{n_norm}  ({c/n_norm*100:.0f}%)")
    print(f"\n  [2] Weight sweep (norm fixed 60/500/100)")
    for wname in weight_schemes:
        print(f"    {wname:<16}: top-1 = {weight_top1[wname]}")
    print(f"\n  [3] Full sweep ({all_configs} configs: norm × weight)")
    for m, c in all_top1.most_common():
        print(f"    Top-1 as {m:<14}: {c}/{all_configs}  ({c/all_configs*100:.0f}%)")
    print("=" * 64)
    dominant = all_top1.most_common(1)[0]
    print(f"  Most robust top-1: {dominant[0]} "
          f"({dominant[1]/all_configs*100:.0f}% of all configurations)")
    out = {
        "baseline_ranking": baseline_rank,
        "normalization_sweep": {
            "n_configs": n_norm,
            "top1_counts": dict(norm_top1),
        },
        "weight_sweep": {
            "top1_by_scheme": weight_top1,
            "full_ranking_by_scheme": weight_ranks,
        },
        "full_sweep": {
            "n_configs": all_configs,
            "top1_counts": dict(all_top1),
            "most_robust_top1": dominant[0],
            "robustness_pct": round(dominant[1] / all_configs * 100, 1),
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[✓] JSON saved: {args.out_json}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    schemes = list(weight_schemes.keys())
    x = np.arange(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, wname in enumerate(schemes):
        w = weight_schemes[wname]
        vals = [gss(mm[m], 60, 500, 100, w) for m in models]
        ax.bar(x + i * width, vals, width, label=wname)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("GSS")
    ax.set_title("GSS under Different Weight Schemes (norm 60/500/100)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_fig, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[✓] Figure saved: {args.out_fig}")
if __name__ == "__main__":
    main()
