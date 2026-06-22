from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
warnings.filterwarnings("ignore")
GSS_MAPE_NORM = 60.0
GSS_LAT_NORM = 500.0
GSS_STAB_NORM = 100.0
LATENCY_REPEAT = 200
def load_cp(base_dir: str = "."):
    for c in [os.path.join(base_dir, "02_train_predictor.py"),
              os.path.join(os.getcwd(), "02_train_predictor.py")]:
        if os.path.exists(c):
            spec = importlib.util.spec_from_file_location("cp_mod", c)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["cp_mod"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("02_train_predictor.py not found")
def build_models(seed: int) -> Dict[str, Any]:
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    models: Dict[str, Any] = {}
    models["Ridge"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MultiOutputRegressor(Ridge(alpha=1.0))),
    ])
    models["Random Forest"] = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, max_depth=12,
                              random_state=seed, n_jobs=-1))
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = MultiOutputRegressor(
            XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=seed, n_jobs=-1, verbosity=0))
    except ImportError:
        pass
    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = MultiOutputRegressor(
            CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05,
                              random_seed=seed, verbose=0))
    except ImportError:
        pass
    models["MLP"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300,
                               early_stopping=True, random_state=seed)),
    ])
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = MultiOutputRegressor(
            LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=63,
                          min_child_samples=8, subsample=0.85,
                          colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=0.1,
                          random_state=seed, verbose=-1))
    except ImportError:
        pass
    return models
def mape(t: np.ndarray, p: np.ndarray) -> float:
    m = t > 0
    if not np.any(m):
        return 0.0
    return float(np.mean(np.abs((t[m] - p[m]) / t[m])) * 100)
def measure_latency(model, X_single: np.ndarray, repeat: int = LATENCY_REPEAT) -> Tuple[float, float]:
    for _ in range(5):
        model.predict(X_single)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        model.predict(X_single)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))
def gss_components(cost_mape: float, lat_mean: float, lat_stdev: float) -> Dict[str, float]:
    s_acc = max(0.0, 1.0 - cost_mape / GSS_MAPE_NORM)
    s_lat = max(0.0, 1.0 - lat_mean / GSS_LAT_NORM)
    s_stab = max(0.0, 1.0 - lat_stdev / GSS_STAB_NORM)
    return {"s_acc": s_acc, "s_lat": s_lat, "s_stab": s_stab,
            "gss": (s_acc + s_lat + s_stab) / 3.0}
def run_single_seed(cp, samples: List[dict], seed: int) -> Dict[str, Dict[str, float]]:
    train, test = cp.group_train_test_split(samples, test_ratio=0.2, seed=seed)
    X_train, y_train = cp.prepare_xy(train)
    X_test, y_test = cp.prepare_xy(test)
    y_train_io = y_train[:, :2]
    y_test_io = y_test[:, :2]
    model_name = cp.infer_model_name(samples)
    pin, pout = cp.PRICING.get(model_name, cp.PRICING["offline"])
    X_single = X_test[:1]
    out: Dict[str, Dict[str, float]] = {}
    for name, model in build_models(seed).items():
        model.fit(X_train, np.log1p(y_train_io))
        log_pred = model.predict(X_test)
        pred_io = np.maximum(np.expm1(log_pred), 0)
        pred_cost = pred_io[:, 0] * pin + pred_io[:, 1] * pout
        true_cost = y_test[:, 2]
        cost_mape = mape(true_cost, pred_cost)
        out_mape = mape(y_test_io[:, 1], pred_io[:, 1])
        lat_mean, lat_stdev = measure_latency(model, X_single)
        comp = gss_components(cost_mape, lat_mean, lat_stdev)
        out[name] = {
            "cost_mape": cost_mape, "output_mape": out_mape,
            "latency_mean_ms": lat_mean, "latency_stdev_ms": lat_stdev,
            **comp,
        }
    return out
def ci95(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    sd = statistics.stdev(values)
    return 1.96 * sd / math.sqrt(n)
def aggregate(per_seed: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    models = list(per_seed[0].keys())
    metrics = list(per_seed[0][models[0]].keys())
    agg: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m in models:
        agg[m] = {}
        for met in metrics:
            vals = [s[m][met] for s in per_seed]
            agg[m][met] = {
                "mean": round(statistics.mean(vals), 4),
                "ci95": round(ci95(vals), 4),
                "values": [round(v, 4) for v in vals],
            }
    return agg
def plot_gss_ci(agg: Dict, out_fig: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = sorted(agg.keys(), key=lambda m: agg[m]["gss"]["mean"])
    means = [agg[m]["gss"]["mean"] for m in models]
    errs = [agg[m]["gss"]["ci95"] for m in models]
    best = models[-1]
    colors = ["#2ca02c" if m == best else "#4c72b0" for m in models]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bars = ax.barh(models, means, xerr=errs, color=colors,
                   edgecolor="white", height=0.62,
                   error_kw=dict(ecolor="#333", capsize=4, lw=1.2))
    ax.set_xlabel("Governance Suitability Score (GSS)  [mean ± 95% CI]")
    ax.set_title("PEEP Candidate Comparison across Seeds", fontweight="bold")
    ax.set_xlim(0, 1.02)
    for m, v, e in zip(models, means, errs):
        ax.text(v + e + 0.012, models.index(m), f"{v:.3f}", va="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[✓] Figure saved: {out_fig}")
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset_with_predictions.json")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--out_json", default="data/ablation_ci.json")
    ap.add_argument("--out_fig", default="results/figures/fig_ablation_ci.png")
    args = ap.parse_args()
    cp = load_cp()
    with open(args.data, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"[*] Loaded {len(samples)} samples")
    seeds = [args.base_seed + i for i in range(args.n_seeds)]
    per_seed = []
    for si, seed in enumerate(seeds, 1):
        print(f"\n[*] Seed {si}/{len(seeds)} (seed={seed}) ...")
        res = run_single_seed(cp, samples, seed)
        per_seed.append(res)
        for name, m in sorted(res.items(), key=lambda kv: -kv[1]["gss"]):
            print(f"    {name:<14} GSS={m['gss']:.3f}  costMAPE={m['cost_mape']:.2f}%  "
                  f"lat={m['latency_mean_ms']:.3f}ms")
    agg = aggregate(per_seed)
    print("\n" + "=" * 72)
    print("  Multi-seed summary (mean ± 95% CI)")
    print("=" * 72)
    print(f"  {'Model':<14} {'GSS':>16} {'Cost MAPE %':>16} {'Latency ms':>16}")
    print("  " + "-" * 68)
    ranked = sorted(agg.keys(), key=lambda m: -agg[m]["gss"]["mean"])
    for m in ranked:
        g = agg[m]["gss"]; c = agg[m]["cost_mape"]; l = agg[m]["latency_mean_ms"]
        print(f"  {m:<14} {g['mean']:>7.3f}±{g['ci95']:<7.3f} "
              f"{c['mean']:>7.2f}±{c['ci95']:<7.2f} "
              f"{l['mean']:>7.3f}±{l['ci95']:<7.3f}")
    print("=" * 72)
    print(f"  Best PEEP by mean GSS: {ranked[0]}")
    out = {
        "config": {"n_seeds": args.n_seeds, "seeds": seeds,
                   "gss_norm": {"mape": GSS_MAPE_NORM, "lat": GSS_LAT_NORM,
                                "stab": GSS_STAB_NORM},
                   "latency_repeat": LATENCY_REPEAT},
        "aggregate": agg,
        "best_model": ranked[0],
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[✓] JSON saved: {args.out_json}")
    plot_gss_ci(agg, args.out_fig)
if __name__ == "__main__":
    main()
