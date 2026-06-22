from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict
def run(cmd: list[str]) -> None:
    print(f"\n{'='*64}\n  $ {' '.join(cmd)}\n{'='*64}")
    r = subprocess.run(cmd, text=True)
    if r.returncode != 0:
        print(f"  [✗] failed (rc={r.returncode})")
        sys.exit(r.returncode)
def load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset_with_predictions.json")
    ap.add_argument("--raw_data", default="data/dataset_synth.json",
                    help="input data for prediction generation")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--skip_predict", action="store_true")
    ap.add_argument("--skip_policy", action="store_true")
    args = ap.parse_args()
    py = sys.executable
    if not args.skip_predict:
        run([py, "02_train_predictor.py", "--data", args.raw_data,
             "--pred_out", args.data])
    if not args.skip_policy:
        run([py, "03_policy_engine.py", "--data", args.data,
             "--results_out", "data/policy_results.json"])
    run([py, "04_ablation_ci.py", "--data", args.data,
         "--n_seeds", str(args.n_seeds)])
    run([py, "05_gss_sensitivity.py", "--ablation", "data/ablation_ci.json"])
    nums: Dict[str, Any] = {}
    try:
        tp = load("data/test_predictions.json")
        def mape(key_t, key_p):
            pairs = [(float(s[key_t]), float(s[key_p])) for s in tp
                     if float(s.get(key_t, 0)) > 0]
            return round(sum(abs(t-p)/t for t, p in pairs)/len(pairs)*100, 2) if pairs else None
        nums["held_out"] = {
            "n": len(tp),
            "input_token_mape": mape("true_input_tokens", "pred_input_tokens"),
            "output_token_mape": mape("true_output_tokens", "pred_output_tokens"),
            "cost_mape": mape("true_cost_usd", "pred_cost_usd"),
        }
    except Exception as e:
        nums["held_out"] = {"error": str(e)}
    if not args.skip_policy:
        pol = load("data/policy_results.json")["metrics"]
        nums["policy"] = {
            "cost_reduction_pct": pol["cost_reduction_pct"],
            "bvr_baseline_pct": pol["budget_violation_rate_baseline_pct"],
            "bvr_proposed_pct": pol["budget_violation_rate_proposed_pct"],
            "bvr_improvement_pct": round(
                (pol["budget_violation_rate_baseline_pct"]
                 - pol["budget_violation_rate_proposed_pct"]), 2),
            "false_suppression_rate_pct": pol["false_suppression_rate_pct"],
            "task_success_rate_pct": pol["task_success_rate_pct"],
        }
    abl = load("data/ablation_ci.json")
    agg = abl["aggregate"]
    nums["ablation"] = {
        "best_model": abl["best_model"],
        "n_seeds": abl["config"]["n_seeds"],
        "table": {
            m: {
                "gss_mean": agg[m]["gss"]["mean"],
                "gss_ci95": agg[m]["gss"]["ci95"],
                "cost_mape_mean": agg[m]["cost_mape"]["mean"],
                "cost_mape_ci95": agg[m]["cost_mape"]["ci95"],
                "latency_mean_ms": agg[m]["latency_mean_ms"]["mean"],
                "latency_stdev_ms": agg[m]["latency_stdev_ms"]["mean"],
            } for m in agg
        },
    }
    sens = load("data/gss_sensitivity.json")
    nums["gss_sensitivity"] = {
        "baseline_ranking": sens["baseline_ranking"],
        "most_robust_top1": sens["full_sweep"]["most_robust_top1"],
        "robustness_pct": sens["full_sweep"]["robustness_pct"],
        "n_configs": sens["full_sweep"]["n_configs"],
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/paper_numbers.json", "w", encoding="utf-8") as f:
        json.dump(nums, f, indent=2, ensure_ascii=False)
    lines = []
    lines.append("=" * 70)
    lines.append("  PAPER NUMBERS — key values for manuscript update")
    lines.append("=" * 70)
    ho = nums.get("held_out", {})
    lines.append("\n[Held-out prediction accuracy]  (Table 5)")
    lines.append(f"  N = {ho.get('n')}")
    lines.append(f"  Input  Token MAPE : {ho.get('input_token_mape')} %")
    lines.append(f"  Output Token MAPE : {ho.get('output_token_mape')} %")
    lines.append(f"  Cost MAPE         : {ho.get('cost_mape')} %")
    if "policy" in nums:
        p = nums["policy"]
        lines.append("\n[Governance system performance]  (Table 4 / Abstract)")
        lines.append(f"  Cost reduction        : {p['cost_reduction_pct']} %")
        lines.append(f"  BVR baseline→proposed : {p['bvr_baseline_pct']} → {p['bvr_proposed_pct']} %")
        lines.append(f"  BVR improvement       : {p['bvr_improvement_pct']} %p")
        lines.append(f"  False suppression rate: {p['false_suppression_rate_pct']} %")
        lines.append(f"  Task success rate     : {p['task_success_rate_pct']} %")
    a = nums["ablation"]
    lines.append(f"\n[Ablation — {a['n_seeds']} seeds, mean ± 95% CI]  (Table 3)")
    lines.append(f"  Best PEEP by GSS: {a['best_model']}")
    lines.append(f"  {'Model':<14}{'GSS':>16}{'CostMAPE%':>16}{'Lat(ms)':>10}")
    ranked = sorted(a["table"].keys(),
                    key=lambda m: -a["table"][m]["gss_mean"])
    for m in ranked:
        t = a["table"][m]
        lines.append(f"  {m:<14}{t['gss_mean']:>7.3f}±{t['gss_ci95']:<7.3f}"
                     f"{t['cost_mape_mean']:>7.2f}±{t['cost_mape_ci95']:<7.2f}"
                     f"{t['latency_mean_ms']:>9.3f}")
    s = nums["gss_sensitivity"]
    lines.append(f"\n[GSS robustness]  (threshold robustness)")
    lines.append(f"  Baseline ranking : {' > '.join(s['baseline_ranking'])}")
    lines.append(f"  Across {s['n_configs']} configs (norm × weight):")
    lines.append(f"  '{s['most_robust_top1']}' stays top-1 in {s['robustness_pct']}% of them")
    lines.append("=" * 70)
    txt = "\n".join(lines)
    with open("results/paper_numbers.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)
    print("\n[✓] results/paper_numbers.json")
    print("[✓] results/paper_numbers.txt")
if __name__ == "__main__":
    main()
