import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Iterable
from pathlib import Path
from collections import defaultdict, Counter
import statistics
import math
RANDOM_SEED = 42
def safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)
def safe_int(v, default=0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return int(default)
def quantile(values: List[float], q: float, fallback: float) -> float:
    if not values:
        return fallback
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = max(0.0, min(1.0, q)) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac
@dataclass
class PolicyConfig:
    alpha: float = 0.40
    beta: float  = 0.35
    gamma: float = 0.25
    tau1: float = 0.15
    tau2: float = 0.30
    tau3: float = 0.50
    tau4: float = 0.72
    calibration_mode: str = "quantile"
    q_tau1: float = 0.50
    q_tau2: float = 0.75
    q_tau3: float = 0.90
    q_tau4: float = 0.97
    max_output_limit: int = 512
    min_budget_usd: float = 0.001
    low_cost_route_factor: float = 0.35
    approval_execution_factor: float = 0.50
    approval_cost_factor: float = 0.70
    max_depth_for_norm: int = 20
    dob_single_cost_ratio_high: float = 0.50
    dob_cpr_ratio_high: float = 0.80
    dob_session_depth_high: int = 15
@dataclass
class CalibrationInfo:
    mode: str
    sample_count: int
    tau1: float
    tau2: float
    tau3: float
    tau4: float
    score_summary: Dict[str, float] = field(default_factory=dict)
def compute_risk_score(
    pred_cost: float,
    remaining_user_budget: float,
    remaining_workspace_budget: float,
    session_depth: int,
    cfg: PolicyConfig,
) -> float:
    Bu = max(cfg.min_budget_usd, remaining_user_budget)
    Bw = max(cfg.min_budget_usd, remaining_workspace_budget)
    cost_ratio_user = pred_cost / Bu
    cost_ratio_ws   = pred_cost / Bw
    depth_norm      = min(session_depth / max(1.0, float(cfg.max_depth_for_norm)), 1.0)
    rho = (cfg.alpha * cost_ratio_user
           + cfg.beta  * cost_ratio_ws
           + cfg.gamma * depth_norm)
    return min(max(rho, 0.0), 1.0)
def calibrate_thresholds(samples: List[dict], cfg: PolicyConfig) -> CalibrationInfo:
    scores = []
    for s in samples:
        pred_cost = safe_float(s.get("pred_cost_usd", s.get("true_cost_usd", 0.0)))
        rem_user = safe_float(s.get("remaining_user_budget", 50.0), 50.0)
        rem_ws = safe_float(s.get("remaining_workspace_budget", 200.0), 200.0)
        depth = safe_int(s.get("session_depth", 1), 1)
        scores.append(compute_risk_score(pred_cost, rem_user, rem_ws, depth, cfg))
    if cfg.calibration_mode == "quantile" and scores:
        tau1 = quantile(scores, cfg.q_tau1, cfg.tau1)
        tau2 = max(tau1 + 1e-6, quantile(scores, cfg.q_tau2, cfg.tau2))
        tau3 = max(tau2 + 1e-6, quantile(scores, cfg.q_tau3, cfg.tau3))
        tau4 = max(tau3 + 1e-6, quantile(scores, cfg.q_tau4, cfg.tau4))
        cfg.tau1, cfg.tau2, cfg.tau3, cfg.tau4 = [min(0.999999, t) for t in (tau1, tau2, tau3, tau4)]
    score_summary = {
        "min": round(min(scores), 4) if scores else 0.0,
        "mean": round(statistics.mean(scores), 4) if scores else 0.0,
        "median": round(statistics.median(scores), 4) if scores else 0.0,
        "max": round(max(scores), 4) if scores else 0.0,
    }
    return CalibrationInfo(
        mode=cfg.calibration_mode,
        sample_count=len(scores),
        tau1=round(cfg.tau1, 4),
        tau2=round(cfg.tau2, 4),
        tau3=round(cfg.tau3, 4),
        tau4=round(cfg.tau4, 4),
        score_summary=score_summary,
    )
CONTROL_ACTIONS = {
    "Allow": 0,
    "LimitOutput": 1,
    "RouteToLowCost": 2,
    "RequireApproval": 3,
    "Suppress": 4,
}
ACTION_LABELS = {v: k for k, v in CONTROL_ACTIONS.items()}
def apply_policy(rho: float, cfg: PolicyConfig) -> str:
    if rho < cfg.tau1:
        return "Allow"
    if rho < cfg.tau2:
        return "LimitOutput"
    if rho < cfg.tau3:
        return "RouteToLowCost"
    if rho < cfg.tau4:
        return "RequireApproval"
    return "Suppress"
def compute_cpr(session_costs: List[float], propagation_weights: Optional[List[float]] = None) -> float:
    if not session_costs:
        return 0.0
    if propagation_weights is None:
        propagation_weights = [1.0 * (1.08 ** i) for i in range(len(session_costs))]
    return float(sum(c * w for c, w in zip(session_costs, propagation_weights)))
@dataclass
class DoBStatus:
    triggered: bool
    reason: str
    severity: str
def detect_dob(
    pred_cost: float,
    remaining_budget: float,
    session_depth: int,
    cpr: float,
    cfg: PolicyConfig,
) -> DoBStatus:
    if remaining_budget <= 0:
        return DoBStatus(True, "Budget exhausted", "critical")
    cost_ratio = pred_cost / max(cfg.min_budget_usd, remaining_budget)
    cpr_ratio = cpr / max(cfg.min_budget_usd, remaining_budget)
    if cost_ratio > 1.0:
        return DoBStatus(True, f"Single request exceeds remaining budget ({cost_ratio:.2f}x)", "critical")
    if cost_ratio >= cfg.dob_single_cost_ratio_high:
        return DoBStatus(True, f"Single request consumes >={cfg.dob_single_cost_ratio_high:.0%} of budget ({cost_ratio:.0%})", "high")
    if session_depth > cfg.dob_session_depth_high:
        return DoBStatus(True, f"Excessive session depth: {session_depth}", "medium")
    if cpr_ratio >= cfg.dob_cpr_ratio_high:
        return DoBStatus(True, f"CPR exceeds >={cfg.dob_cpr_ratio_high:.0%} of remaining budget ({cpr_ratio:.0%})", "high")
    return DoBStatus(False, "Normal", "low")
def estimate_executed_cost(true_cost: float, pred_output_tokens: float, action: str, cfg: PolicyConfig) -> float:
    if action == "Suppress":
        return 0.0
    if action == "Allow":
        return true_cost
    if action == "LimitOutput":
        true_out = max(1.0, pred_output_tokens)
        ratio = min(1.0, cfg.max_output_limit / true_out)
        ratio = max(0.15, ratio)
        return true_cost * ratio
    if action == "RouteToLowCost":
        return true_cost * cfg.low_cost_route_factor
    if action == "RequireApproval":
        return true_cost * cfg.approval_execution_factor * cfg.approval_cost_factor
    return true_cost
@dataclass
class ControlResult:
    sample_id: str
    user_id: str
    category: str
    action: str
    risk_score: float
    pred_cost: float
    true_cost: float
    executed_cost: float
    pred_input_tokens: float
    pred_output_tokens: float
    remaining_user_budget: float
    remaining_workspace_budget: float
    dob: DoBStatus
    cpr: float
    is_suppressed: bool
    is_false_suppression: bool
    budget_violation_baseline: bool
    budget_violation_proposed: bool
    def to_dict(self):
        d = asdict(self)
        d["dob"] = asdict(self.dob)
        return d
def evaluate_sample(
    sample: dict,
    session_costs: List[float],
    cfg: PolicyConfig,
) -> ControlResult:
    pred_cost = safe_float(sample.get("pred_cost_usd", sample.get("true_cost_usd", 0.0)))
    true_cost = safe_float(sample.get("true_cost_usd", 0.0))
    pred_in = safe_float(sample.get("pred_input_tokens", sample.get("true_input_tokens", 0.0)))
    pred_out = safe_float(sample.get("pred_output_tokens", sample.get("true_output_tokens", 0.0)))
    rem_user = safe_float(sample.get("remaining_user_budget", 50.0), 50.0)
    rem_ws = safe_float(sample.get("remaining_workspace_budget", 200.0), 200.0)
    depth = safe_int(sample.get("session_depth", 1), 1)
    cat = str(sample.get("category", "unknown"))
    user_id = str(sample.get("user_id", "unknown_user"))
    rho = compute_risk_score(pred_cost, rem_user, rem_ws, depth, cfg)
    cpr = compute_cpr(session_costs + [pred_cost])
    dob = detect_dob(pred_cost, rem_user, depth, cpr, cfg)
    action = apply_policy(rho, cfg)
    if dob.triggered and action in ("Allow", "LimitOutput"):
        action = "RequireApproval" if dob.severity in ("medium", "high") else "Suppress"
    executed_cost = estimate_executed_cost(true_cost, pred_out, action, cfg)
    is_suppressed = action == "Suppress"
    is_false_suppression = (action == "Suppress") and (cat != "attack")
    budget_violation_baseline = true_cost > rem_user
    budget_violation_proposed = executed_cost > rem_user
    return ControlResult(
        sample_id=str(sample.get("sample_id", "unknown")),
        user_id=user_id,
        category=cat,
        action=action,
        risk_score=round(rho, 4),
        pred_cost=round(pred_cost, 8),
        true_cost=round(true_cost, 8),
        executed_cost=round(executed_cost, 8),
        pred_input_tokens=round(pred_in, 2),
        pred_output_tokens=round(pred_out, 2),
        remaining_user_budget=round(rem_user, 8),
        remaining_workspace_budget=round(rem_ws, 8),
        dob=dob,
        cpr=round(cpr, 6),
        is_suppressed=is_suppressed,
        is_false_suppression=is_false_suppression,
        budget_violation_baseline=budget_violation_baseline,
        budget_violation_proposed=budget_violation_proposed,
    )
def sort_samples_for_session_tracking(samples: List[dict]) -> List[dict]:
    def _key(s: dict):
        return (
            str(s.get("user_id", "")),
            safe_int(s.get("session_depth", 0), 0),
            str(s.get("sample_id", "")),
        )
    return sorted(samples, key=_key)
def run_policy_evaluation(samples: List[dict], cfg: PolicyConfig) -> List[ControlResult]:
    session_tracker: Dict[str, List[float]] = defaultdict(list)
    results = []
    for s in sort_samples_for_session_tracking(samples):
        uid = str(s.get("user_id", "unknown_user"))
        session_costs = list(session_tracker[uid])
        result = evaluate_sample(s, session_costs, cfg)
        if result.executed_cost > 0:
            session_tracker[uid].append(result.executed_cost)
        results.append(result)
    return results
def compute_metrics(results: List[ControlResult], all_samples: List[dict]) -> Dict:
    n = max(1, len(results))
    baseline_total_cost = sum(safe_float(s.get("true_cost_usd", 0.0)) for s in all_samples)
    proposed_total_cost = sum(r.executed_cost for r in results)
    cost_reduction_pct = (baseline_total_cost - proposed_total_cost) / max(1e-9, baseline_total_cost) * 100.0
    bvr_baseline = sum(1 for r in results if r.budget_violation_baseline) / n * 100.0
    bvr_proposed = sum(1 for r in results if r.budget_violation_proposed) / n * 100.0
    benign_results = [r for r in results if r.category != "attack"]
    attack_results = [r for r in results if r.category == "attack"]
    fsr = sum(1 for r in benign_results if r.is_false_suppression) / max(1, len(benign_results)) * 100.0
    task_success = sum(1 for r in benign_results if r.action != "Suppress") / max(1, len(benign_results)) * 100.0
    dob_detection_rate = sum(1 for r in attack_results if r.dob.triggered) / max(1, len(attack_results)) * 100.0
    attack_mitigation_rate = sum(1 for r in attack_results if r.action in ("Suppress", "RequireApproval", "RouteToLowCost", "LimitOutput")) / max(1, len(attack_results)) * 100.0
    action_dist = dict(Counter(r.action for r in results))
    dob_dist = dict(Counter(r.dob.severity for r in results if r.dob.triggered))
    cat_breakdown = {}
    for cat in sorted(set(r.category for r in results)):
        cr = [r for r in results if r.category == cat]
        if not cr:
            continue
        suppressed = sum(1 for r in cr if r.action == "Suppress")
        cat_breakdown[cat] = {
            "n": len(cr),
            "suppressed": suppressed,
            "suppressed_pct": round(suppressed / len(cr) * 100, 1),
            "approval_or_higher_pct": round(sum(1 for r in cr if r.action in ("RequireApproval", "Suppress")) / len(cr) * 100, 1),
            "avg_risk": round(statistics.mean(r.risk_score for r in cr), 4),
            "avg_true_cost": round(statistics.mean(r.true_cost for r in cr), 8),
            "avg_executed_cost": round(statistics.mean(r.executed_cost for r in cr), 8),
        }
    return {
        "n_total": len(results),
        "baseline_total_cost_usd": round(baseline_total_cost, 6),
        "proposed_total_cost_usd": round(proposed_total_cost, 6),
        "cost_reduction_pct": round(cost_reduction_pct, 2),
        "budget_violation_rate_baseline_pct": round(bvr_baseline, 2),
        "budget_violation_rate_proposed_pct": round(bvr_proposed, 2),
        "false_suppression_rate_pct": round(fsr, 2),
        "task_success_rate_pct": round(task_success, 2),
        "dob_detection_rate_pct": round(dob_detection_rate, 2),
        "attack_mitigation_rate_pct": round(attack_mitigation_rate, 2),
        "action_distribution": action_dist,
        "dob_severity_distribution": dob_dist,
        "per_category": cat_breakdown,
    }
def evaluate_sensitivity(samples: List[dict], cfg: PolicyConfig) -> Dict[str, Dict[str, float]]:
    variants = {
        "strict": {"tau1": max(0.0, cfg.tau1 * 0.90), "tau2": max(0.0, cfg.tau2 * 0.90), "tau3": max(0.0, cfg.tau3 * 0.90), "tau4": max(0.0, cfg.tau4 * 0.90)},
        "default": {"tau1": cfg.tau1, "tau2": cfg.tau2, "tau3": cfg.tau3, "tau4": cfg.tau4},
        "relaxed": {"tau1": min(0.999, cfg.tau1 * 1.10), "tau2": min(0.999, cfg.tau2 * 1.10), "tau3": min(0.999, cfg.tau3 * 1.10), "tau4": min(0.999, cfg.tau4 * 1.10)},
    }
    out = {}
    for name, vals in variants.items():
        tmp = PolicyConfig(**{**asdict(cfg), **vals})
        res = run_policy_evaluation(samples, tmp)
        met = compute_metrics(res, samples)
        out[name] = {
            "cost_reduction_pct": met["cost_reduction_pct"],
            "false_suppression_rate_pct": met["false_suppression_rate_pct"],
            "budget_violation_rate_proposed_pct": met["budget_violation_rate_proposed_pct"],
            "attack_mitigation_rate_pct": met["attack_mitigation_rate_pct"],
        }
    return out
def print_metrics(metrics: Dict, calibration: CalibrationInfo, sensitivity: Dict[str, Dict[str, float]]):
    print("\n" + "=" * 68)
    print("  AI-MLA-CostGuard: Policy Evaluation Results")
    print("=" * 68)
    print(f"  Total requests               : {metrics['n_total']}")
    print("\n  [Threshold Calibration]")
    print(f"  Mode                         : {calibration.mode}")
    print(f"  Calibrated thresholds        : τ1={calibration.tau1:.4f} τ2={calibration.tau2:.4f} τ3={calibration.tau3:.4f} τ4={calibration.tau4:.4f}")
    print(f"  Risk score summary           : {calibration.score_summary}")
    print("\n  [Cost Reduction]")
    print(f"  Baseline total cost          : ${metrics['baseline_total_cost_usd']:.6f}")
    print(f"  Proposed total cost          : ${metrics['proposed_total_cost_usd']:.6f}")
    print(f"  Cost reduction               : {metrics['cost_reduction_pct']:.2f}%")
    print("\n  [Budget Violation Rate]")
    print(f"  Baseline BVR                 : {metrics['budget_violation_rate_baseline_pct']:.2f}%")
    print(f"  Proposed BVR                 : {metrics['budget_violation_rate_proposed_pct']:.2f}%")
    print("\n  [Safety / Control Metrics]")
    print(f"  False suppression rate       : {metrics['false_suppression_rate_pct']:.2f}%")
    print(f"  Benign task success rate     : {metrics['task_success_rate_pct']:.2f}%")
    print(f"  DoB detection rate (attack)  : {metrics['dob_detection_rate_pct']:.2f}%")
    print(f"  Attack mitigation rate       : {metrics['attack_mitigation_rate_pct']:.2f}%")
    print("\n  [Action Distribution]")
    for action, cnt in sorted(metrics['action_distribution'].items()):
        pct = cnt / max(1, metrics['n_total']) * 100.0
        bar = "█" * int(pct / 2)
        print(f"  {action:<20} {cnt:>5} ({pct:>5.1f}%) {bar}")
    print("\n  [DoB Severity Distribution]")
    if metrics["dob_severity_distribution"]:
        for sev, cnt in sorted(metrics["dob_severity_distribution"].items()):
            print(f"  {sev:<20} {cnt:>5}")
    else:
        print("  None")
    print("\n  [Per-Category Breakdown]")
    print(f"  {'Category':<18} {'n':>5} {'Supp%':>8} {'Approval+%':>11} {'AvgRisk':>9} {'AvgExecCost':>13}")
    print("  " + "-" * 68)
    for cat, v in metrics["per_category"].items():
        print(f"  {cat:<18} {v['n']:>5} {v['suppressed_pct']:>7.1f}% {v['approval_or_higher_pct']:>10.1f}% {v['avg_risk']:>9.4f} ${v['avg_executed_cost']:>12.8f}")
    print("\n  [Sensitivity Summary]")
    print(f"  {'Setting':<10} {'CostRed%':>10} {'FSR%':>10} {'Prop.BVR%':>12} {'AtkMit%':>10}")
    print("  " + "-" * 60)
    for name, v in sensitivity.items():
        print(f"  {name:<10} {v['cost_reduction_pct']:>9.2f}% {v['false_suppression_rate_pct']:>9.2f}% {v['budget_violation_rate_proposed_pct']:>11.2f}% {v['attack_mitigation_rate_pct']:>9.2f}%")
    print("=" * 68)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset_with_predictions.json")
    parser.add_argument("--results_out", default="data/policy_results.json")
    parser.add_argument("--calibration_mode", choices=["fixed", "quantile"], default="quantile")
    args = parser.parse_args()
    print(f"[*] Loading: {args.data}")
    with open(args.data, "r", encoding="utf-8") as f:
        samples = json.load(f)
    cfg = PolicyConfig(calibration_mode=args.calibration_mode)
    calibration = calibrate_thresholds(samples, cfg)
    print(f"[*] PolicyConfig: α={cfg.alpha} β={cfg.beta} γ={cfg.gamma}")
    print(f"[*] Thresholds: τ1={cfg.tau1:.4f} τ2={cfg.tau2:.4f} τ3={cfg.tau3:.4f} τ4={cfg.tau4:.4f}")
    results = run_policy_evaluation(samples, cfg)
    metrics = compute_metrics(results, samples)
    sensitivity = evaluate_sensitivity(samples, cfg)
    print_metrics(metrics, calibration, sensitivity)
    Path(args.results_out).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "config": asdict(cfg),
        "calibration": asdict(calibration),
        "metrics": metrics,
        "sensitivity": sensitivity,
        "results": [r.to_dict() for r in results],
    }
    with open(args.results_out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[✓] Results saved: {args.results_out}")
