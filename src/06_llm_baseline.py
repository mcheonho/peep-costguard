from __future__ import annotations
import argparse
import json
import os
import time
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple
GSS_MAPE_NORM = 60.0
GSS_LAT_NORM = 500.0
GSS_STAB_NORM = 100.0
PRICING = {
    "gpt-4o-mini": (0.15e-6, 0.60e-6),
    "gpt-4o": (2.50e-6, 10.00e-6),
    "claude-haiku-4-5": (1.00e-6, 5.00e-6),
    "claude-haiku-4-5-20251001": (1.00e-6, 5.00e-6),
    "claude-3-haiku-20240307": (0.25e-6, 1.25e-6),
    "claude-3-5-sonnet-20241022": (3.00e-6, 15.00e-6),
}
STRONG_FEWSHOT_SYSTEM = (
    "You are a runtime governance pre-execution estimator for an enterprise LLM system.\n"
    "Estimate how many tokens an LLM will consume for a request BEFORE execution.\n"
    "\n"
    "Estimation rules:\n"
    "- input_tokens = prompt character count / 3.8 (round to nearest integer)\n"
    "- output_tokens depends on task type and output-triggering signals:\n"
    "  * rag: 15-40 tokens (short factual answers)\n"
    "  * summary: 40-120 tokens\n"
    "  * general: 150-500 tokens\n"
    "  * code: 200-512 tokens\n"
    "  * attack with repeat/exhaustive/list-all signals: 800-2048 tokens\n"
    "  * attack with large context but NO repeat signals: short output (<100)\n"
    "\n"
    "Calibration examples (12, covering category boundaries):\n"
    "Ex01 rag, char=430                         -> {input:113, output:22, confidence:high}\n"
    "Ex02 rag, char=900 (long context)          -> {input:237, output:30, confidence:high}\n"
    "Ex03 summary, char=510                      -> {input:134, output:55, confidence:high}\n"
    "Ex04 summary, char=1200 (long doc)          -> {input:316, output:85, confidence:medium}\n"
    "Ex05 general, char=148                      -> {input:39,  output:300, confidence:medium}\n"
    "Ex06 general, char=300 (explain in depth)   -> {input:79,  output:430, confidence:medium}\n"
    "Ex07 code, char=152                         -> {input:40,  output:240, confidence:medium}\n"
    "Ex08 code, char=320 (with usage example)    -> {input:84,  output:380, confidence:medium}\n"
    "Ex09 attack, char=275, signals=repeat+300   -> {input:72,  output:1800, confidence:high}\n"
    "Ex10 attack, char=180, signals=list-all     -> {input:47,  output:1500, confidence:high}\n"
    "Ex11 attack, char=4870, no repeat signals   -> {input:1281, output:70, confidence:medium}\n"
    "Ex12 attack, char=600, signals=exhaustive   -> {input:158, output:1600, confidence:high}\n"
    "\n"
    "Respond ONLY with valid JSON. No markdown, no explanation.\n"
    "Format: {\"estimated_input_tokens\": <int>, \"estimated_output_tokens\": <int>, \"confidence\": \"low|medium|high\"}"
)
ZEROSHOT_SYSTEM = (
    "You are a runtime governance pre-execution estimator for an enterprise LLM system.\n"
    "Estimate how many tokens an LLM will consume for a request BEFORE execution.\n"
    "Respond ONLY with valid JSON. No markdown, no explanation.\n"
    "Format: {\"estimated_input_tokens\": <int>, \"estimated_output_tokens\": <int>, \"confidence\": \"low|medium|high\"}"
)
USER_TEMPLATE = (
    "Estimate token usage for the following LLM request before execution.\n\n"
    "Task type: {task_type}\n"
    "Session depth: {session_depth}\n"
    "Prompt character count: {char_count}\n"
    "Prompt word count: {word_count}\n"
    "Attack signal keywords: {attack_signals}\n"
    "Prompt (first 400 chars):\n{prompt_snippet}\n\n"
    "Respond with JSON only."
)
ATTACK_KWS = ["repeat", "exhaustive", "every", "list all", "do not stop",
              "keep generating", "1000", "500", "300", "step by step",
              "complete history", "summarize every"]
def build_prompt(sample: dict, shot: str) -> Tuple[str, str]:
    prompt_text = str(sample.get("prompt", ""))
    task_type = str(sample.get("category", "general")).replace("benign_", "")
    session_depth = int(sample.get("session_depth", 1) or 1)
    char_count = len(prompt_text)
    word_count = len(prompt_text.split())
    snippet = prompt_text[:400].replace('"', "'").replace("\n", " ")
    detected = [kw for kw in ATTACK_KWS if kw in prompt_text.lower()]
    attack_signals = ", ".join(detected) if detected else "none"
    user = USER_TEMPLATE.format(
        task_type=task_type, session_depth=session_depth,
        char_count=char_count, word_count=word_count,
        attack_signals=attack_signals, prompt_snippet=snippet,
    )
    system = STRONG_FEWSHOT_SYSTEM if shot == "strong_few" else ZEROSHOT_SYSTEM
    return system, user
def parse_json(text: str) -> Optional[Dict]:
    text = (text or "").strip()
    for fence in ["```json", "```JSON", "```"]:
        if fence in text:
            text = text.split(fence, 1)[-1].rsplit("```", 1)[0].strip()
            break
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        return json.loads(text[s:e + 1])
    except Exception:
        return None
def call_openai(system, user, model, api_key):
    import openai
    client = openai.OpenAI(api_key=api_key)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=128, temperature=0.0, seed=42,
    )
    lat = (time.perf_counter() - t0) * 1000.0
    raw = (resp.choices[0].message.content or "").strip()
    return parse_json(raw), lat, getattr(resp.usage, "prompt_tokens", 0), getattr(resp.usage, "completion_tokens", 0)
def call_anthropic(system, user, model, api_key):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=model, max_tokens=128, system=system,
        messages=[{"role": "user", "content": user}],
    )
    lat = (time.perf_counter() - t0) * 1000.0
    raw = (resp.content[0].text if resp.content else "").strip()
    return parse_json(raw), lat, getattr(resp.usage, "input_tokens", 0), getattr(resp.usage, "output_tokens", 0)
def call_with_retry(fn, *args, max_retries=3, base_delay=2.0):
    import random
    for attempt in range(max_retries):
        try:
            return fn(*args)
        except Exception as exc:
            msg = str(exc).lower()
            retryable = any(k in msg for k in
                            ["rate", "429", "500", "502", "503", "overloaded", "timeout"])
            if retryable and attempt < max_retries - 1:
                d = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"      [retry {attempt+1}] {str(exc)[:60]} — wait {d:.1f}s")
                time.sleep(d)
            else:
                raise
    raise RuntimeError("max retries")
PROVIDER_FN = {"openai": call_openai, "anthropic": call_anthropic}
PROVIDER_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
def mape(pairs):
    vp = [(t, p) for t, p in pairs if t > 0]
    return sum(abs(t - p) / t for t, p in vp) / len(vp) * 100 if vp else 0.0
def median_ape(pairs):
    vp = [abs(t - p) / t * 100 for t, p in pairs if t > 0]
    return statistics.median(vp) if vp else 0.0
def p90_ape(pairs):
    vp = sorted(abs(t - p) / t * 100 for t, p in pairs if t > 0)
    if not vp:
        return 0.0
    k = min(len(vp) - 1, int(0.9 * len(vp)))
    return vp[k]
def gss(cost_mape, lat_mean, lat_stdev):
    s_acc = max(0.0, 1.0 - cost_mape / GSS_MAPE_NORM)
    s_lat = max(0.0, 1.0 - lat_mean / GSS_LAT_NORM)
    s_stab = max(0.0, 1.0 - lat_stdev / GSS_STAB_NORM)
    return (s_acc + s_lat + s_stab) / 3.0
def eval_cell(provider, model, shot, samples, api_key, save_samples=False):
    pin, pout = PRICING.get(model, (0.15e-6, 0.60e-6))
    fn = PROVIDER_FN[provider]
    cost_pairs, out_pairs, in_pairs, latencies, call_costs = [], [], [], [], []
    per_sample = []
    n_fail = 0
    for i, s in enumerate(samples, 1):
        system, user = build_prompt(s, shot)
        try:
            parsed, lat, call_in, call_out = call_with_retry(fn, system, user, model, api_key)
        except Exception as exc:
            print(f"      [skip {i}] {str(exc)[:60]}")
            n_fail += 1
            continue
        if not parsed:
            n_fail += 1
            continue
        est_in = float(parsed.get("estimated_input_tokens", 0) or 0)
        est_out = float(parsed.get("estimated_output_tokens", 0) or 0)
        est_cost = est_in * pin + est_out * pout
        true_in = float(s.get("true_input_tokens", 0.0))
        true_out = float(s.get("true_output_tokens", 0.0))
        true_cost = float(s.get("true_cost_usd", 0.0))
        cost_pairs.append((true_cost, est_cost))
        out_pairs.append((true_out, est_out))
        in_pairs.append((true_in, est_in))
        latencies.append(lat)
        call_costs.append(call_in * pin + call_out * pout)
        if save_samples:
            per_sample.append({
                "category": s.get("category"),
                "true_in": true_in, "est_in": est_in,
                "true_out": true_out, "est_out": est_out,
            })
        if i % 20 == 0:
            print(f"      ... {i}/{len(samples)} done")
    lat_mean = statistics.mean(latencies) if latencies else float("nan")
    lat_stdev = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    n_success = len(latencies)
    min_required = max(1, int(0.5 * len(samples)))
    valid = n_success >= min_required
    def stats3(pairs):
        return {
            "mean": round(mape(pairs), 2),
            "median": round(median_ape(pairs), 2),
            "p90": round(p90_ape(pairs), 2),
        }
    result = {
        "provider": provider, "model": model, "shot": shot,
        "n_eval": len(samples), "n_success": n_success, "n_fail": n_fail,
        "valid": valid,
        "cost_mape_pct": round(mape(cost_pairs), 2) if valid else None,
        "output_mape_pct": round(mape(out_pairs), 2) if valid else None,
        "input_mape_pct": round(mape(in_pairs), 2) if valid else None,
        "cost_ape": stats3(cost_pairs) if valid else None,
        "output_ape": stats3(out_pairs) if valid else None,
        "input_ape": stats3(in_pairs) if valid else None,
        "latency_mean_ms": round(lat_mean, 2) if valid else None,
        "latency_stdev_ms": round(lat_stdev, 2) if valid else None,
        "inference_cost_per_call_usd": round(statistics.mean(call_costs), 8) if (valid and call_costs) else None,
        "gss": round(gss(mape(cost_pairs), lat_mean, lat_stdev), 4) if valid else None,
    }
    if save_samples:
        result["per_sample"] = per_sample
    return result
def lightgbm_reference(samples):
    cost_pairs = [(float(s.get("true_cost_usd", 0)), float(s.get("pred_cost_usd", 0))) for s in samples]
    out_pairs = [(float(s.get("true_output_tokens", 0)), float(s.get("pred_output_tokens", 0))) for s in samples]
    in_pairs = [(float(s.get("true_input_tokens", 0)), float(s.get("pred_input_tokens", 0))) for s in samples]
    def stats3(pairs):
        return {"mean": round(mape(pairs), 2),
                "median": round(median_ape(pairs), 2),
                "p90": round(p90_ape(pairs), 2)}
    return {
        "model": "LightGBM (PEEP)", "shot": "n/a", "n_eval": len(samples),
        "cost_mape_pct": round(mape(cost_pairs), 2),
        "output_mape_pct": round(mape(out_pairs), 2),
        "input_mape_pct": round(mape(in_pairs), 2),
        "cost_ape": stats3(cost_pairs),
        "output_ape": stats3(out_pairs),
        "input_ape": stats3(in_pairs),
        "inference_cost_per_call_usd": 0.0,
        "note": "latency is reported by 04_ablation_ci.py",
    }
def parse_backends(specs):
    return [(s.split(":", 1)[0], s.split(":", 1)[1]) for s in specs]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset_with_predictions.json")
    ap.add_argument("--backends", nargs="+",
                    default=["openai:gpt-4o-mini", "anthropic:claude-haiku-4-5-20251001"])
    ap.add_argument("--shots", nargs="+", default=["zero", "strong_few"])
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--out_json", default="data/llm_strong/llm_matrix.json")
    ap.add_argument("--save_samples", action="store_true",
                    help="save per-sample estimates for diagnostics")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    with open(args.data, "r", encoding="utf-8") as f:
        samples = json.load(f)
    backends = parse_backends(args.backends)
    print(f"[*] Backends: {backends}")
    print(f"[*] Shots: {args.shots}  | n_eval: {args.n_eval}")
    cats = sorted({s.get("category") for s in samples})
    per = max(1, args.n_eval // len(cats))
    selected = []
    for c in cats:
        selected.extend([s for s in samples if s.get("category") == c][:per])
    selected = selected[:args.n_eval]
    print(f"[*] Selected {len(selected)} stratified samples ({per}/cat × {len(cats)})")
    peep = lightgbm_reference(selected)
    print(f"\n[*] LightGBM (PEEP) reference: cost MAPE={peep['cost_mape_pct']}%  output MAPE={peep['output_mape_pct']}%")
    if args.dry_run:
        sysp, _ = build_prompt(selected[0], "strong_few")
        print("\n[DRY RUN] strong_few system prompt preview:")
        print(sysp[:300] + " ...")
        return
    need = {prov for prov, _ in backends}
    for prov in need:
        env = PROVIDER_KEY_ENV[prov]
        if not os.environ.get(env):
            print(f"\n[!] {env} is not set; cannot run {prov} backend")
            print(f"    Windows:  set {env}=...")
            return
    cells = []
    for prov, model in backends:
        api_key = os.environ[PROVIDER_KEY_ENV[prov]]
        for shot in args.shots:
            print(f"\n{'='*60}\n  {prov}:{model}  shot={shot}\n{'='*60}")
            cell = eval_cell(prov, model, shot, selected, api_key, save_samples=args.save_samples)
            cells.append(cell)
            if cell.get("valid", True):
                print(f"  -> cost MAPE={cell['cost_mape_pct']}%  lat={cell['latency_mean_ms']}ms  "
                      f"GSS={cell['gss']}  (fail {cell['n_fail']})")
            else:
                print(f"  -> INVALID: {cell['n_success']}/{cell['n_eval']} succeeded "
                      f"(fail {cell['n_fail']}). Excluded from results.")
    print("\n" + "=" * 100)
    print("  LLM estimator matrix vs LightGBM (PEEP)   [APE: mean / median]")
    print("=" * 100)
    print(f"  {'Backend':<30}{'Shot':<11}{'OutMAPE(mn/md)':>18}{'CostMAPE(mn/md)':>20}{'Lat(ms)':>10}{'GSS':>8}")
    print("  " + "-" * 96)
    def fmt(ape):
        return f"{ape['mean']}/{ape['median']}"
    po, pc = peep["output_ape"], peep["cost_ape"]
    print(f"  {'LightGBM (PEEP)':<30}{'n/a':<11}{fmt(po):>18}{fmt(pc):>20}{'~ms':>10}{'~0.93':>8}")
    for c in cells:
        bk = f"{c['provider'][:3]}:{c['model'][:18]}"
        if not c.get("valid", True):
            print(f"  {bk:<30}{c['shot']:<11}{'INVALID':>18}{'-':>20}{'-':>10}{'-':>8}")
            continue
        print(f"  {bk:<30}{c['shot']:<11}{fmt(c['output_ape']):>18}{fmt(c['cost_ape']):>20}"
              f"{c['latency_mean_ms']:>10}{c['gss']:>8}")
    print("=" * 100)
    print("  Note: large mean cost errors are driven by input-token overestimation; median reflects typical error.")
    n_invalid = sum(1 for c in cells if not c.get("valid", True))
    if n_invalid:
        print(f"  [!] {n_invalid} cell(s) INVALID — success rate below 50%.")
    out = {"n_eval": len(selected), "peep_reference": peep, "cells": cells,
           "config": {"backends": args.backends, "shots": args.shots, "fewshot_examples": 12}}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved: {args.out_json}")
if __name__ == "__main__":
    main()
