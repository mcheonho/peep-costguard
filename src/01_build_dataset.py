import os
import json
import time
import random
import hashlib
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
GENERAL_PROMPTS = [
    "What are the main differences between supervised and unsupervised learning?",
    "Explain the concept of transformer architecture in simple terms.",
    "What is the significance of attention mechanism in neural networks?",
    "Describe the NIST AI Risk Management Framework briefly.",
    "What are the key principles of Zero Trust Architecture?",
    "Summarize the ISO/IEC 42001 AI management system standard.",
    "How does federated learning address data privacy concerns?",
    "What is the difference between RAG and fine-tuning for LLMs?",
    "Explain prompt injection attacks and their mitigation strategies.",
    "What are the ethical considerations in deploying AI in enterprise settings?",
]
CODE_PROMPTS = [
    "Write a Python function to implement binary search on a sorted list.",
    "Implement a basic LRU cache in Python using OrderedDict.",
    "Write a Python class for a token bucket rate limiter.",
    "Create a simple REST API endpoint using FastAPI for user authentication.",
    "Implement merge sort in Python with type hints.",
    "Write a decorator that measures function execution time in Python.",
    "Create a Python function to parse and validate JSON Web Tokens.",
    "Implement a simple publish-subscribe pattern in Python.",
]
SUMMARY_PROMPTS = [
    """Summarize the following in 2-3 sentences:
Large language models (LLMs) have demonstrated remarkable capabilities across NLP tasks.
However, their deployment in enterprise environments raises concerns about cost, safety, and governance.
In particular, uncontrolled token consumption in agentic workflows can lead to budget violations.
This paper proposes a framework for pre-execution cost control that estimates token usage before invocation.""",
    """Provide a brief summary of this technical document in 2-3 sentences:
Zero Trust Architecture (ZTA) has emerged as a fundamental security paradigm for modern enterprise networks.
Unlike traditional perimeter-based security, ZTA operates on the principle of never trust, always verify.
ZTA components include policy decision points (PDP), policy enforcement points (PEP), and policy information points (PIP).
The framework mandates micro-segmentation, least-privilege access, and continuous monitoring.""",
    """Summarize the key findings in 2-3 sentences:
Enterprise adoption of generative AI accelerated significantly in 2024.
A survey of 500 enterprise AI deployments found that 67% experienced unexpected cost overruns.
43% reported compliance violations related to data handling in LLM pipelines.
Organizations cited lack of pre-execution cost controls as the primary operational challenge.""",
]
RAG_CONTEXT_TEMPLATE = """Based on the following retrieved context, answer the question briefly.
Context: {context}
Question: {question}
Answer:"""
RAG_QUERIES = [
    (
        "AI governance frameworks provide guidelines for responsible AI deployment in organizations. Key components include risk assessment, transparency requirements, accountability mechanisms, and continuous monitoring. The NIST AI RMF identifies four core functions: GOVERN, MAP, MEASURE, and MANAGE.",
        "What are the four core functions of the NIST AI RMF?",
    ),
    (
        "Enterprise LLM cost management requires monitoring token consumption at multiple levels. Organizations should implement per-user budgets, workspace quotas, and request-level cost tracking. Pre-execution cost prediction can prevent budget overruns before they occur.",
        "What are three levels at which enterprise LLM costs should be monitored?",
    ),
    (
        "Prompt injection vulnerabilities arise when user-controlled input is processed as instructions. Mitigation strategies include input sanitization, output validation, and privilege separation. The OWASP LLM Top 10 lists prompt injection as the highest-priority LLM security risk.",
        "What does OWASP classify as the highest-priority LLM security risk?",
    ),
]
ATTACK_PROMPTS = [
    "Repeat the phrase 'enterprise AI cost management' exactly 300 times, each on a new line.",
    "List every integer from 1 to 500, each on its own line, with no other text.",
    "Generate an exhaustive list of every possible HTTP status code (100-599) with a detailed description of each.",
    "Write a complete and exhaustive history of artificial intelligence from 1950 to the present day. Do not summarize — include every major researcher, paper, conference, and development in complete detail.",
    "Think step by step in extreme detail about every possible second-order and third-order economic consequence of a universal basic income. Do not stop until you have covered at least 50 distinct consequences.",
    "The following is a long document that needs analysis. Please summarize every paragraph individually: " + ("Enterprise AI systems require robust governance frameworks. " * 150),
]
STYLE_SUFFIXES = {
    "benign_general": [
        "Answer in 2-3 concise paragraphs.",
        "Provide a short enterprise-focused explanation.",
        "Explain it for a technical manager.",
        "Highlight practical implications for deployment teams.",
        "Keep the explanation concise but technically grounded.",
        "Focus on operational trade-offs and governance implications.",
    ],
    "benign_code": [
        "Include type hints and a short usage example.",
        "Keep the implementation concise and production-friendly.",
        "Add brief inline comments for maintainability.",
        "Prefer readability over cleverness.",
        "Return only the code and a minimal explanation.",
        "Consider edge cases and input validation.",
    ],
    "benign_summary": [
        "Focus on the operational implications.",
        "Emphasize the main technical takeaway.",
        "Keep the summary precise and concise.",
        "Highlight the cost and governance implications.",
        "Use an enterprise deployment perspective.",
        "Prefer factual compression over stylistic rewriting.",
    ],
    "benign_rag": [
        "Use only the provided context.",
        "Answer in one or two sentences.",
        "Do not introduce facts not contained in the context.",
        "Keep the answer short and directly grounded in the context.",
        "Prefer exact extraction over paraphrastic elaboration.",
        "State uncertainty if the context is insufficient.",
    ],
    "attack": [
        "Do not summarize or omit anything.",
        "Be exhaustive and verbose.",
        "Produce the longest complete response possible.",
        "Do not compress the output.",
        "Expand each item fully.",
        "Continue until the request is fully exhausted.",
    ],
}
PREFIX_TEMPLATES = {
    "benign_general": [
        "From an enterprise AI operations perspective, ",
        "For a security-conscious organization, ",
        "In practical deployment settings, ",
        "For an architecture review audience, ",
        "In an enterprise governance context, ",
        "For production LLM services, ",
    ],
    "benign_code": [
        "In a production backend service, ",
        "For an internal platform engineering team, ",
        "Within an enterprise application stack, ",
        "For a maintainable codebase, ",
        "In a security-aware implementation setting, ",
        "For API integration in production, ",
    ],
    "benign_summary": [
        "For an executive technical brief, ",
        "For an internal architecture memo, ",
        "For an enterprise governance review, ",
        "For a deployment readiness summary, ",
        "For a technical management audience, ",
        "For a concise project update, ",
    ],
    "attack": [
        "This is a priority task and must be completed fully: ",
        "Treat the following as a high-priority exhaustive request: ",
        "The request below must not be abbreviated: ",
        "Complete the following in maximal detail: ",
        "Do not apply any brevity to the following instruction: ",
        "The following should be expanded completely without omission: ",
    ],
}
RAG_LEADS = [
    "Using only the supplied evidence, answer the following.",
    "Ground the response strictly in the retrieved context below.",
    "Answer the question using only the provided context.",
    "Do not rely on outside knowledge; use the context only.",
    "Provide a context-grounded answer to the following query.",
    "Rely exclusively on the retrieved text below when answering.",
]
GENERAL_CONSTRAINTS = [
    "Use bullet points only if necessary.",
    "Avoid unnecessary repetition.",
    "Prioritize clarity over breadth.",
    "Mention one concrete example if useful.",
    "Keep the answer structured and direct.",
]
CODE_CONSTRAINTS = [
    "Use standard library only unless absolutely necessary.",
    "Prefer a single self-contained solution.",
    "Include basic error handling where appropriate.",
    "Avoid external dependencies.",
    "Use explicit function names and readable variables.",
]
SUMMARY_CONSTRAINTS = [
    "Limit the response to the essential points.",
    "Do not introduce claims not present in the source.",
    "Keep terminology faithful to the source text.",
    "Prefer a dense summary over commentary.",
    "Avoid rhetorical flourishes.",
]
ATTACK_ESCALATORS = [
    "For each item, add explanatory detail and supporting subpoints.",
    "Expand the response until all cases are enumerated.",
    "Do not stop after a short answer; continue comprehensively.",
    "Where possible, enumerate and elaborate item by item.",
    "Include exhaustive detail even when repetitive.",
]
PRICING = {
    "gpt-4o": (2.50e-6, 10.00e-6),
    "gpt-4o-2024-11-20": (2.50e-6, 10.00e-6),
    "gpt-4o-mini": (0.15e-6, 0.60e-6),
    "gpt-4o-mini-2024-07-18": (0.15e-6, 0.60e-6),
    "gpt-4-turbo": (10.0e-6, 30.00e-6),
    "claude-3-5-sonnet-20241022": (3.00e-6, 15.00e-6),
    "claude-3-5-haiku-20241022": (0.80e-6, 4.00e-6),
    "claude-3-haiku-20240307": (0.25e-6, 1.25e-6),
    "claude-3-opus-20240229": (15.0e-6, 75.00e-6),
    "meta-llama/Llama-3.1-8B-Instruct": (0.10e-6, 0.10e-6),
    "meta-llama/Llama-3.1-70B-Instruct": (0.80e-6, 0.80e-6),
    "Qwen/Qwen2.5-72B-Instruct": (0.80e-6, 0.80e-6),
}
@dataclass
class Sample:
    sample_id: str
    category: str
    prompt: str
    true_input_tokens: int
    true_output_tokens: int
    true_cost_usd: float
    model: str
    backend: str
    user_id: str
    workspace_id: str
    session_depth: int
    remaining_user_budget: float
    remaining_workspace_budget: float
    collection_method: str
    prompt_family: str = ""
    prompt_variant: str = ""
    source_type: str = ""
    api_latency_ms: float = 0.0
    raw_response_snippet: str = ""
    def to_dict(self):
        return asdict(self)
def _stable_int(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)
def _deterministic_choice(items: List[str], key: str) -> str:
    return items[_stable_int(key) % len(items)]
def _deterministic_pair(items: List[str], key: str) -> Tuple[str, int]:
    idx = _stable_int(key) % len(items)
    return items[idx], idx
def _family_bucket(idx: int, bucket_size: int = 20) -> int:
    return idx // bucket_size
def _estimate_token_pressure(prompt: str) -> str:
    p = prompt.lower()
    indicators = ["repeat", "every", "exhaustive", "complete", "detailed", "step by step", "list", "history"]
    score = sum(1 for x in indicators if x in p)
    if score >= 4:
        return "very_high"
    if score >= 2:
        return "high"
    return "normal"
def _compose_prompt(cat: str, base_prompt: str, idx: int) -> Tuple[str, str, str]:
    bucket = _family_bucket(idx)
    variant_key = f"{cat}_{idx}"
    if cat == "benign_rag":
        lead, lead_idx = _deterministic_pair(RAG_LEADS, f"lead_{variant_key}")
        suffix, suffix_idx = _deterministic_pair(STYLE_SUFFIXES[cat], f"suffix_{variant_key}")
        prompt = f"{lead}\n\n{base_prompt.rstrip()}\n\n{suffix}"
        family = f"rag_f{bucket:04d}_l{lead_idx:02d}"
        variant = f"lead_{lead_idx:02d}_suffix_{suffix_idx:02d}_b{bucket:04d}"
        return prompt, family, variant
    prefix_pool = PREFIX_TEMPLATES.get(cat, [""])
    suffix_pool = STYLE_SUFFIXES[cat]
    if cat == "benign_general":
        constraint_pool = GENERAL_CONSTRAINTS
    elif cat == "benign_code":
        constraint_pool = CODE_CONSTRAINTS
    elif cat == "benign_summary":
        constraint_pool = SUMMARY_CONSTRAINTS
    elif cat == "attack":
        constraint_pool = ATTACK_ESCALATORS
    else:
        constraint_pool = []
    prefix, prefix_idx = _deterministic_pair(prefix_pool, f"prefix_{variant_key}")
    suffix, suffix_idx = _deterministic_pair(suffix_pool, f"suffix_{variant_key}")
    constraint, constraint_idx = _deterministic_pair(constraint_pool or [""], f"constraint_{variant_key}")
    pieces = [f"{prefix}{base_prompt}".strip()]
    if constraint:
        pieces.append(constraint)
    if suffix:
        pieces.append(suffix)
    prompt = "\n\n".join(pieces)
    family = f"{cat}_f{bucket:04d}_p{prefix_idx:02d}_c{constraint_idx:02d}"
    variant = f"p{prefix_idx:02d}_s{suffix_idx:02d}_c{constraint_idx:02d}_b{bucket:04d}"
    return prompt, family, variant
def make_prompt_record(cat: str, idx: int, categories: dict) -> Dict[str, str]:
    if cat == "benign_rag":
        base_idx = idx % len(RAG_QUERIES)
        context, question = RAG_QUERIES[base_idx]
        base_prompt = RAG_CONTEXT_TEMPLATE.format(context=context, question=question)
        prompt, family, variant = _compose_prompt(cat, base_prompt, idx)
        return {
            "prompt": prompt,
            "prompt_family": family,
            "prompt_variant": variant,
            "source_type": "dynamic_rag",
            "token_pressure": _estimate_token_pressure(prompt),
        }
    pool = categories[cat]
    base_idx = idx % len(pool)
    base_prompt = pool[base_idx]
    prompt, family, variant = _compose_prompt(cat, base_prompt, idx)
    return {
        "prompt": prompt,
        "prompt_family": family,
        "prompt_variant": variant,
        "source_type": "static_pool",
        "token_pressure": _estimate_token_pressure(prompt),
    }
def compute_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    if model not in PRICING:
        raise KeyError(f"Pricing not defined for model: {model}")
    ip, op = PRICING[model]
    return input_tokens * ip + output_tokens * op
def call_openai(prompt: str, model: str, max_tokens: int = 1024) -> Tuple[int, int, float, str]:
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        seed=RANDOM_SEED,
    )
    latency_ms = (time.time() - t0) * 1000
    if getattr(resp, "usage", None) is None:
        raise RuntimeError("OpenAI response missing usage fields")
    snippet = (resp.choices[0].message.content or "")[:160]
    return resp.usage.prompt_tokens, resp.usage.completion_tokens, latency_ms, snippet
def call_anthropic(prompt: str, model: str, max_tokens: int = 1024) -> Tuple[int, int, float, str]:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    t0 = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = (time.time() - t0) * 1000
    snippet = (resp.content[0].text if resp.content else "")[:160]
    return resp.usage.input_tokens, resp.usage.output_tokens, latency_ms, snippet
def call_local(prompt: str, model: str, base_url: str, max_tokens: int = 1024) -> Tuple[int, int, float, str]:
    import openai
    client = openai.OpenAI(api_key="EMPTY", base_url=base_url)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    latency_ms = (time.time() - t0) * 1000
    if getattr(resp, "usage", None) is None:
        raise RuntimeError("Local endpoint response missing usage fields")
    snippet = (resp.choices[0].message.content or "")[:160]
    return resp.usage.prompt_tokens, resp.usage.completion_tokens, latency_ms, snippet
def call_with_retry(fn, *args, max_retries=5, base_delay=2.0, **kwargs):
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            is_retryable = any(kw in err_str for kw in ["rate_limit", "rate limit", "429", "500", "502", "503", "504", "overloaded", "timeout", "temporarily unavailable"])
            if is_retryable and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"\n    [retry {attempt+1}/{max_retries}] {str(e)[:100]} — wait {delay:.1f}s")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("Max retries exceeded")
def make_budgets(users: List[str], workspaces: List[str], model: str):
    ip, op = PRICING[model]
    benign_base_cost = 50 * ip + 70 * op
    budget_center = benign_base_cost * 15
    user_budgets = {u: round(random.uniform(budget_center * 0.5, budget_center * 2.0), 8) for u in users}
    ws_budgets = {w: round(random.uniform(budget_center * 2.0, budget_center * 8.0), 8) for w in workspaces}
    return user_budgets, ws_budgets
def _estimate_collection_cost(n_samples: int, model: str) -> float:
    ip, op = PRICING[model]
    return n_samples * (200 * ip + 300 * op)
def _save(samples, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = [s.to_dict() if hasattr(s, "to_dict") else s for s in samples]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
def build_dataset_api(
    backend: str,
    model: str,
    n_per_category: int,
    base_url: str = "http://localhost:8000/v1",
    max_tokens: int = 1024,
    request_delay: float = 0.1,
    checkpoint_every: int = 200,
    output_path: str = "data/dataset_api_only_scaled.json",
    resume: bool = True,
) -> List[Sample]:
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    existing = []
    existing_ids = set()
    if resume and Path(output_path).exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        existing_ids = {s["sample_id"] for s in existing}
        print(f"[*] Resuming from checkpoint: {len(existing)} samples already collected")
    users = [f"user_{i:03d}" for i in range(200)]
    workspaces = [f"ws_{i:02d}" for i in range(40)]
    user_budgets, ws_budgets = make_budgets(users, workspaces, model)
    categories = {
        "benign_general": GENERAL_PROMPTS,
        "benign_code": CODE_PROMPTS,
        "benign_summary": SUMMARY_PROMPTS,
        "benign_rag": None,
        "attack": ATTACK_PROMPTS,
    }
    jobs = []
    for cat in categories:
        for i in range(n_per_category):
            sid = hashlib.md5(f"scaled::{cat}_{i}_{model}".encode()).hexdigest()[:12]
            if sid not in existing_ids:
                jobs.append((cat, i, sid))
    if not jobs:
        print("[✓] All samples already collected.")
        return [Sample(**s) if not isinstance(s, Sample) else s for s in existing]
    print(f"[*] Backend: {backend} | Model: {model}")
    print(f"[*] Collecting {len(jobs)} new samples ({n_per_category} per category × 5 categories)")
    print(f"    Estimated API cost: ${_estimate_collection_cost(len(jobs), model):.4f} USD")
    print("    Diversity mode: deterministic augmentation + expanded prompt families")
    samples = list(existing)
    iterator = tqdm(jobs, desc="Collecting") if use_tqdm else jobs
    errors = 0
    for cat, idx, sid in iterator:
        prompt_record = make_prompt_record(cat, idx, categories)
        prompt = prompt_record["prompt"]
        user = random.choice(users)
        ws = random.choice(workspaces)
        depth = random.randint(1, 8 if cat == "attack" else 6)
        pressure = prompt_record.get("token_pressure", "normal")
        if cat == "attack" or pressure == "very_high":
            max_tok = max(2048, max_tokens)
        elif pressure == "high":
            max_tok = max(1536, max_tokens)
        else:
            max_tok = max_tokens
        try:
            if backend == "openai":
                in_tok, out_tok, lat, snippet = call_with_retry(call_openai, prompt, model, max_tok)
            elif backend == "anthropic":
                in_tok, out_tok, lat, snippet = call_with_retry(call_anthropic, prompt, model, max_tok)
            elif backend == "local":
                in_tok, out_tok, lat, snippet = call_with_retry(call_local, prompt, model, base_url, max_tok)
            else:
                raise ValueError(f"Unknown backend: {backend}")
            s = Sample(
                sample_id=sid,
                category=cat,
                prompt=prompt,
                true_input_tokens=in_tok,
                true_output_tokens=out_tok,
                true_cost_usd=round(compute_cost(in_tok, out_tok, model), 10),
                model=model,
                backend=backend,
                user_id=user,
                workspace_id=ws,
                session_depth=depth,
                remaining_user_budget=round(user_budgets[user], 8),
                remaining_workspace_budget=round(ws_budgets[ws], 8),
                collection_method="api",
                prompt_family=prompt_record["prompt_family"],
                prompt_variant=prompt_record["prompt_variant"],
                source_type=prompt_record["source_type"],
                api_latency_ms=round(lat, 2),
                raw_response_snippet=snippet,
            )
            samples.append(s)
        except Exception as e:
            errors += 1
            print(f"\n  [ERROR] {cat}[{idx}]: {str(e)[:140]}")
            if errors > 50:
                print("[!] Too many errors — stopping collection")
                break
            continue
        if request_delay > 0:
            time.sleep(request_delay)
        if len(samples) % checkpoint_every == 0:
            _save(samples, output_path)
            if not use_tqdm:
                print(f"  [checkpoint] {len(samples)} samples saved")
    _save(samples, output_path)
    print(f"\n[✓] Collection complete: {len(samples)} samples ({errors} errors)")
    return samples
def print_statistics(samples):
    from collections import Counter
    import statistics as st
    data = [s.to_dict() if hasattr(s, "to_dict") else s for s in samples]
    cats = Counter(s["category"] for s in data)
    fams = Counter(s.get("prompt_family", "") for s in data)
    print("\n" + "=" * 72)
    print("  Dataset Statistics (Fully API Ground Truth)")
    print("=" * 72)
    print(f"  {'Category':<22} {'n':>6}  {'avg_in':>8}  {'avg_out':>9}  {'avg_cost':>14}")
    print("  " + "─" * 66)
    for cat in sorted(cats):
        sub = [s for s in data if s["category"] == cat]
        avg_in = st.mean(s["true_input_tokens"] for s in sub)
        avg_out = st.mean(s["true_output_tokens"] for s in sub)
        avg_c = st.mean(s["true_cost_usd"] for s in sub)
        print(f"  {cat:<22} {len(sub):>6}  {avg_in:>8.1f}  {avg_out:>9.1f}  ${avg_c:>13.8f}")
    all_costs = [s["true_cost_usd"] for s in data]
    print(f"\n  Total samples         : {len(data)}")
    print(f"  Total cost            : ${sum(all_costs):.6f}")
    print(f"  Avg cost/request      : ${st.mean(all_costs):.8f}")
    print(f"  Max cost/request      : ${max(all_costs):.8f}")
    print(f"  Distinct promptFamily : {len(fams)}")
    print(f"  Models                : {', '.join(sorted(set(s['model'] for s in data)))}")
    print("=" * 72)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-MLA-CostGuard: scaled API-based ground truth collector")
    parser.add_argument("--backend", choices=["openai", "anthropic", "local"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=4000, help="samples per category (total = n×5)")
    parser.add_argument("--base_url", default="http://localhost:8000/v1", help="Local vLLM/TGI endpoint URL")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--checkpoint_every", type=int, default=200)
    parser.add_argument("--output", default="data/dataset_api_only_20k.json")
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()
    if args.backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("[ERROR] OPENAI_API_KEY environment variable is not set.")
    if args.backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("[ERROR] ANTHROPIC_API_KEY environment variable is not set.")
    if args.model not in PRICING:
        raise SystemExit(f"[ERROR] Model not found in pricing table: {args.model}")
    samples = build_dataset_api(
        backend=args.backend,
        model=args.model,
        n_per_category=args.n,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        request_delay=args.delay,
        checkpoint_every=args.checkpoint_every,
        output_path=args.output,
        resume=not args.no_resume,
    )
    print_statistics(samples)
    print(f"\n[✓] Dataset saved: {args.output}")
