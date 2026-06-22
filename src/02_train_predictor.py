import json
import math
import random
import argparse
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
TASK_KEYWORDS = {
    "code": ["python", "function", "implement", "class", "algorithm", "code", "write a"],
    "summary": ["summarize", "summary", "brief", "shorten", "condense", "tldr"],
    "rag": ["context", "retrieved", "based on the", "given the following", "documents"],
    "attack": ["repeat", "1000", "500", "300", "exhaustive", "every", "do not stop", "keep generating"],
    "general": [],
}
PRICING = {
    "gpt-4o": (2.50e-6, 10.00e-6),
    "gpt-4o-2024-11-20": (2.50e-6, 10.00e-6),
    "gpt-4o-mini": (0.15e-6, 0.60e-6),
    "gpt-4o-mini-2024-07-18": (0.15e-6, 0.60e-6),
    "gpt-4-turbo": (10.0e-6, 30.00e-6),
    "claude-3-haiku-20240307": (0.25e-6, 1.25e-6),
    "claude-3-5-haiku-20241022": (0.80e-6, 4.00e-6),
    "claude-3-5-sonnet-20241022": (3.00e-6, 15.00e-6),
    "claude-haiku-4-5": (1.00e-6, 5.00e-6),
    "claude-haiku-4-5-20251001": (1.00e-6, 5.00e-6),
    "claude-3-opus-20240229": (15.0e-6, 75.0e-6),
    "meta-llama/Llama-3.1-8B-Instruct": (0.10e-6, 0.10e-6),
    "meta-llama/Llama-3.1-70B-Instruct": (0.80e-6, 0.80e-6),
    "Qwen/Qwen2.5-72B-Instruct": (0.80e-6, 0.80e-6),
    "offline": (0.15e-6, 0.60e-6),
}
def extract_features(sample: dict) -> np.ndarray:
    prompt = sample["prompt"]
    words = prompt.split()
    chars = list(prompt)
    prompt_lower = prompt.lower()
    p_char_count = len(prompt)
    p_word_count = len(words)
    p_sentence_count = max(1, prompt.count('.') + prompt.count('?') + prompt.count('!'))
    p_avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    p_unique_ratio = len(set(words)) / max(1, len(words))
    p_digit_ratio = sum(c.isdigit() for c in chars) / max(1, len(chars))
    p_upper_ratio = sum(c.isupper() for c in chars) / max(1, len(chars))
    p_newline_count = prompt.count('\n')
    p_question_marks = prompt.count('?')
    p_has_code_block = int('```' in prompt or 'def ' in prompt_lower or 'class ' in prompt_lower)
    p_approx_tokens = max(1, int(len(prompt) / 4))
    p_entropy = _char_entropy(prompt)
    p_repeat_kw = sum(1 for kw in ["repeat", "1000", "500", "every", "all", "exhaustive", "complete", "list all", "step by step", "keep going"] if kw in prompt_lower)
    p_long_kw = sum(1 for kw in ["detail", "explain", "describe", "comprehensive", "elaborate", "in depth", "thoroughly"] if kw in prompt_lower)
    p_context_len = int("context" in prompt_lower or "retrieved" in prompt_lower) * p_char_count
    p_log_char = math.log1p(p_char_count)
    p_colon_count = prompt.count(':')
    p_comma_count = prompt.count(',')
    p_quote_count = prompt.count('"') + prompt.count("'")
    p_number_count = sum(c.isdigit() for c in prompt)
    task_vec = _task_onehot(prompt)
    u_remaining_user_budget = float(sample.get("remaining_user_budget", 50.0))
    u_remaining_ws_budget = float(sample.get("remaining_workspace_budget", 200.0))
    u_budget_ratio = u_remaining_user_budget / max(0.01, u_remaining_ws_budget)
    s_session_depth = float(sample.get("session_depth", 1))
    s_depth_log = math.log1p(s_session_depth)
    s_has_history = int(s_session_depth > 1)
    feature_vec = np.array([
        p_char_count, p_word_count, p_sentence_count,
        p_avg_word_len, p_unique_ratio, p_digit_ratio,
        p_upper_ratio, p_newline_count, p_question_marks,
        p_has_code_block, p_approx_tokens, p_entropy,
        p_repeat_kw, p_long_kw, p_context_len, p_log_char,
        p_colon_count, p_comma_count, p_quote_count, p_number_count,
        *task_vec,
        u_remaining_user_budget, u_remaining_ws_budget, u_budget_ratio,
        s_session_depth, s_depth_log, s_has_history,
    ], dtype=np.float32)
    return feature_vec
FEATURE_NAMES = [
    "p_char_count", "p_word_count", "p_sentence_count",
    "p_avg_word_len", "p_unique_ratio", "p_digit_ratio",
    "p_upper_ratio", "p_newline_count", "p_question_marks",
    "p_has_code_block", "p_approx_tokens", "p_entropy",
    "p_repeat_kw", "p_long_kw", "p_context_len", "p_log_char",
    "p_colon_count", "p_comma_count", "p_quote_count", "p_number_count",
    "t_code", "t_summary", "t_rag", "t_attack", "t_general",
    "u_remaining_user_budget", "u_remaining_ws_budget", "u_budget_ratio",
    "s_session_depth", "s_depth_log", "s_has_history",
]
def _char_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    n = len(text)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())
def _task_onehot(prompt: str) -> List[float]:
    p = prompt.lower()
    scores = {k: sum(1 for kw in v if kw in p) for k, v in TASK_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    order = ["code", "summary", "rag", "attack", "general"]
    return [1.0 if best == t else 0.0 for t in order]
def infer_model_name(samples: List[dict], default: str = "offline") -> str:
    models = [s.get("model") for s in samples if s.get("model")]
    if not models:
        return default
    uniq = sorted(set(models))
    if len(uniq) == 1:
        return uniq[0]
    print(f"[!] Multiple models found in dataset: {uniq}")
    print(f"    Using first model pricing for cost reconstruction: {uniq[0]}")
    return uniq[0]
def get_group_id(sample: dict) -> str:
    family = str(sample.get("prompt_family") or "")
    source_type = str(sample.get("source_type") or "")
    category = str(sample.get("category") or "")
    prompt = str(sample.get("prompt") or "")
    if family:
        return f"{category}::{source_type}::{family}"
    digest = abs(hash(prompt[:200])) % 10_000_000
    return f"{category}::legacy::{digest}"
def prepare_xy(samples: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([extract_features(s) for s in samples])
    y = np.array([
        [s["true_input_tokens"], s["true_output_tokens"], s["true_cost_usd"]]
        for s in samples
    ], dtype=np.float64)
    return X, y
def summarize_split(train_samples: List[dict], test_samples: List[dict]) -> None:
    def _count_by(samples, key):
        out = {}
        for s in samples:
            val = s.get(key, "<missing>")
            out[val] = out.get(val, 0) + 1
        return out
    print("[*] Split summary")
    print(f"    Train groups: {len({get_group_id(s) for s in train_samples})}")
    print(f"    Test groups : {len({get_group_id(s) for s in test_samples})}")
    print(f"    Train categories: {_count_by(train_samples, 'category')}")
    print(f"    Test categories : {_count_by(test_samples, 'category')}")
def group_train_test_split(samples: List[dict], test_ratio: float = 0.2, seed: int = RANDOM_SEED) -> Tuple[List[dict], List[dict]]:
    group_to_samples: Dict[str, List[dict]] = {}
    for s in samples:
        gid = get_group_id(s)
        group_to_samples.setdefault(gid, []).append(s)
    groups = list(group_to_samples.keys())
    rng = random.Random(seed)
    rng.shuffle(groups)
    target_test_n = max(1, int(round(len(samples) * test_ratio)))
    test_groups = []
    test_count = 0
    for gid in groups:
        if test_count >= target_test_n:
            break
        test_groups.append(gid)
        test_count += len(group_to_samples[gid])
    test_group_set = set(test_groups)
    train_samples, test_samples = [], []
    for gid, ss in group_to_samples.items():
        if gid in test_group_set:
            test_samples.extend(ss)
        else:
            train_samples.extend(ss)
    if not train_samples or not test_samples:
        raise ValueError("Group split failed: empty train or test set. Increase dataset size or reduce test_ratio.")
    return train_samples, test_samples
class CostPredictor:
    def __init__(self):
        self.models: Dict[str, object] = {}
        self.use_lgbm = False
        self.model_name = "offline"
        self._check_lgbm()
    def _check_lgbm(self):
        try:
            import lightgbm
            self.use_lgbm = True
            print("[*] LightGBM available — using LGBM regressor")
        except ImportError:
            print("[!] LightGBM not found — using Ridge fallback")
            print("    Install: pip install lightgbm --break-system-packages")
    def fit(self, X: np.ndarray, y: np.ndarray, model_name: str = "offline"):
        self.model_name = model_name
        targets = ["input_tokens", "output_tokens"]
        for i, name in enumerate(targets):
            yi = y[:, i]
            if self.use_lgbm:
                import lightgbm as lgb
                import pandas as pd
                X_df = pd.DataFrame(X, columns=FEATURE_NAMES)
                model = lgb.LGBMRegressor(
                    n_estimators=400,
                    learning_rate=0.04,
                    num_leaves=63,
                    min_child_samples=8,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.05,
                    reg_lambda=0.1,
                    random_state=RANDOM_SEED,
                    verbose=-1,
                )
                model.fit(X_df, np.log1p(yi))
            else:
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import Ridge
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=1.0)),
                ])
                model.fit(X, np.log1p(yi))
            self.models[name] = model
            print(f"  [✓] Trained predictor: {name}")
    def predict(self, X: np.ndarray) -> np.ndarray:
        ip, op = PRICING.get(self.model_name, PRICING["offline"])
        preds = []
        for name in ["input_tokens", "output_tokens"]:
            model = self.models[name]
            if self.use_lgbm:
                import pandas as pd
                X_df = pd.DataFrame(X, columns=FEATURE_NAMES)
                log_pred = model.predict(X_df)
            else:
                log_pred = model.predict(X)
            pred = np.expm1(log_pred)
            pred = np.maximum(pred, 0)
            preds.append(pred)
        pred_in = preds[0]
        pred_out = preds[1]
        pred_cost = pred_in * ip + pred_out * op
        return np.stack([pred_in, pred_out, pred_cost], axis=1)
    def save(self, path: str = "models/cost_predictor.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[✓] Model saved: {path}")
    @staticmethod
    def load(path: str = "models/cost_predictor.pkl") -> "CostPredictor":
        with open(path, "rb") as f:
            return pickle.load(f)
    def get_feature_importance(self, target: str = "output_tokens") -> Dict[str, float]:
        if not self.use_lgbm:
            raise RuntimeError("Feature importance is only available with LightGBM.")
        model = self.models.get(target)
        if model is None:
            raise RuntimeError(f"Model for '{target}' not found. Call fit() first.")
        importances = model.booster_.feature_importance(importance_type="gain")
        return dict(zip(FEATURE_NAMES, importances.tolist()))
    def plot_feature_importance(
        self,
        target: str = "output_tokens",
        top_n: int = 15,
        out_path: str = "results/figures/fig_feature_importance.png",
    ) -> Dict[str, float]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        importance = self.get_feature_importance(target)
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        labels = [item[0] for item in reversed(sorted_items)]
        values = [item[1] for item in reversed(sorted_items)]
        group_colors = {
            "p_": "#1f77b4",
            "t_": "#ff7f0e",
            "u_": "#2ca02c",
            "s_": "#d62728",
        }
        def get_color(name: str) -> str:
            for prefix, color in group_colors.items():
                if name.startswith(prefix):
                    return color
            return "#7f7f7f"
        colors = [get_color(l) for l in labels]
        fig, ax = plt.subplots(figsize=(8.5, 6.0))
        bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.65)
        ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
        ax.set_title(
            f"Feature Importance — {target.replace('_', ' ').title()} Predictor\n(Top {top_n})",
            fontsize=12,
            fontweight="bold",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#1f77b4", label="Prompt (P)"),
            Patch(facecolor="#ff7f0e", label="Task (T)"),
            Patch(facecolor="#2ca02c", label="User (U)"),
            Patch(facecolor="#d62728", label="Session (S)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.7)
        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=160)
        plt.close()
        print(f"[✓] Feature importance figure saved: {out_path}")
        return importance
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    labels = ["input_tokens", "output_tokens", "cost_usd"]
    results = {}
    for i, name in enumerate(labels):
        results[name] = {
            "MAE": round(mae(y_true[:, i], y_pred[:, i]), 4),
            "MAPE": round(mape(y_true[:, i], y_pred[:, i]), 2),
        }
    return results
def evaluate_by_category(samples: List[dict], y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    categories = sorted({s.get("category", "unknown") for s in samples})
    out = {}
    for cat in categories:
        idx = [i for i, s in enumerate(samples) if s.get("category") == cat]
        if not idx:
            continue
        cat_true = y_true[idx]
        cat_pred = y_pred[idx]
        out[cat] = {
            "output_MAE": round(mae(cat_true[:, 1], cat_pred[:, 1]), 4),
            "output_MAPE": round(mape(cat_true[:, 1], cat_pred[:, 1]), 2),
            "cost_MAE": round(mae(cat_true[:, 2], cat_pred[:, 2]), 6),
            "cost_MAPE": round(mape(cat_true[:, 2], cat_pred[:, 2]), 2),
            "n": len(idx),
        }
    return out
def print_eval(results: Dict):
    print("\n" + "=" * 55)
    print("  Cost Prediction Evaluation (Section 4.2)")
    print("=" * 55)
    print(f"  {'Target':<20} {'MAE':>10} {'MAPE (%)':>12}")
    print("  " + "-" * 43)
    for name, metrics in results.items():
        print(f"  {name:<20} {metrics['MAE']:>10.4f} {metrics['MAPE']:>10.2f}%")
    print("=" * 55)
def print_category_eval(results: Dict[str, Dict[str, float]]):
    print("\n  [Category-wise test performance]")
    print(f"  {'Category':<18} {'N':>5} {'Out_MAPE':>10} {'Cost_MAPE':>11}")
    print("  " + "-" * 50)
    for cat, metrics in results.items():
        print(f"  {cat:<18} {metrics['n']:>5} {metrics['output_MAPE']:>9.2f}% {metrics['cost_MAPE']:>10.2f}%")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset.json")
    parser.add_argument("--model_out", default="models/cost_predictor.pkl")
    parser.add_argument("--pred_out", default="data/dataset_with_predictions.json")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    args = parser.parse_args()
    print(f"[*] Loading dataset: {args.data}")
    with open(args.data, "r", encoding="utf-8") as f:
        all_samples = json.load(f)
    if len(all_samples) < 20:
        raise ValueError("Dataset too small for reliable split. Collect more API samples first.")
    train_samples, test_samples = group_train_test_split(all_samples, test_ratio=args.test_ratio)
    print(f"[*] Train: {len(train_samples)}, Test: {len(test_samples)}")
    summarize_split(train_samples, test_samples)
    X_train, y_train = prepare_xy(train_samples)
    X_test, y_test = prepare_xy(test_samples)
    dataset_model = infer_model_name(all_samples)
    print(f"[*] Pricing model for cost reconstruction: {dataset_model}")
    predictor = CostPredictor()
    predictor.fit(X_train, y_train, model_name=dataset_model)
    y_pred = predictor.predict(X_test)
    results = evaluate_predictions(y_test, y_pred)
    print_eval(results)
    category_results = evaluate_by_category(test_samples, y_test, y_pred)
    print_category_eval(category_results)
    predictor.save(args.model_out)
    test_pred_output = []
    for i, s in enumerate(test_samples):
        row = dict(s)
        row["pred_input_tokens"] = round(float(y_pred[i, 0]), 2)
        row["pred_output_tokens"] = round(float(y_pred[i, 1]), 2)
        row["pred_cost_usd"] = round(float(y_pred[i, 2]), 8)
        row["split_group"] = get_group_id(s)
        test_pred_output.append(row)
    Path("data").mkdir(parents=True, exist_ok=True)
    with open("data/test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(test_pred_output, f, indent=2, ensure_ascii=False)
    print(f"[✓] Test predictions saved: data/test_predictions.json ({len(test_pred_output)} samples)")
    print("\n  [Sample Predictions (first 5)]")
    print(f"  {'ID':<12} {'Family':<18} {'True_out':>10} {'Pred_out':>10} {'True_cost':>12} {'Pred_cost':>12}")
    for i in range(min(5, len(test_samples))):
        s = test_samples[i]
        fam = (s.get("prompt_family") or "-")[:18]
        print(
            f"  {s['sample_id']:<12} {fam:<18} "
            f"{int(y_test[i,1]):>10} {int(y_pred[i,1]):>10} ${y_test[i,2]:.6f} ${y_pred[i,2]:.6f}"
        )
    pred_output = []
    all_X, _ = prepare_xy(all_samples)
    all_pred = predictor.predict(all_X)
    for i, s in enumerate(all_samples):
        row = dict(s)
        row["pred_input_tokens"] = round(float(all_pred[i, 0]), 2)
        row["pred_output_tokens"] = round(float(all_pred[i, 1]), 2)
        row["pred_cost_usd"] = round(float(all_pred[i, 2]), 8)
        row["split_group"] = get_group_id(s)
        pred_output.append(row)
    Path(args.pred_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.pred_out, "w", encoding="utf-8") as f:
        json.dump(pred_output, f, indent=2, ensure_ascii=False)
    print(f"\n[✓] Predictions saved: {args.pred_out}")
    if predictor.use_lgbm:
        print("\n[*] Generating feature importance figures...")
        for target in ["input_tokens", "output_tokens"]:
            out_path = f"results/figures/fig_feature_importance_{target}.png"
            importance = predictor.plot_feature_importance(
                target=target, top_n=15, out_path=out_path
            )
            top5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top-5 ({target}): " + ", ".join(f"{k}={v:.1f}" for k, v in top5))
    else:
        print("[!] Feature importance skipped (LightGBM not available)")
