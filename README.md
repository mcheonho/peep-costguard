# PEEP CostGuard: Lightweight Structured Prediction for Pre-execution Governance

This repository contains the reproducibility package for the paper:

**Lightweight Structured Prediction for Pre-execution Governance in Agentic AI Systems: A Comparison with LLM-based Estimators**

The code supports API-grounded dataset construction, structured token/cost prediction, policy-engine evaluation, model ablation with confidence intervals, GSS sensitivity analysis, and LLM-estimator baselines.

## Repository structure

```text
src/
  01_build_dataset.py        # API-only ground-truth collector with scalable prompt augmentation
  02_train_predictor.py      # Feature extraction, group-aware split, LightGBM training/prediction
  03_policy_engine.py        # Pre-execution policy engine, BVR/FSR/cost-reduction evaluation
  04_ablation_ci.py          # Ridge/RF/XGBoost/CatBoost/MLP/LightGBM comparison with 95% CI
  05_gss_sensitivity.py      # GSS threshold/weight sensitivity analysis
  06_llm_baseline.py         # GPT/Claude LLM-estimator baseline comparison
  07_run_all.py              # Pipeline runner and paper-number summary
  08_make_public_dataset.py  # Utility for removing response snippets and pseudonymizing IDs

data/
  dataset_with_predictions_public.json
```

## Public dataset note

The public JSON keeps the synthetic prompts needed to reproduce feature extraction and LLM-baseline experiments. API response snippets are removed, and user/workspace identifiers are pseudonymized. The released records include true API usage fields, predicted token/cost fields, and split-group metadata.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce core paper numbers from the public dataset

From the repository root:

```bash
python src/07_run_all.py \
  --data data/dataset_with_predictions_public.json \
  --skip_predict
```

This reuses the provided prediction dataset and runs the policy evaluation, ablation with confidence intervals, GSS sensitivity analysis, and summary extraction.

To retrain the structured predictor and regenerate predictions:

```bash
python src/02_train_predictor.py \
  --data data/dataset_with_predictions_public.json \
  --pred_out data/dataset_with_predictions_regenerated.json
```

Then run:

```bash
python src/07_run_all.py \
  --data data/dataset_with_predictions_regenerated.json \
  --skip_predict
```

## Build a new API-grounded dataset

OpenAI example:

```bash
export OPENAI_API_KEY=YOUR_KEY
python src/01_build_dataset.py \
  --backend openai \
  --model gpt-4o-mini \
  --n 4000 \
  --delay 0.1 \
  --checkpoint_every 200 \
  --output data/dataset_api_only_20k.json
```

Anthropic example:

```bash
export ANTHROPIC_API_KEY=YOUR_KEY
python src/01_build_dataset.py \
  --backend anthropic \
  --model claude-haiku-4-5-20251001 \
  --n 4000 \
  --output data/dataset_api_only_20k_anthropic.json
```

Local OpenAI-compatible endpoint example:

```bash
python src/01_build_dataset.py \
  --backend local \
  --base_url http://localhost:8000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --n 4000
```

## Run individual experiments

```bash
python src/03_policy_engine.py --data data/dataset_with_predictions_public.json
python src/04_ablation_ci.py --data data/dataset_with_predictions_public.json --n_seeds 5
python src/05_gss_sensitivity.py --ablation data/ablation_ci.json
```

The LLM-baseline script requires API keys and makes paid API calls:

```bash
export OPENAI_API_KEY=YOUR_KEY
export ANTHROPIC_API_KEY=YOUR_KEY
python src/06_llm_baseline.py \
  --data data/dataset_with_predictions_public.json \
  --backends openai:gpt-4o-mini anthropic:claude-haiku-4-5-20251001 \
  --shots zero strong_few \
  --n_eval 100
```

## Outputs

Typical outputs are written to:

```text
data/test_predictions.json
data/policy_results.json
data/ablation_ci.json
data/gss_sensitivity.json
results/paper_numbers.json
results/paper_numbers.txt
results/figures/
```

## Reproducibility details

- All core scripts use fixed random seeds by default.
- Dataset splitting is group-aware using `prompt_family`/`split_group` metadata.
- The public dataset is derived from API-grounded token/cost logs, not simulated token labels.
- API-based regeneration may produce slightly different latencies and outputs depending on backend availability, model version, and service-side changes.
