from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
def pseudo(value: Any, prefix: str) -> str:
    return f"{prefix}_{hashlib.sha256(str(value).encode()).hexdigest()[:8]}"
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Private/raw dataset JSON")
    ap.add_argument("--output", default="data/dataset_with_predictions_public.json")
    ap.add_argument("--drop_prompt", action="store_true", help="Drop prompt text; core predictor reproduction will require regenerated features.")
    args = ap.parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    cleaned = []
    for row in data:
        out = dict(row)
        out.pop("raw_response_snippet", None)
        if args.drop_prompt:
            out.pop("prompt", None)
        if "user_id" in out:
            out["user_id"] = pseudo(out["user_id"], "user")
        if "workspace_id" in out:
            out["workspace_id"] = pseudo(out["workspace_id"], "workspace")
        cleaned.append(out)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    print(f"[✓] Public dataset saved: {args.output} ({len(cleaned)} records)")
if __name__ == "__main__":
    main()
