"""
Offline prediction script for Digikala review analysis.

Loads the fine-tuned BERT models for Task A (recommendation) and
Task B (rating_band) and predicts both from a single review body.

Usage:
    python predict.py --text "کیفیت محصول خیلی خوب بود"
    echo "کیفیت محصول خیلی خوب بود" | python predict.py
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Paths --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_A_DIR = PROJECT_ROOT / "artifacts" / "bert_recommendation" / "model_taskA_bert"
MODEL_B_DIR = PROJECT_ROOT / "artifacts" / "model_band_best"

# Fallbacks (different naming in different accounts)
if not MODEL_A_DIR.exists():
    MODEL_A_DIR = PROJECT_ROOT / "artifacts" / "bert_recommendation" / "model_v5_best"
if not MODEL_B_DIR.exists():
    MODEL_B_DIR = PROJECT_ROOT / "artifacts" / "bert_rating_band" / "model_band_best"

MAX_LENGTH = 128
MODEL_VERSION = "1.0.0"

# --- Global state (loaded once) ----------------------------------------
_TOK_A = None
_TOK_B = None
_MODEL_A = None
_MODEL_B = None


def load_models():
    """Load both models and tokenizers once (call at startup)."""
    global _TOK_A, _TOK_B, _MODEL_A, _MODEL_B
    if _MODEL_A is not None and _MODEL_B is not None:
        return

    if not MODEL_A_DIR.exists():
        raise FileNotFoundError(f"Task A model not found: {MODEL_A_DIR}")
    if not MODEL_B_DIR.exists():
        raise FileNotFoundError(f"Task B model not found: {MODEL_B_DIR}")

    print(f"Loading Task A model from: {MODEL_A_DIR}", file=sys.stderr)
    _TOK_A = AutoTokenizer.from_pretrained(str(MODEL_A_DIR / "tokenizer"))
    _MODEL_A = AutoModelForSequenceClassification.from_pretrained(str(MODEL_A_DIR))
    _MODEL_A.eval()

    print(f"Loading Task B model from: {MODEL_B_DIR}", file=sys.stderr)
    _TOK_B = AutoTokenizer.from_pretrained(str(MODEL_B_DIR / "tokenizer"))
    _MODEL_B = AutoModelForSequenceClassification.from_pretrained(str(MODEL_B_DIR))
    _MODEL_B.eval()

    print("Models loaded.", file=sys.stderr)


def predict(text: str) -> dict:
    """
    Predict recommendation status and rating band for a single review.

    Returns a dict with model version, latency, token count, and both
    predictions with probabilities.
    """
    if _MODEL_A is None or _MODEL_B is None:
        load_models()

    if not text or not text.strip():
        raise ValueError("Empty text")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _MODEL_A.to(device)
    _MODEL_B.to(device)

    t0 = time.time()

    inputs = _TOK_A(
        text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
    ).to(device)
    token_count = int(inputs["attention_mask"].sum().item())

    with torch.no_grad():
        logits_a = _MODEL_A(**inputs).logits
        probs_a = torch.softmax(logits_a, dim=-1)[0].cpu().tolist()

        logits_b = _MODEL_B(**inputs).logits
        probs_b = torch.softmax(logits_b, dim=-1)[0].cpu().tolist()

    latency_ms = (time.time() - t0) * 1000

    # Task A
    pred_a_idx = int(max(range(len(probs_a)), key=lambda i: probs_a[i]))
    label_a = _MODEL_A.config.id2label[pred_a_idx]
    probs_a_dict = {_MODEL_A.config.id2label[i]: round(p, 4) for i, p in enumerate(probs_a)}

    # Task B
    pred_b_idx = int(max(range(len(probs_b)), key=lambda i: probs_b[i]))
    label_b = _MODEL_B.config.id2label[pred_b_idx]
    probs_b_dict = {_MODEL_B.config.id2label[i]: round(p, 4) for i, p in enumerate(probs_b)}

    return {
        "model_version": MODEL_VERSION,
        "text": text,
        "token_count": token_count,
        "truncated": token_count >= MAX_LENGTH,
        "latency_ms": round(latency_ms, 2),
        "recommendation": {
            "label": label_a,
            "confidence": probs_a_dict[label_a],
            "probabilities": probs_a_dict,
        },
        "rating_band": {
            "label": label_b,
            "confidence": probs_b_dict[label_b],
            "probabilities": probs_b_dict,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict Digikala review recommendation and rating band."
    )
    parser.add_argument(
        "--text", "-t", type=str, default=None,
        help="Review text. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON (default: human-readable).",
    )
    args = parser.parse_args()

    if args.text:
        text = args.text
    else:
        text = sys.stdin.read().strip()

    if not text:
        print("Error: empty input.", file=sys.stderr)
        sys.exit(1)

    result = predict(text)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("=" * 60)
        print(f"Text: {result['text']}")
        print(f"Tokens: {result['token_count']}  "
              f"Truncated: {result['truncated']}  "
              f"Latency: {result['latency_ms']}ms")
        print("-" * 60)
        print(f"Recommendation: {result['recommendation']['label']}  "
              f"(conf={result['recommendation']['confidence']})")
        for k, v in result['recommendation']['probabilities'].items():
            print(f"  {k:<18} {v}")
        print("-" * 60)
        print(f"Rating band: {result['rating_band']['label']}  "
              f"(conf={result['rating_band']['confidence']})")
        for k, v in result['rating_band']['probabilities'].items():
            print(f"  {k:<18} {v}")
        print("=" * 60)


if __name__ == "__main__":
    main()