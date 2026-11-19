from pathlib import Path
import time

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_A_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "bert_recommendation"
    / "model_taskA_bert"
)

MODEL_B_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "model_band_best"
)

MAX_LENGTH = 128


class ReviewPredictor:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer_a = AutoTokenizer.from_pretrained(
            str(MODEL_A_DIR / "tokenizer")
        )

        self.tokenizer_b = AutoTokenizer.from_pretrained(
            str(MODEL_B_DIR / "tokenizer")
        )

        self.model_a = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_A_DIR)
        ).to(self.device)

        self.model_b = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_B_DIR)
        ).to(self.device)

        self.model_a.eval()
        self.model_b.eval()

        print(f"Models loaded on {self.device}")

    def _predict_model(self, texts, tokenizer, model):
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(self.device)

        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)

        return probabilities.cpu()

    def predict_batch(self, texts):
        if not texts:
            return []

        start_time = time.perf_counter()

        probs_a = self._predict_model(
            texts,
            self.tokenizer_a,
            self.model_a,
        )

        probs_b = self._predict_model(
            texts,
            self.tokenizer_b,
            self.model_b,
        )

        results = []

        for index, text in enumerate(texts):
            pred_a_idx = int(probs_a[index].argmax())
            pred_b_idx = int(probs_b[index].argmax())

            labels_a = self.model_a.config.id2label
            labels_b = self.model_b.config.id2label

            recommendation_probs = {
                labels_a[i]: round(float(p), 4)
                for i, p in enumerate(probs_a[index])
            }

            rating_probs = {
                labels_b[i]: round(float(p), 4)
                for i, p in enumerate(probs_b[index])
            }

            recommendation_label = labels_a[pred_a_idx]
            rating_label = labels_b[pred_b_idx]

            results.append({
                "text": text,
                "recommendation": {
                    "label": recommendation_label,
                    "confidence": recommendation_probs[recommendation_label],
                    "probabilities": recommendation_probs,
                },
                "rating_band": {
                    "label": rating_label,
                    "confidence": rating_probs[rating_label],
                    "probabilities": rating_probs,
                },
            })

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return results