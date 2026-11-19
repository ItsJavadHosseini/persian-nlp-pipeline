from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


LABELS = {
    "recommendation": [
        "recommended",
        "not_recommended",
        "no_idea",
    ],
    "rating_band": [
        "low",
        "medium",
        "high",
    ],
}


class CSVEvaluator:
    def __init__(self, predictor):
        self.predictor = predictor

    def _calculate_metrics(self, y_true, y_pred, labels):
        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )

        return {
            "samples": len(y_true),
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_precision": round(
                precision_score(
                    y_true, y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                ), 4
            ),
            "macro_recall": round(
                recall_score(
                    y_true, y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                ), 4
            ),
            "macro_f1": round(
                f1_score(
                    y_true, y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                ), 4
            ),
            "weighted_f1": round(
                f1_score(
                    y_true, y_pred,
                    labels=labels,
                    average="weighted",
                    zero_division=0,
                ), 4
            ),
            "confusion_matrix": {
                "labels": labels,
                "matrix": matrix.tolist(),
            },
            "classification_report": report,
        }

    def evaluate_csv(
        self,
        input_path,
        output_path,
        task="both",
        batch_size=32,
        chunksize=1000,
    ):
        """
        Run predictions on a CSV and calculate evaluation metrics.

        Supported tasks:
            recommendation
            rating_band
            both
        """

        if task not in {"recommendation", "rating_band", "both"}:
            raise ValueError(f"Unsupported task: {task}")

        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(input_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        required_columns = {
            "body_normalized",
            "recommendation_status",
            "rating_band",
            "is_valid_rating_band",
        }

        y_true = {
            "recommendation": [],
            "rating_band": [],
        }

        y_pred = {
            "recommendation": [],
            "rating_band": [],
        }

        first_chunk = True

        reader = pd.read_csv(
            input_path,
            chunksize=chunksize,
        )

        for chunk_number, chunk in enumerate(reader, start=1):

            missing = required_columns - set(chunk.columns)

            if missing:
                raise ValueError(
                    f"Missing CSV columns: {sorted(missing)}"
                )

            texts = chunk["body_normalized"].fillna("").astype(str)
            valid_text_mask = texts.str.strip().ne("")

            chunk["recommendation_prediction"] = None
            chunk["recommendation_confidence"] = np.nan

            chunk["rating_band_prediction"] = None
            chunk["rating_band_confidence"] = np.nan

            valid_indices = chunk.index[valid_text_mask].tolist()

            for start in range(0, len(valid_indices), batch_size):
                indices = valid_indices[start:start + batch_size]

                batch_texts = texts.loc[indices].tolist()

                predictions = self.predictor.predict_batch(batch_texts)

                for index, prediction in zip(indices, predictions):
                    recommendation = prediction["recommendation"]
                    rating = prediction["rating_band"]

                    chunk.at[index, "recommendation_prediction"] = (
                        recommendation["label"]
                    )
                    chunk.at[index, "recommendation_confidence"] = (
                        recommendation["confidence"]
                    )

                    chunk.at[index, "rating_band_prediction"] = (
                        rating["label"]
                    )
                    chunk.at[index, "rating_band_confidence"] = (
                        rating["confidence"]
                    )

            if task in {"recommendation", "both"}:
                mask = (
                    valid_text_mask
                    & chunk["recommendation_status"].isin(
                        LABELS["recommendation"]
                    )
                )

                y_true["recommendation"].extend(
                    chunk.loc[mask, "recommendation_status"].tolist()
                )

                y_pred["recommendation"].extend(
                    chunk.loc[mask, "recommendation_prediction"].tolist()
                )

            if task in {"rating_band", "both"}:
                valid_rating = (
                    chunk["is_valid_rating_band"]
                    .astype(str)
                    .str.lower()
                    .isin(["true", "1"])
                )

                mask = (
                    valid_text_mask
                    & valid_rating
                    & chunk["rating_band"].isin(
                        LABELS["rating_band"]
                    )
                )

                y_true["rating_band"].extend(
                    chunk.loc[mask, "rating_band"].tolist()
                )

                y_pred["rating_band"].extend(
                    chunk.loc[mask, "rating_band_prediction"].tolist()
                )

            chunk.to_csv(
                output_path,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
            )

            first_chunk = False

            print(
                f"Processed chunk {chunk_number} "
                f"({len(chunk)} rows)"
            )

        results = {}

        if task in {"recommendation", "both"}:
            results["recommendation"] = self._calculate_metrics(
                y_true["recommendation"],
                y_pred["recommendation"],
                LABELS["recommendation"],
            )

        if task in {"rating_band", "both"}:
            results["rating_band"] = self._calculate_metrics(
                y_true["rating_band"],
                y_pred["rating_band"],
                LABELS["rating_band"],
            )

        return results