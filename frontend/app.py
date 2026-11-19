
import os
import tempfile

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


# --------------------------------------------------
# API Functions
# --------------------------------------------------

def predict_review(text):
    if not text or not text.strip():
        return {"error": "Please enter a review."}

    try:
        response = requests.post(
            f"{API_URL}/api/v1/predict",
            json={"text": text},
            timeout=120,
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException as exc:
        return {"error": str(exc)}


def plot_confusion_matrix(metrics, task):
    matrix_data = metrics[task]["confusion_matrix"]

    labels = matrix_data["labels"]
    matrix = matrix_data["matrix"]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        ax=ax,
    )

    ax.set_title(
        task.replace("_", " ").title(),
        fontsize=13,
        pad=15,
    )

    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("Actual Label", fontsize=10)

    plt.xticks(rotation=15, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    return fig


def build_metrics_table(metrics):
    rows = []

    for task, values in metrics.items():
        rows.append({
            "Task": task.replace("_", " ").title(),
            "Samples": values["samples"],
            "Accuracy": values["accuracy"],
            "Macro Precision": values["macro_precision"],
            "Macro Recall": values["macro_recall"],
            "Macro F1": values["macro_f1"],
            "Weighted F1": values["weighted_f1"],
        })

    return pd.DataFrame(rows)


def evaluate_csv(file_path, task, batch_size, chunksize):
    if file_path is None:
        return (
            pd.DataFrame(),
            None,
            None,
            None,
            {"error": "Please upload a CSV file."},
        )

    try:
        with open(file_path, "rb") as file:
            response = requests.post(
                f"{API_URL}/api/v1/evaluate",
                files={
                    "file": (
                        os.path.basename(file_path),
                        file,
                        "text/csv",
                    )
                },
                data={
                    "task": task,
                    "batch_size": int(batch_size),
                    "chunksize": int(chunksize),
                },
                timeout=None,
            )

        response.raise_for_status()
        result = response.json()

        metrics = result["metrics"]

        download_response = requests.get(
            f"{API_URL}{result['download_url']}",
            timeout=120,
        )

        download_response.raise_for_status()

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix="_predictions.csv",
            delete=False,
        ) as output_file:

            output_file.write(download_response.content)
            output_path = output_file.name

        metrics_table = build_metrics_table(metrics)

        recommendation_fig = None
        rating_fig = None

        if "recommendation" in metrics:
            recommendation_fig = plot_confusion_matrix(
                metrics,
                "recommendation",
            )

        if "rating_band" in metrics:
            rating_fig = plot_confusion_matrix(
                metrics,
                "rating_band",
            )

        return (
            metrics_table,
            recommendation_fig,
            rating_fig,
            output_path,
            metrics,
        )

    except requests.RequestException as exc:
        return (
            pd.DataFrame(),
            None,
            None,
            None,
            {"error": str(exc)},
        )

    except Exception as exc:
        return (
            pd.DataFrame(),
            None,
            None,
            None,
            {"error": str(exc)},
        )


# --------------------------------------------------
# Custom Styling
# --------------------------------------------------

CUSTOM_CSS = """
/* Main application container */
.gradio-container {
    max-width: 1450px !important;
    width: calc(100% - 48px) !important;
    margin: 0 auto !important;
    padding: 24px 0 40px !important;
}

/* Main header */
.main-header {
    padding: 10px 0 24px;
    border-bottom: 1px solid var(--border-color-primary);
    margin-bottom: 24px;
}

.main-header h1 {
    font-size: 28px !important;
    font-weight: 650 !important;
    margin-bottom: 6px !important;
}

.main-header p {
    color: var(--body-text-color-subdued);
    font-size: 14px;
    margin: 0;
}

/* Section headings */
.section-heading {
    font-size: 18px !important;
    font-weight: 600 !important;
    margin: 8px 0 14px !important;
}

/* Tabs */
.tabs {
    border: none !important;
}

.tab-nav {
    margin-bottom: 24px !important;
}

/* Content spacing */
.content-section {
    padding: 8px 0 18px;
}

/* Input fields */
.rtl-input textarea {
    direction: rtl !important;
    text-align: right !important;
    line-height: 1.9 !important;
}

/* Buttons */
.primary-button {
    min-height: 44px !important;
    font-weight: 600 !important;
}

/* Metrics table */
.metrics-table {
    margin-top: 8px;
}

/* Plot containers */
.plot-container {
    min-height: 350px;
}

/* Footer */
footer {
    display: none !important;
}

/* Laptop and desktop */
@media (min-width: 1100px) {
    .gradio-container {
        padding-left: 20px !important;
        padding-right: 20px !important;
    }
}

/* Smaller screens */
@media (max-width: 900px) {
    .gradio-container {
        width: calc(100% - 24px) !important;
        padding-top: 12px !important;
    }

    .main-header h1 {
        font-size: 23px !important;
    }
}
"""


# --------------------------------------------------
# Gradio Interface
# --------------------------------------------------

with gr.Blocks(
    title="Persian NLP Pipeline",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
) as demo:

    # Header
    gr.Markdown(
        """
        <div class="main-header">
            <h1>Persian NLP Pipeline</h1>
            <p>Persian Product Review Analysis</p>
        </div>
        """
    )

    with gr.Tabs():

        # ==========================================
        # Single Prediction
        # ==========================================

        with gr.Tab("Single Prediction"):

            gr.Markdown(
                "### Review Analysis",
                elem_classes=["section-heading"],
            )

            with gr.Row():

                with gr.Column(scale=7):

                    text_input = gr.Textbox(
                        label="Review Text",
                        placeholder="نظر خود را اینجا وارد کنید...",
                        lines=6,
                        max_lines=10,
                        elem_classes=["rtl-input"],
                    )

                    predict_button = gr.Button(
                        "Analyze Review",
                        variant="primary",
                        elem_classes=["primary-button"],
                    )

                with gr.Column(scale=5):

                    prediction_output = gr.JSON(
                        label="Prediction Results",
                        height=300,
                    )

            predict_button.click(
                fn=predict_review,
                inputs=text_input,
                outputs=prediction_output,
            )

        # ==========================================
        # CSV Evaluation
        # ==========================================

        with gr.Tab("CSV Evaluation"):

            gr.Markdown(
                "### Dataset Evaluation",
                elem_classes=["section-heading"],
            )

            # Input and configuration
            with gr.Row(equal_height=True):

                with gr.Column(scale=2):

                    csv_input = gr.File(
                        label="Upload CSV Dataset",
                        file_types=[".csv"],
                        type="filepath",
                    )

                with gr.Column(scale=2):

                    task_input = gr.Radio(
                        choices=[
                            ("Both Models", "both"),
                            ("Recommendation", "recommendation"),
                            ("Rating Band", "rating_band"),
                        ],
                        value="both",
                        label="Evaluation Task",
                    )

            # Evaluation parameters
            gr.Markdown("#### Processing Configuration")

            with gr.Row():

                batch_input = gr.Slider(
                    minimum=1,
                    maximum=64,
                    value=16,
                    step=1,
                    label="Batch Size",
                )

                chunksize_input = gr.Slider(
                    minimum=50,
                    maximum=2000,
                    value=100,
                    step=50,
                    label="Chunk Size",
                )

            evaluate_button = gr.Button(
                "Run Evaluation",
                variant="primary",
                elem_classes=["primary-button"],
            )

            # Metrics
            gr.Markdown(
                "### Evaluation Metrics",
                elem_classes=["section-heading"],
            )

            metrics_output = gr.Dataframe(
                label="Model Performance",
                interactive=False,
                wrap=True,
                elem_classes=["metrics-table"],
            )

            # Confusion matrices
            gr.Markdown(
                "### Confusion Matrices",
                elem_classes=["section-heading"],
            )

            with gr.Row():

                with gr.Column(scale=1):

                    recommendation_plot = gr.Plot(
                        label="Recommendation",
                        elem_classes=["plot-container"],
                    )

                with gr.Column(scale=1):

                    rating_plot = gr.Plot(
                        label="Rating Band",
                        elem_classes=["plot-container"],
                    )

            # Download
            gr.Markdown(
                "### Export Results",
                elem_classes=["section-heading"],
            )

            download_output = gr.File(
                label="Download Predictions CSV",
            )

            raw_metrics_output = gr.JSON(
                label="Detailed Classification Reports",
                visible=False,
            )

            evaluate_button.click(
                fn=evaluate_csv,
                inputs=[
                    csv_input,
                    task_input,
                    batch_input,
                    chunksize_input,
                ],
                outputs=[
                    metrics_output,
                    recommendation_plot,
                    rating_plot,
                    download_output,
                    raw_metrics_output,
                ],
            )


# --------------------------------------------------
# Launch
# --------------------------------------------------

if __name__ == "__main__":
    demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
)