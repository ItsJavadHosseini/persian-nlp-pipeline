# Persian NLP Pipeline --- Final Project Report

**Repository:**
[ItsJavadHosseini/persian-nlp-pipeline](https://github.com/ItsJavadHosseini/persian-nlp-pipeline)\
**Branch:** `develop`\
**Project type:** End-to-end Persian NLP / Text Classification\
**Status:** Completed

------------------------------------------------------------------------

## 1. Executive Summary

The Persian NLP Pipeline is an end-to-end machine learning project for
analyzing Persian customer reviews from Digikala. It uses a fine-tuned
ParsBERT model to perform two text-classification tasks:

1.  **Recommendation Prediction** --- predicts whether a review is
    recommended, not recommended, or has no explicit recommendation.
2.  **Rating Band Prediction** --- predicts a categorical rating band
    (`low`, `medium`, or `high`) from review text.

The project covers the workflow from raw data preparation and model
fine-tuning to inference services, batch evaluation, a user interface,
and containerized deployment. The application is built with PyTorch,
Hugging Face Transformers, FastAPI, Gradio, and Docker Compose.

The verified quantitative model results documented in this report are
for Task A. Task B's model artifact is present in the project, but
verified evaluation metrics were not available for this report;
therefore, no Task B performance figures are claimed.

## 2. Project Objectives

The main objectives were to:

-   Build a reproducible preprocessing and data-splitting pipeline for a
    large Persian review dataset.
-   Fine-tune a Persian pretrained language model for review
    classification.
-   Reduce product-level data leakage by splitting data according to
    product IDs.
-   Provide an inference API and a simple interactive interface.
-   Support batch evaluation using CSV files.
-   Package the application for reproducible deployment with Docker
    Compose.

## 3. Dataset

### 3.1 Source

The project uses the **Digikala Comments & Products** dataset published
by RadeAI.

-   [Hugging Face
    Dataset](https://huggingface.co/datasets/RadeAI/Digikala_comments_products)
-   [Kaggle
    Dataset](https://www.kaggle.com/datasets/radeai/digikala-comments-and-products)

The primary input for the classification models is the review text in
the `body` field. The main target fields are `recommendation_status` and
`rate`.

### 3.2 Dataset Scale

  Item                                                                   Count
  -------------------------------------------- -------------------------------
  Raw review records                                                 6,156,289
  Records after cleaning                                             6,149,834
  Unique products used for grouped splitting                           331,512
  Cleaned review dataset size                    178.5 MB (Parquet, Zstandard)

The source dataset does not provide an official train/validation/test
split for this project. A product-grouped split was therefore created as
part of the pipeline.

## 4. Data Preparation

The preprocessing workflow is implemented in
`notebooks/02_preprocessing_and_split.ipynb`.

### 4.1 Persian Text Normalization

The pipeline uses Hazm's normalizer and additional deterministic
transformations:

-   Standardizes Arabic `ي` and `ك` to Persian `ی` and `ک`.
-   Converts Persian and Arabic digits to ASCII digits.
-   Removes URLs and HTML content.
-   Normalizes horizontal whitespace while preserving newline
    characters.
-   Preserves ZWNJ characters, emojis, and negation expressions.

An idempotence check was performed on the normalization function. No
violations were observed in the nine test cases used.

### 4.2 Cleaning and Feature Preparation

The cleaning stage includes:

-   Converting sentinel string values such as `"nan"`, `"null"`, and
    `"None"` to actual null values in relevant fields.
-   Removing duplicate review IDs.
-   Removing unused metadata columns.
-   Retaining the review text, identifiers, recommendation target,
    rating, and derived rating-band fields.

After duplicate removal, 6,149,834 review records remained.

### 4.3 Rating Band Construction

The numeric `rate` field is mapped into three categorical classes:

  Rating interval        Class
  ---------------------- ----------
  `0 <= rate < 2.5`      `low`
  `2.5 <= rate < 4.0`    `medium`
  `4.0 <= rate <= 5.0`   `high`

Out-of-range or missing ratings are marked as invalid. One record was
identified as having an invalid rating value.

## 5. Data Splitting and Leakage Prevention

### 5.1 Product-Grouped Split

To reduce leakage between training and evaluation data, the split is
performed at the **product level**, rather than randomly splitting
individual reviews.

Products are stratified using the majority recommendation class
associated with each product. The resulting product allocation is
approximately:

-   70% training
-   15% validation
-   15% test

The resulting review-level datasets are:

  Split                   Review records
  --------------------- ----------------
  Training supervised          3,578,692
  Validation                     913,570
  Test                           919,841

The product-level disjointness check passed: no product IDs overlap
across the train, validation, and test splits.

### 5.2 Semi-Supervised Training Data

A deterministic masking procedure was applied to the supervised training
partition:

-   Masking rate: 20%
-   Seed: `digikala-semi-supervised-v1`
-   Selection: stable SHA-256 hash derived from the row ID, applied
    within each class

This produced:

  Training artifact                  Records
  ------------------------------ -----------
  `train_supervised.csv`           3,578,692
  `train_pseudo_unlabeled.csv`       737,731

The masked records are retained as text-only examples in
`train_pseudo_unlabeled.csv`. The two files form a complete partition of
the original training partition.

The project also creates `natural_unlabeled.csv` with 627,763 records
whose recommendation label is naturally missing. This file is a subset
of `train_supervised.csv`, not an additional independent split.

### 5.3 Main Data Artifacts

The preprocessing notebook exports the following artifacts:

-   `train_supervised.csv`
-   `train_pseudo_unlabeled.csv`
-   `natural_unlabeled.csv`
-   `validation.csv`
-   `test.csv`
-   `split_manifest.json`

The manifest records important configuration details, including split
strategy, seeds, masking rate, class counts, and rating-band thresholds.

## 6. Tokenization

The tokenizer is loaded from the ParsBERT base model:

[HooshvareLab/bert-fa-base-uncased](https://huggingface.co/HooshvareLab/bert-fa-base-uncased)

Key configuration:

  Setting                   Value
  ------------------------- ---------------------------
  Tokenizer                 ParsBERT v2.0 tokenizer
  Vocabulary size           100,000
  Maximum sequence length   128
  Padding strategy          Dynamic padding
  Collator                  `DataCollatorWithPadding`

The maximum sequence length was selected after inspecting the
token-length distribution. In a 5,000-example sample, the 99th
percentile was approximately 64 tokens, and only 0.08% of examples
reached the configured maximum length of 128.

Tokenization is performed through a `ReviewDataset` implementation.
Dynamic padding is applied at batch time to avoid padding every example
to the maximum length in advance. Missing labels are represented with
the ignore index `-100` where applicable.

## 7. Model Architecture and Fine-Tuning

### 7.1 Base Model

Both tasks use ParsBERT v2.0:

-   Model: `HooshvareLab/bert-fa-base-uncased`
-   Architecture: BERT-base
-   Transformer layers: 12
-   Hidden size: 768
-   Attention heads: 12
-   Total parameters in the classification model: approximately 162.8
    million

The classification models are created using Hugging Face's
`AutoModelForSequenceClassification`.

### 7.2 Task A --- Recommendation Prediction

Task A is a three-class classification problem:

-   `recommended`
-   `not_recommended`
-   `no_idea`

The training setup includes:

-   Cross-entropy loss with inverse-frequency class weights.
-   AdamW optimizer with separate learning rates for the encoder and
    classification head.
-   Encoder learning rate: `2e-5`.
-   Classification-head learning rate: `1e-4`.
-   Linear learning-rate scheduler with 10% warmup.
-   Batch size: 16, with gradient accumulation of 2 (effective batch
    size: 32).
-   Maximum sequence length: 128.
-   Mixed-precision training (`fp16`).
-   Weight decay in the final experiment: `0.05`.
-   Model selection based on validation Macro F1.

Class weights used in the weighted-loss experiment:

  Class                 Weight
  ------------------- --------
  `recommended`           0.42
  `not_recommended`       3.49
  `no_idea`               2.96

### 7.3 Task A Experiments

Several training configurations were tested:

  Experiment   Configuration             Best reported validation Macro F1
  ------------ ----------------------- -----------------------------------
  v1           No class weights                                      0.677
  v3           Class-weighted loss                                   0.693
  v5           Revised configuration                                 0.702

**Important checkpoint distinction:** The best observed v5 validation
Macro F1 was 0.702 at step 10,000. That checkpoint was not retained
because of the checkpoint retention limit. The final saved model
corresponds to checkpoint 4,000, whose reported final evaluation is
presented below. The best observed validation score must not be confused
with the performance of the saved model.

## 8. Task A Evaluation

The final saved Task A model was evaluated with the following results:

  Metric                 Score
  ------------------- --------
  Accuracy              0.8361
  Balanced Accuracy     0.7141
  Macro F1              0.6941

### 8.1 Per-Class Results

  Class                 Precision   Recall   F1-score
  ------------------- ----------- -------- ----------
  `recommended`             0.951    0.894      0.922
  `not_recommended`         0.721    0.678      0.699
  `no_idea`                 0.388    0.570      0.462

### 8.2 Interpretation

The model performs strongest on the `recommended` class. Its precision
and recall are both high, indicating that it identifies this majority
class reliably.

Performance is lower on `not_recommended` and especially on `no_idea`.
The `no_idea` class has the lowest precision, which indicates
substantial confusion with other classes. This is an important
limitation when using the model for automated review analysis.

Accuracy alone does not fully describe performance on this imbalanced
dataset. Balanced Accuracy and Macro F1 provide additional information
about performance across classes.

## 9. Task B --- Rating Band Prediction

Task B predicts one of three rating bands from the review text:

-   `low`
-   `medium`
-   `high`

The fine-tuned model artifact is stored at:

`artifacts/model_band_best/`

The available project information confirms the task and the model
artifact location. However, verified validation/test metrics and
per-class results were not available when this report was prepared.
Therefore, this report does not make quantitative performance claims for
Task B.

## 10. Inference API

The inference service is implemented with FastAPI and Uvicorn.

Main application file:

`src/persian_nlp_pipeline/api.py`

The documented endpoints are:

  ---------------------------------------------------------------------------------
  Endpoint                          Method                  Purpose
  --------------------------------- ----------------------- -----------------------
  `/health`                         GET                     Health check

  `/api/v1/predict`                 POST                    Run prediction

  `/api/v1/evaluate`                POST                    Submit a CSV for batch
                                                            evaluation

  `/api/v1/evaluation/{filename}`   GET                     Retrieve an evaluation
                                                            output
  ---------------------------------------------------------------------------------

The prediction logic is separated into dedicated modules, including
`predictor.py` and `evaluator.py`.

## 11. Batch Evaluation

The batch evaluation component accepts CSV input and performs batch
inference. Its workflow includes:

1.  Reading the input CSV.
2.  Running model inference over the review records.
3.  Computing evaluation metrics when ground-truth labels are available.
4.  Producing a confusion matrix and prediction outputs.
5.  Saving evaluation artifacts under `artifacts/evaluation/`.

This makes it possible to evaluate the model on external datasets
without manually running notebook cells.

## 12. Gradio User Interface

The frontend is implemented in:

`frontend/app.py`

Gradio provides an interactive interface for submitting review text and
viewing predictions. The frontend communicates with the FastAPI backend
through the `API_URL` environment variable.

The default API URL is configured for local development, while Docker
Compose supplies the service URL for containerized execution.

## 13. Docker Deployment

The project includes:

-   `Dockerfile`
-   `docker-compose.yml`
-   `.dockerignore`
-   `pyproject.toml`
-   `uv.lock`

Docker Compose runs two services:

  Service    Purpose                       Port
  ---------- --------------------------- ------
  API        FastAPI inference backend     8000
  Frontend   Gradio interface              7860

Both services use the project image but run separate application
processes. Health checks are configured, and a bind mount is used to
persist evaluation artifacts.

### 13.1 Quick Start

Clone the repository and start the services:

``` bash
git clone https://github.com/ItsJavadHosseini/persian-nlp-pipeline.git
cd persian-nlp-pipeline
docker compose up -d
```

Open the following local addresses:

  Service             Address
  ------------------- ------------------------------
  Gradio interface    http://localhost:7860
  API                 http://localhost:8000
  API documentation   http://localhost:8000/docs
  Health check        http://localhost:8000/health

To stop the services:

``` bash
docker compose down
```

## 14. Project Structure

The main project components include:

``` text
persian-nlp-pipeline/
├── artifacts/
│   ├── bert_recommendation/
│   │   └── model_taskA_bert/
│   ├── model_band_best/
│   └── evaluation/
├── data/
│   └── processed/
├── frontend/
│   └── app.py
├── notebooks/
│   ├── 02_preprocessing_and_split.ipynb
│   └── 04_transfer_learning_finetuning.ipynb
├── src/
│   └── persian_nlp_pipeline/
│       ├── api.py
│       ├── evaluator.py
│       └── predictor.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

This is a high-level representation of the documented project
components, not a guarantee that every generated data file or model
artifact is tracked in Git.

## 15. Technology Stack

  Area                         Technologies
  ---------------------------- ---------------------------
  Programming language         Python
  Deep learning                PyTorch
  NLP and pretrained models    Hugging Face Transformers
  Persian text processing      Hazm
  Data processing              Pandas, NumPy
  Machine learning utilities   Scikit-learn
  API                          FastAPI, Uvicorn
  User interface               Gradio
  Dependency management        uv
  Containerization             Docker, Docker Compose

## 16. Limitations and Future Improvements

The following are practical areas for further work:

1.  **Improve minority-class performance:** Investigate class-specific
    errors, alternative loss functions, resampling, and
    decision-threshold calibration, especially for `no_idea`.
2.  **Preserve the best checkpoint:** Configure checkpoint retention or
    explicitly save the best validation checkpoint so the best observed
    model is not lost.
3.  **Complete Task B evaluation:** Record validation and test metrics,
    per-class precision/recall/F1, and a confusion matrix.
4.  **Evaluate on independent data:** Test robustness on reviews from a
    different time period or source.
5.  **Add automated tests:** Cover text normalization, data validation,
    prediction schemas, API behavior, and batch evaluation.
6.  **Add experiment tracking:** Record model configurations,
    checkpoints, metrics, and dataset versions in a consistent
    experiment-tracking system.
7.  **Document model artifact distribution:** The fine-tuned models are
    currently stored locally in the project artifact directories and do
    not have public Hugging Face model pages.

## 17. Model and Dataset References

### Dataset

-   [RadeAI --- Digikala Comments & Products (Hugging
    Face)](https://huggingface.co/datasets/RadeAI/Digikala_comments_products)
-   [RadeAI --- Digikala Comments & Products
    (Kaggle)](https://www.kaggle.com/datasets/radeai/digikala-comments-and-products)

### Base Model

-   [ParsBERT v2.0 ---
    HooshvareLab](https://huggingface.co/HooshvareLab/bert-fa-base-uncased)

### Fine-Tuned Model Artifacts

-   Task A: `artifacts/bert_recommendation/model_taskA_bert/`
-   Task B: `artifacts/model_band_best/`

The fine-tuned models are currently stored locally and are not publicly
available through Hugging Face.

------------------------------------------------------------------------

## 18. Conclusion

The Persian NLP Pipeline demonstrates a complete applied NLP workflow:
large-scale Persian review preprocessing, product-aware dataset
splitting, ParsBERT fine-tuning, evaluation, API development, an
interactive frontend, and Docker-based deployment.

For Task A, the final saved model achieved an Accuracy of **0.8361**,
Balanced Accuracy of **0.7141**, and Macro F1 of **0.6941**. The
class-level results also reveal a clear opportunity to improve
minority-class prediction, particularly for `no_idea`.

The project provides a deployable foundation for Persian customer-review
classification. Further work should focus on improving minority-class
performance, retaining the best training checkpoint, and completing a
documented evaluation of Task B.
