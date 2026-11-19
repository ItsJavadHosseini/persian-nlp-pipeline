# Persian NLP Pipeline

An end-to-end Persian NLP project for analyzing Digikala customer reviews using fine-tuned BERT models, FastAPI, Gradio, and Docker.

The application provides two prediction tasks:

- **Recommendation Prediction:** Predicts the recommendation label for a customer review.
- **Rating Band Prediction:** Predicts the rating category associated with a review.

The project includes model inference, CSV-based batch evaluation, a REST API, and a web interface.

---

## Quick Start

### Prerequisites

- Docker
- Docker Compose

### 1. Clone the Repository

```bash
git clone https://github.com/ItsJavadHosseini/persian-nlp-pipeline.git
cd persian-nlp-pipeline
```

### 2. Start the Application

```bash
docker compose up -d
```

Docker Compose starts the API and frontend services.

### 3. Access the Application

| Service | URL |
|---|---|
| Gradio Web Interface | http://localhost:7860 |
| FastAPI | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |

### 4. Stop the Application

```bash
docker compose down
```

Evaluation outputs are persisted in the local `artifacts/evaluation/` directory and remain available after containers are stopped.

---

## Project Pipeline

### 1. Data Preparation

The project is based on Persian customer reviews from Digikala.

The data preparation stage provides the foundation for the two classification tasks:

- Recommendation prediction
- Rating band prediction

### 2. Model Development

Two separate BERT-based models are used for the classification tasks.

| Model | Task |
|---|---|
| Recommendation BERT | Review recommendation classification |
| Rating Band BERT | Rating category classification |

Both models are loaded during API startup and used for inference.

### 3. Prediction Pipeline

The inference layer handles incoming Persian text and prepares model predictions.

It provides a unified interface for running predictions with the two trained models.

### 4. Evaluation Pipeline

The evaluation module supports batch prediction and evaluation using CSV files.

It provides:

- CSV input handling
- Batch inference
- Evaluation metrics
- Confusion matrix
- Prediction result export

Generated prediction files and evaluation outputs are stored under:

```text
artifacts/evaluation/
```

### 5. API Development

FastAPI exposes the prediction and evaluation functionality through REST endpoints.

Main endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Application health check |
| `/api/v1/predict` | POST | Review prediction |
| `/api/v1/evaluate` | POST | CSV evaluation |
| `/api/v1/evaluation/{filename}` | GET | Retrieve evaluation output |

Interactive API documentation is available at `/docs`.

### 6. Frontend Development

The user interface is implemented with Gradio.

It communicates with the FastAPI backend and provides access to:

- Review prediction
- Batch CSV evaluation
- Evaluation results

The frontend communicates with the API through the `API_URL` environment variable.

### 7. Dockerization

The application is containerized using Docker.

The Docker Compose configuration defines two services:

| Service | Technology | Container Port |
|---|---|---|
| API | FastAPI + Uvicorn | 8000 |
| Frontend | Gradio | 7860 |

Both services use the same Docker image but run separate application processes.

Docker Compose handles service startup and API health checks. Evaluation files are persisted through a bind mount.

---

## Project Structure

```text
persian-nlp-pipeline/
│
├── artifacts/
│   ├── bert_recommendation/
│   ├── model_band_best/
│   └── evaluation/
│
├── frontend/
│   └── app.py
│
├── src/
│   └── persian_nlp_pipeline/
│       ├── api.py
│       ├── evaluator.py
│       └── predictor.py
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| NLP Models | BERT, Hugging Face Transformers |
| Persian NLP | Hazm |
| Machine Learning Evaluation | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Backend | FastAPI, Uvicorn |
| Frontend | Gradio |
| Package Management | uv |
| Containerization | Docker, Docker Compose |

---

## Main Dependencies & Documentation

- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Hazm](https://github.com/roshan-research/hazm)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Gradio](https://www.gradio.app/docs)
- [Docker](https://docs.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [uv](https://docs.astral.sh/uv/)

---
## Dataset & Models

### Dataset
- [Hugging Face – Digikala Comments & Products](https://huggingface.co/datasets/RadeAI/Digikala_comments_products)
- [Kaggle – Digikala Comments & Products](https://www.kaggle.com/datasets/radeai/digikala-comments-and-products)

### Base Model
- [ParsBERT v2.0 – HooshvareLab](https://huggingface.co/HooshvareLab/bert-fa-base-uncased)

### Fine-tuned Models
The fine-tuned models are currently stored locally and are not publicly available
---
## Repository

[GitHub Repository](https://github.com/ItsJavadHosseini/persian-nlp-pipeline)