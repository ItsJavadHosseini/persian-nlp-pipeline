from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

from persian_nlp_pipeline.evaluator import CSVEvaluator
from persian_nlp_pipeline.predictor import ReviewPredictor


ROOT_DIR = Path(__file__).resolve().parents[2]
EVALUATION_DIR = ROOT_DIR / "artifacts" / "evaluation"
UPLOAD_DIR = EVALUATION_DIR / "uploads"

EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once when the API starts.
    predictor = ReviewPredictor()

    app.state.predictor = predictor
    app.state.evaluator = CSVEvaluator(predictor)

    yield


app = FastAPI(
    title="Persian NLP Pipeline API",
    description="API for Persian review prediction and evaluation.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    text: str


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": hasattr(app.state, "predictor"),
    }


@app.post("/api/v1/predict")
def predict(request: PredictionRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    results = app.state.predictor.predict_batch([text])

    return results[0]


@app.post("/api/v1/evaluate")
async def evaluate_csv(
    file: UploadFile = File(...),
    task: str = Form("both"),
    batch_size: int = Form(16),
    chunksize: int = Form(1000),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    if task not in {"both", "recommendation", "rating_band"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid task.",
        )

    if batch_size < 1 or chunksize < 1:
        raise HTTPException(
            status_code=400,
            detail="batch_size and chunksize must be positive.",
        )

    file_id = uuid4().hex
    input_path = UPLOAD_DIR / f"{file_id}.csv"
    output_path = EVALUATION_DIR / f"{file_id}_predictions.csv"

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV is empty.",
            )

        input_path.write_bytes(content)

        results = await run_in_threadpool(
            app.state.evaluator.evaluate_csv,
            input_path=str(input_path),
            output_path=str(output_path),
            task=task,
            batch_size=batch_size,
            chunksize=chunksize,
        )

        return {
            "status": "completed",
            "metrics": results,
            "output_file": output_path.name,
            "download_url": f"/api/v1/evaluation/{output_path.name}",
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(exc)}",
        )

    finally:
        input_path.unlink(missing_ok=True)


@app.get("/api/v1/evaluation/{filename}")
def download_evaluation(filename: str):
    # Prevent directory traversal.
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_path = EVALUATION_DIR / filename

    if not file_path.is_file() or not filename.endswith("_predictions.csv"):
        raise HTTPException(
            status_code=404,
            detail="Evaluation file not found.",
        )

    return FileResponse(
        path=file_path,
        media_type="text/csv",
        filename=filename,
    )