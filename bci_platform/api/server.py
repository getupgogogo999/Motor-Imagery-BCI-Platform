"""
轻量本地推理 API（FastAPI）

启动:
    python -m bci_platform.api.server
    python -m bci_platform.api.server --model models/motor_imagery_a09.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from bci_platform.inference_engine import InferenceEngine
from bci_platform.model_registry import ModelRegistry

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn

    class PredictRequest(BaseModel):
        epochs: List[List[List[float]]]

    class PredictResponse(BaseModel):
        predictions: List[Dict[str, Any]]
        model_subject: str
        model_name: str

except ImportError:
    FastAPI = None
    BaseModel = object  # type: ignore


_engine: Optional[InferenceEngine] = None


def get_engine(model_path: str | None = None) -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine(model_path=model_path)
    return _engine


def create_app(model_path: str | None = None) -> "FastAPI":
    if FastAPI is None:
        raise ImportError("请安装: pip install fastapi uvicorn")

    app = FastAPI(title="BCI Inference API", version="1.0.0")
    engine = get_engine(model_path)

    @app.get("/health")
    def health():
        return {"status": "ok", "subject": engine.config["subject"]}

    @app.get("/models")
    def list_models():
        return {k: str(v) for k, v in ModelRegistry.list_available_models().items()}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        try:
            X = np.array(req.epochs, dtype=np.float64)
        except Exception as exc:
            raise HTTPException(400, f"Invalid epochs: {exc}") from exc
        if X.ndim != 3:
            raise HTTPException(400, "epochs shape must be (n_trials, n_channels, n_times)")
        results = engine.predict_batch(X)
        return PredictResponse(
            predictions=[r.to_dict() for r in results],
            model_subject=engine.config["subject"],
            model_name=engine.bundle.get("model_name", ""),
        )

    @app.post("/replay")
    def replay(batch_size: int = 32):
        return engine.replay_all(batch_size=batch_size)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    app = create_app(args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
