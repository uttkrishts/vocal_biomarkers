"""FastAPI backend for KintsugiHealth DAM voice biomarker model."""

import io
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add dam_model directory to path so we can import the pipeline
DAM_MODEL_DIR = str(Path(__file__).parent / "dam_model")
sys.path.insert(0, DAM_MODEL_DIR)

# ---------------------------------------------------------------------------
# Global pipeline singleton
# ---------------------------------------------------------------------------
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the DAM pipeline once at startup."""
    global pipeline
    print("Loading DAM pipeline — this may take a moment on first run...")
    from dam_model.pipeline import Pipeline as DAMPipeline

    pipeline = DAMPipeline()
    print("DAM pipeline loaded successfully.")
    yield
    # Cleanup
    pipeline = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Vocal Biomarker API",
    description="Analyze voice recordings for depression & anxiety biomarkers using the KintsugiHealth DAM model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}

SEVERITY_LABELS = {
    "depression": {
        0: {"label": "No Depression", "description": "PHQ-9 ≤ 9"},
        1: {"label": "Mild to Moderate", "description": "10 ≤ PHQ-9 ≤ 14"},
        2: {"label": "Severe", "description": "PHQ-9 ≥ 15"},
    },
    "anxiety": {
        0: {"label": "No Anxiety", "description": "GAD-7 ≤ 4"},
        1: {"label": "Mild", "description": "5 ≤ GAD-7 ≤ 9"},
        2: {"label": "Moderate", "description": "10 ≤ GAD-7 ≤ 14"},
        3: {"label": "Severe", "description": "GAD-7 ≥ 15"},
    },
}


def _enrich_result(result: dict, quantize: bool) -> dict:
    """Add severity labels when scores are quantized."""
    enriched = {"scores": result, "quantized": quantize}
    if quantize:
        enriched["labels"] = {}
        for key in ("depression", "anxiety"):
            score = result.get(key)
            if score is not None and score in SEVERITY_LABELS.get(key, {}):
                enriched["labels"][key] = SEVERITY_LABELS[key][score]
    return enriched


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
    }


@app.post("/analyze/file")
async def analyze_file(
    file: UploadFile = File(...),
    quantize: bool = Query(
        True,
        description="Return quantized severity levels (true) or raw scores (false)",
    ),
):
    """Analyze an uploaded audio file for depression & anxiety biomarkers."""
    if pipeline is None:
        return JSONResponse(
            status_code=503, content={"error": "Model is still loading."}
        )

    # Validate extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            },
        )

    # Save to temp file and run analysis
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = pipeline.run_on_file(tmp_path, quantize=quantize)
        return _enrich_result(result, quantize)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.websocket("/analyze/stream")
async def analyze_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming audio analysis.

    Protocol:
    1. Client connects
    2. Client sends binary audio chunks
    3. Client sends JSON: {"action": "analyze", "quantize": true/false}
    4. Server responds with JSON result
    5. Connection can be reused for another analysis or closed
    """
    await websocket.accept()
    audio_chunks: list[bytes] = []

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # Binary data = audio chunk
            if "bytes" in message:
                audio_chunks.append(message["bytes"])
                continue

            # Text data = control message
            if "text" in message:
                import json

                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"error": "Invalid JSON"})
                    continue

                action = data.get("action")

                if action == "analyze":
                    if pipeline is None:
                        await websocket.send_json({"error": "Model is still loading."})
                        continue

                    if not audio_chunks:
                        await websocket.send_json({"error": "No audio data received."})
                        continue

                    quantize = data.get("quantize", True)

                    # Combine chunks and save to temp file
                    combined = b"".join(audio_chunks)
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            suffix=".webm", delete=False
                        ) as tmp:
                            tmp.write(combined)
                            tmp_path = tmp.name

                        result = pipeline.run_on_file(tmp_path, quantize=quantize)
                        enriched = _enrich_result(result, quantize)
                        await websocket.send_json(enriched)
                    except Exception as e:
                        await websocket.send_json({"error": str(e)})
                    finally:
                        if tmp_path:
                            try:
                                os.unlink(tmp_path)
                            except Exception:
                                pass

                    # Reset for next recording
                    audio_chunks = []

                elif action == "reset":
                    audio_chunks = []
                    await websocket.send_json({"status": "reset"})

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
