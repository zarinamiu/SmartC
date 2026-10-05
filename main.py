import io
import logging
from contextlib import asynccontextmanager
from typing import NamedTuple

import easyocr
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, Form, Request
from fastapi.responses import RedirectResponse
from PIL import Image
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("app.services")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "IlyaGusev/rut5_base_sum_gazeta"

SUMMARIZER_CONFIG = {
    "max_length": 250,
    "min_length": 30,
    "num_beams": 4,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.5,
    "early_stopping": True
}


class MLModels(NamedTuple):
    ocr: easyocr.Reader
    tokenizer: AutoTokenizer
    summarizer: AutoModelForSeq2SeqLM


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info(f"Initializing ML models on device: {DEVICE}")

        ocr_reader = easyocr.Reader(['ru', 'en'], gpu=(DEVICE == "cuda"))
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        summarizer = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

        app.state.ml_models = MLModels(
            ocr=ocr_reader,
            tokenizer=tokenizer,
            summarizer=summarizer
        )

        logger.info("All models loaded successfully.")
        yield
    except Exception as e:
        logger.critical(f"Critical error during model loading: {e}", exc_info=True)
        raise SystemExit(1)
    finally:
        if hasattr(app.state, "ml_models"):
            del app.state.ml_models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(title="Умный Конспект", lifespan=lifespan)


def _clean_text(text_lines: list[str]) -> str:
    cleaned = []
    for line in text_lines:
        line = line.strip()
        if not line:
            continue
        cleaned.append(line)
    return " ".join(cleaned)


@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


@app.post("/api/image-to-text")
async def process_image_to_text(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Use JPG or PNG.")

    try:
        content = await file.read()
        models = request.app.state.ml_models

        with Image.open(io.BytesIO(content)) as img:
            image_rgb = img.convert("RGB")
            text_lines = models.ocr.readtext(np.array(image_rgb), detail=0)

        return {
            "filename": file.filename,
            "raw_lines_count": len(text_lines),
            "extracted_text": _clean_text(text_lines)
        }
    except Exception as e:
        logger.error(f"OCR pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process image.")


@app.post("/api/summarize")
async def summarize_lecture(request: Request, text: str = Form(...)):
    clean_text = text.strip()

    if len(clean_text) < 100:
        raise HTTPException(status_code=400, detail="Text is too short for summarization (minimum 100 chars).")

    try:
        models = request.app.state.ml_models

        inputs = models.tokenizer(
            [clean_text],
            max_length=1024,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            output_ids = models.summarizer.generate(
                **inputs,
                **SUMMARIZER_CONFIG
            )

        summary = models.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {
            "original_length": len(clean_text),
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Summarization pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate summary.")
