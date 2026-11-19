"""REST API (FastAPI). Run with: uvicorn querentcrew.api:app --reload"""

from __future__ import annotations

import io
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import __version__
from .service import QuerentCrew

app = FastAPI(
    title="QuerentCrew API",
    version=__version__,
    description="Multi-agent analytics assistant: ask business questions in plain English.",
)


@lru_cache(maxsize=1)
def get_pilot() -> QuerentCrew:
    return QuerentCrew()


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000, examples=["Why did revenue drop in the South in Q3?"])
    thread_id: str | None = Field(default=None, description="Reuse to continue a conversation")


class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool
    comment: str = ""


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/schema")
def schema() -> dict:
    return get_pilot().schema()


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    try:
        return get_pilot().ask(req.question, req.thread_id).to_dict()
    except Exception as exc:  # surface provider/config errors clearly
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/approve")
def approve(req: ApprovalRequest) -> dict:
    try:
        return get_pilot().resume(req.thread_id, req.approved, req.comment).to_dict()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.get("/history/{thread_id}")
def history(thread_id: str) -> dict:
    return {"thread_id": thread_id, "history": get_pilot().history(thread_id)}


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)) -> dict:
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File larger than 20 MB")
    table, rows = get_pilot().import_csv(io.BytesIO(content), file.filename)
    return {"table": table, "rows": rows}
