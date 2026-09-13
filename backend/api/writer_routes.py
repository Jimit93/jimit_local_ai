"""Refine-loop routes for the LinkedIn writer.

Kept in their own router so app.py only gains one include line. Both routes run in the
server process and reuse the model it already holds - never a second copy.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/writer", tags=["writer-refine"])


class RefineIn(BaseModel):
    topic: str = Field(min_length=3, max_length=2000)
    audience: str = ""
    pillar: str = ""
    length: str = "medium"
    cta: str = ""
    facts: str = ""
    temperature: float = 0.75
    max_iterations: int = Field(20, ge=0, le=20)
    max_llm_fixes: int = Field(3, ge=0, le=6)


class RefineBacktestIn(BaseModel):
    max_iterations: int = Field(20, ge=0, le=20)
    max_llm_fixes: int = Field(3, ge=0, le=6)
    brief_ids: list[str] | None = None


@router.post("/refine")
def writer_refine(inp: RefineIn) -> dict:
    """Describe -> Generate -> Test -> Refine, returning the post and every micro-iteration."""
    from backend.services.refine_loop import refine

    out = refine(inp.topic, audience=inp.audience, pillar=inp.pillar, length=inp.length,
                 cta=inp.cta, facts=inp.facts, temperature=inp.temperature,
                 max_iterations=inp.max_iterations, max_llm_fixes=inp.max_llm_fixes)
    if "error" in out:
        raise HTTPException(400, out["error"])
    return out


@router.post("/refine-backtest")
def writer_refine_backtest(inp: RefineBacktestIn) -> dict:
    """One-shot vs refine loop over the fixed brief set. Several minutes on CPU."""
    from backend.benchmarks.refine_backtest import BRIEFS, run_refine_backtest

    briefs = [b for b in BRIEFS if not inp.brief_ids or b["id"] in inp.brief_ids]
    if not briefs:
        raise HTTPException(400, "no briefs match brief_ids")
    return run_refine_backtest(briefs, max_iterations=inp.max_iterations,
                               max_llm_fixes=inp.max_llm_fixes)


@router.get("/refine-backtest/latest")
def writer_refine_backtest_latest() -> dict:
    from backend.benchmarks.refine_backtest import latest

    rep = latest()
    if rep is None:
        raise HTTPException(404, "no refine backtest has been run yet")
    return rep
