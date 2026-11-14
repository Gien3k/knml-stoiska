"""
Kiosk Targowy KN Machine Learning - backend.

- serwuje frontend (SPA) i logotypy z folderu assets/ (podmiana bez przebudowy),
- zapisuje statystyki gier i rekordy dnia w SQLite (wolumen Dockera, przetrwa restart),
- przechowuje n-gramy ciągów 0/1 odwiedzających (wspólna pamięć predyktora w grze 1).
Wszystko działa lokalnie, bez internetu.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

BASE = Path(__file__).resolve().parent.parent
FRONTEND = Path(os.environ.get("FRONTEND_DIR", BASE / "frontend"))
ASSETS = Path(os.environ.get("ASSETS_DIR", BASE / "assets"))
DEFAULT_ASSETS = Path(os.environ.get("DEFAULT_ASSETS_DIR", BASE / "default-assets"))
DB_PATH = Path(os.environ.get("DB_PATH", BASE / "data" / "kiosk.db"))

mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("image/svg+xml", ".svg")

LOGO_NAMES = {"kolo", "rzeszow"}
LOGO_EXT = [("png", "image/png"), ("webp", "image/webp"), ("jpg", "image/jpeg"),
            ("jpeg", "image/jpeg"), ("svg", "image/svg+xml")]
GAMES = {"mind", "illusion", "autopilot"}
SEQ_RE = re.compile(r"^[01]{10,40}$")
MAX_CTX = 4
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

app = FastAPI(title="KNML Kiosk", docs_url=None, redoc_url=None)


# ----------------------------------------------------------------------------- baza
def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with db() as c:
        c.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS games(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                day TEXT NOT NULL, ts REAL NOT NULL,
                game TEXT NOT NULL, mode TEXT, score REAL NOT NULL, meta TEXT);
            CREATE INDEX IF NOT EXISTS games_day ON games(day);
            CREATE TABLE IF NOT EXISTS ngram(
                ctx TEXT PRIMARY KEY, c0 INTEGER NOT NULL DEFAULT 0, c1 INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


@app.on_event("startup")
def _startup() -> None:
    init_db()


# ----------------------------------------------------------------------------- nagłówki cache
class NoCacheStatic(BaseHTTPMiddleware):
    """Kiosk: zawsze świeże pliki po aktualizacji (ETag i tak oszczędza transfer)."""

    async def dispatch(self, request, call_next):
        resp = await call_next(request)
        path = request.url.path
        if path.startswith("/fonts/"):
            resp.headers["Cache-Control"] = "public, max-age=604800"
        elif not path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp


app.add_middleware(NoCacheStatic)


# ----------------------------------------------------------------------------- modele
class GameResult(BaseModel):
    game: str
    mode: Optional[str] = Field(default=None, max_length=32)
    score: float = Field(ge=0.0, le=1.0)
    day: str
    meta: Optional[dict] = None


# ----------------------------------------------------------------------------- API
@app.get("/api/health")
def health():
    return {"ok": True, "time": time.time()}


@app.get("/api/stats")
def stats(day: str = Query(...)):
    if not DAY_RE.match(day):
        raise HTTPException(400, "day must be YYYY-MM-DD")
    with db() as c:
        # win: udział gier wygranych przez ludzi (wynik > 50% = człowiek, < 50% = AI, dokładnie 50% = remis po pół punktu)
        row = c.execute("""SELECT COUNT(*) n, AVG(score) s,
                                AVG(CASE WHEN score > 0.5 + 1e-9 THEN 1.0 WHEN score < 0.5 - 1e-9 THEN 0.0 ELSE 0.5 END) w
                             FROM games WHERE day=?""", (day,)).fetchone()
        by = {r["game"]: {"games": r["n"], "human": round(r["s"] or 0, 4)}
              for r in c.execute("SELECT game, COUNT(*) n, AVG(score) s FROM games WHERE day=? GROUP BY game", (day,))}
        total = c.execute("SELECT COUNT(*) n FROM games").fetchone()["n"]
        mind = c.execute("SELECT MIN(json_extract(meta, '$.acc')) r FROM games WHERE day=? AND game='mind'",
                         (day,)).fetchone()["r"]
        best = c.execute("SELECT MIN(json_extract(meta, '$.seconds')) r FROM games WHERE day=? AND game='autopilot'",
                         (day,)).fetchone()["r"]
    return {"day": day, "games": row["n"], "human": round(row["s"], 4) if row["s"] is not None else None,
            "win": round(row["w"], 4) if row["w"] is not None else None,
            "byGame": by, "records": {"mind": mind, "autopilot": best}, "allTime": total}


@app.post("/api/result")
def add_result(r: GameResult):
    if r.game not in GAMES or not DAY_RE.match(r.day):
        raise HTTPException(400, "bad game/day")
    meta = json.dumps(r.meta, ensure_ascii=False)[:2000] if r.meta else None
    with db() as c:
        c.execute("INSERT INTO games(day, ts, game, mode, score, meta) VALUES(?,?,?,?,?,?)",
                  (r.day, time.time(), r.game, r.mode, float(r.score), meta))
    return stats(r.day)


@app.delete("/api/stats")
def reset_stats(day: str = Query(...)):
    if not DAY_RE.match(day):
        raise HTTPException(400, "day must be YYYY-MM-DD")
    with db() as c:
        c.execute("DELETE FROM games WHERE day=?", (day,))
    return stats(day)


# ----------------------------------------------------------------------------- wspólna pamięć n-gramów (gra 1)
# ----------------------------------------------------------------------------- ustawienia stoiska
# statsVisible: czy strona główna pokazuje dzisiejsze statystyki (obsługa może je schować)
SETTINGS_DEFAULT = {"statsVisible": True}


class Settings(BaseModel):
    statsVisible: Optional[bool] = None


@app.get("/api/settings")
def get_settings():
    out = dict(SETTINGS_DEFAULT)
    with db() as c:
        for r in c.execute("SELECT key, value FROM settings"):
            if r["key"] in out:
                out[r["key"]] = json.loads(r["value"])
    return out


@app.post("/api/settings")
def set_settings(s: Settings):
    with db() as c:
        for k, v in s.model_dump(exclude_none=True).items():
            c.execute("INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (k, json.dumps(v)))
    return get_settings()


class Sequence(BaseModel):
    seq: str


@app.post("/api/markov/sequence")
def add_sequence(s: Sequence):
    if not SEQ_RE.match(s.seq):
        raise HTTPException(400, "sequence must be 10-40 chars of 0/1")
    seq = s.seq
    with db() as c:
        for i in range(1, len(seq)):
            col = "c1" if seq[i] == "1" else "c0"
            for k in range(1, MAX_CTX + 1):
                if i - k < 0:
                    break
                ctx = seq[i - k:i]
                c.execute("INSERT INTO ngram(ctx, c0, c1) VALUES(?, 0, 0) ON CONFLICT(ctx) DO NOTHING", (ctx,))
                c.execute(f"UPDATE ngram SET {col} = {col} + 1 WHERE ctx=?", (ctx,))
    return {"ok": True}


@app.get("/api/markov/ngram")
def ngram():
    with db() as c:
        return {"counts": {r["ctx"]: [r["c0"], r["c1"]] for r in c.execute("SELECT ctx, c0, c1 FROM ngram")}}


# ----------------------------------------------------------------------------- logotypy
@app.get("/logo/{name}")
def logo(name: str):
    if name not in LOGO_NAMES:
        raise HTTPException(404)
    for folder in (ASSETS, DEFAULT_ASSETS):
        for ext, mime in LOGO_EXT:
            p = folder / f"logo-{name}.{ext}"
            if p.is_file():
                return FileResponse(p, media_type=mime, headers={"Cache-Control": "no-store"})
    raise HTTPException(404)


@app.exception_handler(404)
async def not_found(request, exc):
    if request.url.path.startswith(("/api/", "/logo/")):
        return JSONResponse({"detail": "not found"}, status_code=404)
    return FileResponse(FRONTEND / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
