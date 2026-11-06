"""Толкувач: локален сервер на http://127.0.0.1:8813

    .venv\\Scripts\\python.exe server.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import sys

for _s in (sys.stdout, sys.stderr):  # кирилица во Windows конзола
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
import threading
import time
import wave
from contextlib import asynccontextmanager
from dataclasses import fields
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

import engine

PORT = int(__import__("os").environ.get("PORT", 8813))
ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Толкувач")

clients: set[WebSocket] = set()
loop: asyncio.AbstractEventLoop | None = None
history: list[dict] = []
_last_level: dict[str, float] = {}


def emit(ev: dict, throttle: bool = False):
    """Се повикува од аудио/GPU нишките. Нивоата на звук се праќаат најмногу 10 пати во секунда."""
    if throttle:
        k = ev.get("who", "")
        now = time.time()
        if now - _last_level.get(k, 0) < 0.1:
            return
        _last_level[k] = now
    if ev.get("type") in ("line", "log", "status", "dropped"):
        history.append(ev)
        del history[:-300]
    if ev.get("type") == "log":
        print(ev.get("text"), flush=True)
    if loop is None:
        return
    msg = json.dumps(ev, ensure_ascii=False)
    for ws in list(clients):
        asyncio.run_coroutine_threadsafe(_send(ws, msg), loop)


async def _send(ws: WebSocket, msg: str):
    try:
        await ws.send_text(msg)
    except Exception:  # noqa: BLE001
        clients.discard(ws)


interp = engine.Interpreter(emit)


@asynccontextmanager
async def lifespan(_app):
    global loop
    loop = asyncio.get_running_loop()
    threading.Thread(target=interp.load_model, daemon=True).start()
    yield
    interp.stop()


app.router.lifespan_context = lifespan


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/status")
def status():
    return interp.status()


@app.get("/api/devices")
def devices():
    return engine.list_devices()


@app.get("/api/languages")
def languages():
    return {"their": engine.SPEECH_LANGS, "mine": engine.MY_LANGS}


@app.post("/api/start")
async def start(body: dict):
    names = {f.name for f in fields(engine.Settings)}
    s = engine.Settings(**{k: v for k, v in body.items() if k in names})
    if s.their_lang not in engine.SPEECH_LANGS:
        raise HTTPException(400, "Тој јазик нема говорен излез.")
    try:
        await asyncio.to_thread(interp.start, s)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return interp.status()


@app.post("/api/stop")
async def stop():
    await asyncio.to_thread(interp.stop)
    return interp.status()


@app.post("/api/passthrough")
def passthrough(body: dict):
    interp.passthrough = bool(body.get("on"))
    emit({"type": "passthrough", "on": interp.passthrough})
    return {"on": interp.passthrough}


def _read_wav(data: bytes) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf  # ако е инсталиран, чита и mp3/flac/ogg
        a, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return a.mean(axis=1), sr
    except Exception:  # noqa: BLE001
        pass
    with wave.open(io.BytesIO(data)) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise HTTPException(400, "Прати WAV со 16-битни примероци.")
    a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    return a.reshape(-1, ch).mean(axis=1), sr


def _wav_b64(x: np.ndarray, sr: int = engine.SR) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode()


@app.post("/api/test")
async def test(file: UploadFile = File(...), mode: str = Form("speech"), tgt: str = Form("fra"),
               voice: str = Form("local"), gender: str = Form("f")):
    audio, sr = _read_wav(await file.read())
    if len(audio) / sr > 30:
        raise HTTPException(400, "Пробата е до 30 секунди.")
    try:
        r = await asyncio.to_thread(interp.translate_file, audio, sr, mode, tgt, voice, gender)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    wav = r.pop("wav")
    if wav is not None:
        r["wav_b64"] = _wav_b64(wav)
    return JSONResponse(r)


@app.post("/api/voice_preview")
async def voice_preview(body: dict):
    tgt = body.get("tgt", "fra")
    if tgt not in engine.SPEECH_LANGS:
        raise HTTPException(400, "Тој јазик нема говорен излез.")
    try:
        r = await asyncio.to_thread(interp.preview_voice, tgt, body.get("voice", "natural"), body.get("gender", "f"))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    r["wav_b64"] = _wav_b64(r.pop("wav"))
    return r


@app.post("/api/inject")
async def inject(file: UploadFile = File(...), who: str = Form("me")):
    audio, sr = _read_wav(await file.read())
    try:
        secs = interp.inject(audio, sr, "me" if who == "me" else "them")
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"seconds": secs}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    clients.add(sock)
    await sock.send_text(json.dumps({"type": "hello", "status": interp.status(), "history": history[-80:]},
                                    ensure_ascii=False))
    try:
        while True:
            await sock.receive_text()
    except WebSocketDisconnect:
        clients.discard(sock)


if __name__ == "__main__":
    print(f"Толкувач: http://127.0.0.1:{PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
