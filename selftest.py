"""Самопроверка на вистинскиот модел, без микрофон.

1. Создава говор: англиски (Windows SAPI) и македонски (edge-tts, потребен е интернет и ffmpeg).
2. Говор -> француски говор + текст, за секоја реченица: време и VRAM.
3. Обратна насока: францускиот излез -> македонски текст (како титлови).
4. VAD: споени реченици со паузи и една долга без пауза -> колку реченици излегуваат.

    .venv\\Scripts\\python.exe selftest.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

import engine

OUT = Path(__file__).resolve().parent / "selftest_out"
OUT.mkdir(exist_ok=True)

EN = [
    "Good morning everyone, thank you for joining the call today.",
    "We need to confirm the delivery date before the end of the week.",
    "Could you please send me the updated price list by email?",
]
MK = [
    "Добро утро на сите, ви благодарам што дојдовте на состанокот.",
    "Треба да го потврдиме датумот на испорака до крајот на неделата.",
    "Ве молам, испратете ми го новиот ценовник по е-пошта.",
]


def save(path: Path, x: np.ndarray, sr: int = engine.SR):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def load(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return engine.resample(a.reshape(-1, ch).mean(axis=1), sr, engine.SR)


def sapi(text: str, path: Path):
    ps = ("Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SelectVoice('Microsoft Zira Desktop'); $s.SetOutputToWaveFile('{path}'); $s.Speak(\"{text}\"); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def edge_mk(text: str, path: Path):
    """Македонски тест говор преку edge-tts (mk-MK-AleksandarNeural), mp3 -> wav со ffmpeg."""
    exe = Path(sys.executable).with_name("edge-tts.exe")
    mp3 = path.with_suffix(".mp3")
    subprocess.run([str(exe), "--voice", "mk-MK-AleksandarNeural", "--text", text, "--write-media", str(mp3)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-ac", "1", "-ar", "16000", str(path)], check=True)
    mp3.unlink()


def main():
    report: dict = {"rows": []}
    # 1. тест говор
    en = []
    for i, t in enumerate(EN):
        p = OUT / f"en_{i}.wav"
        if not p.exists():
            sapi(t, p)
        en.append(load(p))
    mk = []
    for i, t in enumerate(MK):
        p = OUT / f"mk_{i}.wav"
        if not p.exists():
            edge_mk(t, p)
        mk.append(load(p))
    print(f"Тест говор: {len(en)} англиски, {len(mk)} македонски реченици")

    # 2. модел
    tr = engine.Translator()
    report["load_s"] = round(tr.load_s, 1)
    report["vram_after_load_gb"] = round(tr.vram_gb, 2)
    tr.speech_to_speech(en[0], "fra")  # загревање (CUDA kernels)

    for lang, clips, src in (("eng", en, EN), ("mkd", mk, MK)):
        for i, a in enumerate(clips):
            t0 = time.time(); fr_text, fr_wav = tr.speech_to_speech(a, "fra"); s2s = time.time() - t0
            save(OUT / f"{lang}_{i}_fra.wav", fr_wav)
            t0 = time.time(); asr = tr.speech_to_text(a, lang); asr_s = time.time() - t0
            t0 = time.time(); back = tr.speech_to_text(fr_wav, "mkd"); back_s = time.time() - t0
            row = {"src_lang": lang, "said": src[i], "heard": asr, "fra_text": fr_text,
                   "fra_to_mkd_subtitle": back, "speech_s": round(len(a) / engine.SR, 2),
                   "fra_speech_s": round(len(fr_wav) / engine.SR, 2), "s2s_s": round(s2s, 2),
                   "asr_s": round(asr_s, 2), "subtitle_s": round(back_s, 2)}
            report["rows"].append(row)
            print(json.dumps(row, ensure_ascii=False))
    report["vram_peak_gb"] = round(tr.vram_peak_gb(), 2)

    # 4. VAD
    sil = np.zeros(int(0.9 * engine.SR), np.float32)
    stream = np.concatenate([sil] + [x for a in en for x in (a, sil)])
    long = np.concatenate(en + en)  # без паузи меѓу реченици, > 12 сек
    stream = np.concatenate([stream, long, sil, sil])
    segs: list[np.ndarray] = []
    seg = engine.Segmenter(segs.append)
    for i in range(0, len(stream), 2048):
        seg.feed(stream[i:i + 2048])
    total_in = sum(len(a) for a in en) * 3 / engine.SR
    total_out = sum(len(s) for s in segs) / engine.SR
    report["vad"] = {"segments": len(segs), "segment_s": [round(len(s) / engine.SR, 1) for s in segs],
                     "speech_in_s": round(total_in, 1), "speech_out_s": round(total_out, 1),
                     "long_s": round(len(long) / engine.SR, 1)}
    print("VAD:", report["vad"])

    s2s = [r["s2s_s"] for r in report["rows"]]
    report["s2s_median_s"] = round(float(np.median(s2s)), 2)
    report["s2s_max_s"] = round(float(np.max(s2s)), 2)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nВчитување {report['load_s']} сек | VRAM по вчитување {report['vram_after_load_gb']} GB | "
          f"максимум {report['vram_peak_gb']} GB | говор->говор медијана {report['s2s_median_s']} сек, max {report['s2s_max_s']} сек")
    return 0


if __name__ == "__main__":
    sys.exit(main())
