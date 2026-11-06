"""Толкувач: моторот.

Еден модел (SeamlessM4T v2) на GPU, две насоки:
  ТИ   : микрофон -> VAD -> говор во говор (на пр. француски) -> излезен уред (VB-Cable)
  ТИЕ  : звук од состанокот (WASAPI loopback) -> VAD -> текст на твојот јазик -> титлови

Една GPU нишка ги работи сите задачи. Твојот говор има предност; ако нивните
реченици заостанат повеќе од max_lag секунди, се прескокнуваат.
"""
from __future__ import annotations

import ctypes
import datetime as dt
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import itertools
import queue
import threading
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

SR = 16000            # моделот и VAD работат на 16 kHz
FRAME = 512           # Silero VAD бара 512 примероци на 16 kHz (32 ms)
MODEL_ID = "facebook/seamless-m4t-v2-large"
ROOT = Path(__file__).resolve().parent
TRANSCRIPTS = ROOT / "transcripts"

# loopback без звук на Windows фрла ова при секое читање; безопасно е.
# soundcard го поставува својот филтер при import, затоа прво import, па нашиот филтер.
import soundcard  # noqa: E402,F401
warnings.filterwarnings("ignore", message="data discontinuity")

# Silero VAD (TorchScript) не смее да се повикува од две нишки истовремено:
# тоа завршува со нативен пад (0xC0000409). Една брава за сите повици.
VAD_LOCK = threading.Lock()

# Јазици со говорен излез кај SeamlessM4T v2 (подмножество, со имиња на македонски)
SPEECH_LANGS = {
    "fra": "француски", "eng": "англиски", "deu": "германски", "ita": "италијански",
    "spa": "шпански", "por": "португалски", "nld": "холандски", "pol": "полски",
    "ces": "чешки", "slk": "словачки", "ron": "романски", "rus": "руски",
    "ukr": "украински", "tur": "турски", "swe": "шведски", "dan": "дански",
    "fin": "фински", "cmn": "кинески", "jpn": "јапонски", "kor": "корејски",
    "arb": "арапски",
}
# Јазици за тебе: моделот ги прима како говор и пишува текст на нив (македонски нема говорен излез)
MY_LANGS = {
    "mkd": "македонски", "eng": "англиски", "srp": "српски", "hrv": "хрватски",
    "bul": "бугарски", "slv": "словенечки", "deu": "германски", "fra": "француски",
}


# Природни гласови (Microsoft Edge neural TTS, бара интернет): (женски, машки) по јазик
NEURAL_VOICES = {
    "fra": ("fr-FR-DeniseNeural", "fr-FR-HenriNeural"), "eng": ("en-US-AvaNeural", "en-US-AndrewNeural"),
    "deu": ("de-DE-KatjaNeural", "de-DE-ConradNeural"), "ita": ("it-IT-ElsaNeural", "it-IT-DiegoNeural"),
    "spa": ("es-ES-ElviraNeural", "es-ES-AlvaroNeural"), "por": ("pt-PT-RaquelNeural", "pt-PT-DuarteNeural"),
    "nld": ("nl-NL-ColetteNeural", "nl-NL-MaartenNeural"), "pol": ("pl-PL-ZofiaNeural", "pl-PL-MarekNeural"),
    "ces": ("cs-CZ-VlastaNeural", "cs-CZ-AntoninNeural"), "slk": ("sk-SK-ViktoriaNeural", "sk-SK-LukasNeural"),
    "ron": ("ro-RO-AlinaNeural", "ro-RO-EmilNeural"), "rus": ("ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"),
    "ukr": ("uk-UA-PolinaNeural", "uk-UA-OstapNeural"), "tur": ("tr-TR-EmelNeural", "tr-TR-AhmetNeural"),
    "swe": ("sv-SE-SofieNeural", "sv-SE-MattiasNeural"), "dan": ("da-DK-ChristelNeural", "da-DK-JeppeNeural"),
    "fin": ("fi-FI-NooraNeural", "fi-FI-HarriNeural"), "cmn": ("zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"),
    "jpn": ("ja-JP-NanamiNeural", "ja-JP-KeitaNeural"), "kor": ("ko-KR-SunHiNeural", "ko-KR-InJoonNeural"),
    "arb": ("ar-SA-ZariyahNeural", "ar-SA-HamedNeural"),
}
# Локални гласови на Seamless (speaker_id), избрани по висина и мирност на гласот: (женски, машки)
LOCAL_VOICES = {"f": 31, "m": 32}  # од скен на сите 200: 184 Hz и 121 Hz, најмирна висина


def neural_tts(text: str, lang: str, gender: str = "f", timeout: float = 8.0, on_audio=None) -> np.ndarray:
    """Природен глас преку edge-tts. Враќа звук 16 kHz float32. Фрла грешка ако нема интернет.

    Ако е даден `on_audio`, звукот се праќа на делови додека пристигнува, па говорот
    почнува пред целата реченица да биде симната. MP3 се декодира одново од почеток
    (кратко е), а се праќа само новото, без последните ~150 ms кои уште може да се сменат.
    """
    import asyncio
    import io

    import edge_tts
    import soundfile as sf

    voice = NEURAL_VOICES[lang][0 if gender == "f" else 1]
    buf = bytearray()
    sent = 0
    margin = int(0.15 * SR)

    def decode() -> np.ndarray:
        a, sr = sf.read(io.BytesIO(bytes(buf)), dtype="float32", always_2d=True)
        return resample(a.mean(axis=1), sr, SR)

    def push(final: bool):
        nonlocal sent
        try:
            x = decode()
        except Exception:  # noqa: BLE001  уште нема цела MP3 рамка
            return
        end = len(x) if final else len(x) - margin
        if end > sent:
            on_audio(x[sent:end].copy())
            sent = end

    async def go():
        last = 0
        async for c in edge_tts.Communicate(text, voice).stream():
            if c["type"] == "audio":
                buf.extend(c["data"])
                if on_audio and len(buf) - last >= 3000:  # ~0.5 сек звук (48 kbit/s)
                    last = len(buf)
                    push(False)

    asyncio.run(asyncio.wait_for(go(), timeout))
    if on_audio:
        push(True)
    return decode()


def com_init() -> None:
    """soundcard користи WASAPI преку COM; секоја нишка мора да го иницијализира."""
    try:
        ctypes.windll.ole32.CoInitializeEx(None, 0)
    except Exception:
        pass


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x.astype(np.float32, copy=False)
    from scipy.signal import resample_poly
    g = np.gcd(int(sr_from), int(sr_to))
    return resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)


# ----------------------------------------------------------------------------- модел

class Translator:
    def __init__(self, log=print):
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2Model

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        log(f"Вчитувам {MODEL_ID} на {self.device} ({str(self.dtype).split('.')[-1]}) ...")
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = SeamlessM4Tv2Model.from_pretrained(MODEL_ID, torch_dtype=self.dtype).to(self.device).eval()
        self.load_s = time.time() - t0
        self.vram_gb = (torch.cuda.memory_allocated() / 1e9) if self.device == "cuda" else 0.0
        log(f"Моделот е вчитан за {self.load_s:.0f} сек, VRAM {self.vram_gb:.2f} GB")

    def _inputs(self, audio: np.ndarray):
        inp = self.processor(audio=audio, sampling_rate=SR, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        if "input_features" in inp:
            inp["input_features"] = inp["input_features"].to(self.dtype)
        return inp

    def _decode(self, ids) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self.processor.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

    def speech_to_speech(self, audio: np.ndarray, tgt: str, speaker_id: int = 0):
        """Говор -> говор. Враќа (текст на превод, звук 16 kHz float32)."""
        with self.torch.inference_mode():
            out = self.model.generate(**self._inputs(audio), tgt_lang=tgt, speaker_id=speaker_id,
                                      return_intermediate_token_ids=True)
        wl = getattr(out, "waveform_lengths", None)
        n = int(wl.reshape(-1)[0]) if wl is not None else None
        wav = out.waveform[0].float().cpu().numpy()
        if n:
            wav = wav[:n]
        return self._decode(out.sequences[0]), wav

    def text_to_speech(self, text: str, src: str, tgt: str, speaker_id: int = 0):
        """Текст -> преведен говор (за преслушување на глас)."""
        inp = self.processor(text=text, src_lang=src, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        with self.torch.inference_mode():
            out = self.model.generate(**inp, tgt_lang=tgt, speaker_id=speaker_id, return_intermediate_token_ids=True)
        wl = getattr(out, "waveform_lengths", None)
        n = int(wl.reshape(-1)[0]) if wl is not None else None
        wav = out.waveform[0].float().cpu().numpy()
        return self._decode(out.sequences[0]), (wav[:n] if n else wav)

    def speech_to_text(self, audio: np.ndarray, tgt: str) -> str:
        with self.torch.inference_mode():
            out = self.model.generate(**self._inputs(audio), tgt_lang=tgt, generate_speech=False)
        seq = out.sequences if hasattr(out, "sequences") else out[0]
        return self._decode(seq[0])

    def vram_peak_gb(self) -> float:
        if self.device != "cuda":
            return 0.0
        return self.torch.cuda.max_memory_allocated() / 1e9


# ----------------------------------------------------------------------------- VAD

class Segmenter:
    """Дели континуиран звук на реченици со Silero VAD.

    Реченица завршува по `silence_ms` тишина. Ако трае подолго од `max_s`, се сече
    на најтивкото место во последните 1.5 сек, а остатокот веднаш продолжува како
    нова реченица (не се губи ништо до следниот почеток на говор).
    """

    def __init__(self, on_segment, threshold=0.5, silence_ms=550, min_s=0.4, max_s=12.0, preroll_ms=250):
        from silero_vad import load_silero_vad
        self.vad = load_silero_vad()
        self.on_segment = on_segment
        self.th = threshold
        self.silence_frames = int(silence_ms / 32)
        self.min_frames = int(min_s * SR / FRAME)
        self.max_frames = int(max_s * SR / FRAME)
        self.pre_frames = int(preroll_ms / 32)
        self.buf = np.zeros(0, np.float32)
        self.pre: list[np.ndarray] = []
        self.frames: list[np.ndarray] = []
        self.probs: list[float] = []
        self.silent = 0
        self.active = False
        self.level = 0.0

    def reset(self):
        with VAD_LOCK:
            self.vad.reset_states()
            self.buf = np.zeros(0, np.float32)
            self.pre, self.frames, self.probs = [], [], []
            self.silent, self.active = 0, False

    def feed(self, x: np.ndarray):
        with VAD_LOCK:
            self._feed(x)

    def _feed(self, x: np.ndarray):
        import torch
        self.buf = np.concatenate([self.buf, x.astype(np.float32, copy=False)])
        while len(self.buf) >= FRAME:
            fr, self.buf = self.buf[:FRAME], self.buf[FRAME:]
            self.level = 0.8 * self.level + 0.2 * float(np.sqrt(np.mean(fr * fr)))
            p = float(self.vad(torch.from_numpy(fr.copy()), SR).item())
            if not self.active:
                self.pre.append(fr)
                if len(self.pre) > self.pre_frames:
                    self.pre.pop(0)
                if p >= self.th:
                    self.active, self.silent = True, 0
                    self.frames = list(self.pre)
                    self.probs = [0.0] * len(self.pre)
                    self.pre = []
                continue
            self.frames.append(fr)
            self.probs.append(p)
            self.silent = self.silent + 1 if p < self.th - 0.15 else 0
            if self.silent >= self.silence_frames:
                self._emit(len(self.frames) - self.silent + 3, keep_active=False)
            elif len(self.frames) >= self.max_frames:
                look = min(len(self.frames) - 1, int(1.5 * SR / FRAME))
                tail = self.probs[-look:]
                cut = len(self.frames) - look + int(np.argmin(tail)) + 1
                self._emit(cut, keep_active=True)

    def _emit(self, cut: int, keep_active: bool):
        cut = max(1, min(cut, len(self.frames)))
        seg, rest = self.frames[:cut], self.frames[cut:]
        speech = sum(1 for q in self.probs[:cut] if q >= self.th)
        if speech >= self.min_frames // 2 and len(seg) >= self.min_frames:
            self.on_segment(np.concatenate(seg))
        if keep_active:
            self.frames, self.probs = rest, self.probs[cut:]
            self.silent = 0
        else:
            self.frames, self.probs, self.pre = [], [], rest[-self.pre_frames:]
            self.active, self.silent = False, 0


# ----------------------------------------------------------------------------- уреди

def list_devices() -> dict:
    import soundcard as sc
    com_init()
    mics = [m.name for m in sc.all_microphones(include_loopback=False)]
    spks = [s.name for s in sc.all_speakers()]
    return {
        "microphones": mics,
        "speakers": spks,
        "default_microphone": sc.default_microphone().name,
        "default_speaker": sc.default_speaker().name,
        "vb_cable": next((s for s in spks if "cable" in s.lower()), None),
    }


def _find(items, name):
    for it in items:
        if it.name == name:
            return it
    for it in items:
        if name and name.lower() in it.name.lower():
            return it
    return None


# ----------------------------------------------------------------------------- сесија

@dataclass
class Settings:
    my_lang: str = "mkd"
    their_lang: str = "fra"
    mic: str = ""
    out_device: str = ""      # VB-Cable за состанок, слушалки за проба
    meeting_device: str = ""  # каде го слушаш состанокот (од тука loopback)
    translate_me: bool = True
    translate_them: bool = True
    show_original: bool = True
    voice: str = "natural"    # "natural" = природен глас (интернет), "local" = гласот на Seamless
    gender: str = "f"         # "f" женски, "m" машки
    speaker_id: int = -1      # -1 = локалниот глас се избира според gender
    max_lag: float = 8.0
    silence_ms: int = 550


@dataclass(order=True)
class Job:
    priority: int
    seq: int
    kind: str = field(compare=False)       # "out" = ти -> тие, "in" = тие -> ти
    audio: np.ndarray = field(compare=False, repr=False)
    t_end: float = field(compare=False)    # кога заврши реченицата


class Interpreter:
    def __init__(self, emit):
        self.emit = emit                  # emit(dict) - праќа настан до UI
        self.tr: Translator | None = None
        self.settings = Settings()
        self.jobs: "queue.PriorityQueue[Job]" = queue.PriorityQueue()
        self.play_q: "queue.Queue[np.ndarray | None]" = queue.Queue()
        self.seq = itertools.count()
        self.running = False
        self.passthrough = False
        self.stop_ev = threading.Event()
        self.threads: list[threading.Thread] = []
        self.playing_until = 0.0
        self.seg_me: Segmenter | None = None
        self.seg_them: Segmenter | None = None
        self.stats = {"out": [], "in": [], "dropped": 0}
        self.log_path: Path | None = None
        self.loading = False
        self.injecting: set[str] = set()
        self.natural_down_until = 0.0  # по грешка на природниот глас, минута зборува локалниот
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- модел
    def load_model(self):
        if self.tr or self.loading:
            return
        self.loading = True
        self.emit({"type": "status", "state": "loading"})
        try:
            self.tr = Translator(log=lambda m: self.emit({"type": "log", "text": m}))
            self.emit({"type": "status", "state": "ready", "vram_gb": round(self.tr.vram_gb, 2),
                       "device": self.tr.device, "load_s": round(self.tr.load_s)})
        except Exception as e:  # noqa: BLE001
            self.emit({"type": "status", "state": "error", "error": f"{type(e).__name__}: {e}"})
        finally:
            self.loading = False

    # ---- старт/стоп
    def start(self, settings: Settings):
        if self.running:
            self.stop()
        if not self.tr:
            raise RuntimeError("Моделот уште не е вчитан.")
        self.settings = settings
        self.stop_ev.clear()
        self.running = True
        self.stats = {"out": [], "in": [], "dropped": 0}
        TRANSCRIPTS.mkdir(exist_ok=True)
        self.log_path = TRANSCRIPTS / f"sostanok_{dt.datetime.now():%Y-%m-%d_%H%M}.txt"
        self._write(f"# Состанок {dt.datetime.now():%Y-%m-%d %H:%M}  ти: {settings.my_lang}  тие: {settings.their_lang}\n")
        self.seg_me = Segmenter(lambda a: self._enqueue("out", a), silence_ms=settings.silence_ms)
        self.seg_them = Segmenter(lambda a: self._enqueue("in", a), silence_ms=settings.silence_ms)
        self.threads = [
            threading.Thread(target=self._mic_loop, daemon=True),
            threading.Thread(target=self._loopback_loop, daemon=True),
            threading.Thread(target=self._player_loop, daemon=True),
        ]
        for t in self.threads:
            t.start()
        if settings.voice == "natural" and settings.their_lang in NEURAL_VOICES:
            # првиот повик до природниот глас е бавен (врска, DNS); загреј го пред првата реченица
            def warm():
                try:
                    neural_tts("OK", settings.their_lang, settings.gender)
                except Exception:  # noqa: BLE001
                    pass
            threading.Thread(target=warm, daemon=True).start()
        self.emit({"type": "status", "state": "running", "log": self.log_path.name})

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.stop_ev.set()
        self.play_q.put(None)
        for t in self.threads:
            t.join(timeout=3)
        self.threads = []
        while not self.jobs.empty():
            try:
                self.jobs.get_nowait()
            except queue.Empty:
                break
        self.emit({"type": "status", "state": "ready", "summary": self.summary()})

    def summary(self) -> dict:
        def agg(xs, k):
            v = [x[k] for x in xs]
            return round(float(np.median(v)), 2) if v else None
        return {
            "out_n": len(self.stats["out"]), "in_n": len(self.stats["in"]),
            "out_latency_med": agg(self.stats["out"], "latency"),
            "in_latency_med": agg(self.stats["in"], "latency"),
            "out_compute_med": agg(self.stats["out"], "compute"),
            "dropped": self.stats["dropped"],
            "vram_peak_gb": round(self.tr.vram_peak_gb(), 2) if self.tr else None,
        }

    # ---- аудио нишки
    def _mic_loop(self):
        import soundcard as sc
        com_init()
        s = self.settings
        mic = _find(sc.all_microphones(), s.mic) or sc.default_microphone()
        self.emit({"type": "log", "text": f"Микрофон: {mic.name}"})
        try:
            with mic.recorder(samplerate=SR, channels=1, blocksize=FRAME) as rec:
                while not self.stop_ev.is_set():
                    x = rec.record(numframes=FRAME * 4)[:, 0]
                    if self.passthrough:
                        self.play_q.put(x.copy())
                        self.seg_me.reset()
                    elif s.translate_me and "me" not in self.injecting:
                        self.seg_me.feed(x)
                    self.emit({"type": "level", "who": "me", "v": round(self.seg_me.level, 4)}, throttle=True)
        except Exception as e:  # noqa: BLE001
            self.emit({"type": "log", "level": "error", "text": f"Микрофон: {type(e).__name__}: {e}"})

    def _loopback_loop(self):
        import soundcard as sc
        com_init()
        s = self.settings
        spk = _find(sc.all_speakers(), s.meeting_device) or sc.default_speaker()
        src = sc.get_microphone(id=str(spk.name), include_loopback=True)
        out = _find(sc.all_speakers(), s.out_device) or sc.default_speaker()
        same_as_out = out.name == spk.name
        self.emit({"type": "log", "text": f"Звук од состанокот: {spk.name}"
                   + ("  (ист уред како излезот: додека се пушта твојот превод, не слушам)" if same_as_out else "")})
        try:
            with src.recorder(samplerate=SR, channels=2, blocksize=FRAME) as rec:
                while not self.stop_ev.is_set():
                    x = rec.record(numframes=FRAME * 4).mean(axis=1).astype(np.float32)
                    if not s.translate_them or "them" in self.injecting:
                        continue
                    if same_as_out and time.time() < self.playing_until + 0.3:
                        x = np.zeros_like(x)  # не го преведувај сопствениот превод
                    self.seg_them.feed(x)
                    self.emit({"type": "level", "who": "them", "v": round(self.seg_them.level, 4)}, throttle=True)
        except Exception as e:  # noqa: BLE001
            self.emit({"type": "log", "level": "error", "text": f"Loopback: {type(e).__name__}: {e}"})

    def _player_loop(self):
        import soundcard as sc
        com_init()
        s = self.settings
        spk = _find(sc.all_speakers(), s.out_device) or sc.default_speaker()
        self.emit({"type": "log", "text": f"Превод излегува на: {spk.name}"})
        try:
            with spk.player(samplerate=SR, channels=1, blocksize=FRAME) as pl:
                while not self.stop_ev.is_set():
                    try:
                        x = self.play_q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if x is None:
                        break
                    self.playing_until = max(self.playing_until, time.time()) + len(x) / SR
                    pl.play(x.astype(np.float32))
        except Exception as e:  # noqa: BLE001
            self.emit({"type": "log", "level": "error", "text": f"Излез: {type(e).__name__}: {e}"})

    # ---- GPU
    def _enqueue(self, kind: str, audio: np.ndarray):
        self.jobs.put(Job(0 if kind == "out" else 1, next(self.seq), kind, audio, time.time()))
        self.emit({"type": "queue", "n": self.jobs.qsize()})

    def _worker(self):
        while True:
            job = self.jobs.get()
            if not self.running or not self.tr:
                continue
            try:
                self._run(job)
            except Exception as e:  # noqa: BLE001
                self.emit({"type": "log", "level": "error", "text": f"Превод: {type(e).__name__}: {e}"})

    def _run(self, job: Job):
        s = self.settings
        wait = time.time() - job.t_end
        dur = len(job.audio) / SR
        if job.kind == "in" and wait > s.max_lag:
            self.stats["dropped"] += 1
            self.emit({"type": "dropped", "wait": round(wait, 1), "dropped": self.stats["dropped"]})
            return
        t0 = time.time()
        if job.kind == "out":
            first = []

            def play(x):  # првиот дел звук е тоа што другите го чујат, од тука се мери доцнењето
                if not first:
                    first.append(time.time())
                if not self.passthrough:
                    self.play_q.put(x)
            text, _, used = self.speak(job.audio, s.their_lang, s.voice, s.gender, s.speaker_id, on_audio=play)
            compute = (first[0] if first else time.time()) - t0
            latency = (first[0] if first else time.time()) - job.t_end
        else:
            text, used = self.tr.speech_to_text(job.audio, s.my_lang), None
            compute = time.time() - t0
            latency = time.time() - job.t_end
        original = ""
        if s.show_original and self.jobs.empty():
            original = self.tr.speech_to_text(job.audio, s.my_lang if job.kind == "out" else s.their_lang)
        rec = {"audio": round(dur, 2), "compute": round(compute, 2), "latency": round(latency, 2), "wait": round(wait, 2)}
        self.stats[job.kind].append(rec)
        self.emit({"type": "line", "dir": job.kind, "text": text, "original": original,
                   "time": dt.datetime.now().strftime("%H:%M:%S"), "voice": used, **rec})
        who = "ТИ  " if job.kind == "out" else "ТИЕ "
        self._write(f"[{dt.datetime.now():%H:%M:%S}] {who}{('(' + original + ') ') if original else ''}-> {text}"
                    f"   [превод {compute:.1f} сек, доцнење {latency:.1f} сек]\n")

    def speak(self, audio: np.ndarray, tgt: str, voice: str, gender: str, speaker_id: int = -1, on_audio=None):
        """Говор -> преведен говор со избраниот глас. Враќа (текст, звук, кој глас навистина зборуваше).

        Ако е даден `on_audio`, звукот се праќа таму (природниот глас на делови, додека пристигнува).
        """
        sid = speaker_id if speaker_id >= 0 else LOCAL_VOICES.get(gender, 0)
        if voice == "natural" and tgt in NEURAL_VOICES and time.time() >= self.natural_down_until:
            text = self.tr.speech_to_text(audio, tgt)
            if not text:
                return "", np.zeros(0, np.float32), "natural"
            sent = []

            def part(x):
                sent.append(1)
                on_audio(x)
            try:
                return text, neural_tts(text, tgt, gender, on_audio=part if on_audio else None), "natural"
            except Exception as e:  # noqa: BLE001  нема интернет -> локален глас, реченицата не се губи
                if sent:  # дел веќе се слушна; да не се повтори реченицата со друг глас
                    self.emit({"type": "log", "level": "error",
                               "text": f"Природниот глас прекина на половина реченица ({type(e).__name__})."})
                    return text, np.zeros(0, np.float32), "natural"
                self.natural_down_until = time.time() + 60
                self.emit({"type": "log", "level": "error",
                           "text": f"Природниот глас не одговори ({type(e).__name__}). Следната минута зборува локалниот."})
        text, wav = self.tr.speech_to_speech(audio, tgt, sid)
        if on_audio and wav.size:
            on_audio(wav)
        return text, wav, "local"

    def preview_voice(self, tgt: str, voice: str, gender: str, speaker_id: int = -1) -> dict:
        """Кратка реченица на целниот јазик со избраниот глас."""
        if not self.tr:
            raise RuntimeError("Моделот уште не е вчитан.")
        sid = speaker_id if speaker_id >= 0 else LOCAL_VOICES.get(gender, 0)
        sample = "Good morning everyone. This is how my voice will sound in the meeting today."
        t0 = time.time()
        text, wav = self.tr.text_to_speech(sample, "eng", tgt, sid)
        used = "local"
        if voice == "natural" and tgt in NEURAL_VOICES:
            try:
                wav, used = neural_tts(text, tgt, gender), "natural"
            except Exception as e:  # noqa: BLE001
                self.emit({"type": "log", "level": "error", "text": f"Природниот глас не одговори ({type(e).__name__})."})
        return {"text": text, "wav": wav, "used": used, "compute_s": round(time.time() - t0, 2)}

    def _write(self, line: str):
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line)

    def inject(self, audio: np.ndarray, sr: int, who: str = "me"):
        """Пушта датотека во вистинско време како да доаѓа од микрофонот (или од состанокот)."""
        if not self.running:
            raise RuntimeError("Прво притисни Старт.")
        seg = self.seg_me if who == "me" else self.seg_them
        a = np.concatenate([resample(audio, sr, SR), np.zeros(SR, np.float32)])

        def run():
            step = FRAME * 4
            t0 = time.time()
            self.injecting.add(who)  # вистинскиот извор паузира додека трае пробата
            try:
                for i in range(0, len(a), step):
                    if self.stop_ev.is_set():
                        return
                    seg.feed(a[i:i + step])
                    ahead = (i + step) / SR - (time.time() - t0)
                    if ahead > 0:
                        time.sleep(ahead)
            finally:
                self.injecting.discard(who)
        threading.Thread(target=run, daemon=True).start()
        return round(len(a) / SR, 2)

    # ---- проба од датотека (без микрофон)
    def translate_file(self, audio: np.ndarray, sr: int, mode: str, tgt: str,
                       voice: str = "local", gender: str = "f") -> dict:
        if not self.tr:
            raise RuntimeError("Моделот уште не е вчитан.")
        a = resample(audio, sr, SR)
        t0 = time.time()
        used, first = None, []
        if mode == "speech":
            text, wav, used = self.speak(a, tgt, voice, gender,
                                         on_audio=lambda x: first or first.append(time.time() - t0))
        else:
            text, wav = self.tr.speech_to_text(a, tgt), None
        return {"text": text, "wav": wav, "used": used, "first_audio_s": round(first[0], 2) if first else None, "audio_s": round(len(a) / SR, 2), "compute_s": round(time.time() - t0, 2),
                "vram_peak_gb": round(self.tr.vram_peak_gb(), 2)}

    def status(self) -> dict:
        return {"model": "ready" if self.tr else ("loading" if self.loading else "none"),
                "running": self.running, "passthrough": self.passthrough,
                "settings": asdict(self.settings), "summary": self.summary(),
                "vram_gb": round(self.tr.vram_gb, 2) if self.tr else None}
