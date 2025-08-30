#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ai Realtime (OpenAI Realtime API via WebSocket)
Versione ottimizzata per Windows e Raspberry Pi:
- Microfono DISATTIVATO mentre l'AI parla
- Buffer microfono svuotato localmente e sul server
- Nessun barge-in o filtri audio complessi
- Cooldown aumentato per stabilità su Windows
- Integrazione braccio robotico Arduino invariata
- Log colorati per distinguere messaggi utente, AI e sistema
- Mostra il testo dell'AI a console mentre parla (via audio_transcript.*)
- Auto-detect della porta seriale Arduino anche con hub USB

Aggiornamenti:
- Stato connessione: OFFLINE / RECONNECTING / CONNECTED
- Riconnessione robusta con backoff e "resume" logico (token opzionale)
- HUD opzionale:
  - modalità OpenCV (se disponibile) oppure fallback console senza installare OpenCV
  - mostra rete, stato connessione, tentativi, ultimo errore, microfono
- Mic diagnostics: is_hearing() e get_last_rms()
"""

import os, ssl, json, base64, asyncio, sys, collections, threading, re, pathlib, contextlib, random, socket
import numpy as np
import sounddevice as sd
import signal
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK
from time import strftime
from dotenv import load_dotenv
from colorama import init, Fore, Style  # Colori console
from enum import Enum
from typing import Optional  # <-- aggiunto per compatibilità 3.9

# Inizializza colorama per la compatibilità cross-platform
init(autoreset=True)

# =========================
# Config base
# =========================
load_dotenv()
try:
    # Evita problemi con caratteri UTF-8 su Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL   = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime")
URL     = f"wss://api.openai.com/v1/realtime?model={MODEL}"

SCRIPT_DIR = pathlib.Path(__file__).parent
SYSTEM_PROMPT_PATH = SCRIPT_DIR / "ai_prompt.txt"

# Prompt di default se non troviamo ai_prompt.txt accanto allo script
SYSTEM_PROMPT_DEFAULT = """
Sei Diego, un assistente vocale dotato di un braccio robotico a 5 gradi di libertà (inclusa una pinza).
Puoi muovere il tuo braccio per eseguire azioni fisiche come salutare, afferrare oggetti, rilasciarli o disegnare forme semplici.
Quando l’utente ti chiede di fare qualcosa che puoi realizzare con il braccio, conferma l’azione e procedi a eseguirla.
Rispondi con massimo 15 parole. Parla SOLO in italiano; se l'utente parla chiaramente in inglese, rispondi in inglese.
Mantieni un tono amichevole e collaborativo.
""".strip()

# =========================
# Parametri audio
# =========================
SAMPLE_RATE        = int(os.environ.get("SAMPLE_RATE", "24000"))
CHUNK_MS           = int(os.environ.get("CHUNK_MS", "20"))  # dimensione chunk microfono (ms)
SAMPLES_PER_CHUNK  = max(1, SAMPLE_RATE * CHUNK_MS // 1000)

# VAD server-side (semplificato, senza barge-in)
VAD_THRESHOLD   = float(os.environ.get("VAD_THRESHOLD", "0.2"))
VAD_SILENCE_MS  = int(os.environ.get("VAD_SILENCE_MS", "800"))
VAD_PREFIX_MS   = int(os.environ.get("VAD_PREFIX_MS", "500"))

# Filtro minimo per stampare la trascrizione utente
MIN_USER_CHARS  = int(os.environ.get("MIN_USER_CHARS", "6"))

# Buffer massimo per l’uscita audio TTS (secondi)
OUTPUT_MAX_BUFFER_SEC = int(os.environ.get("OUTPUT_MAX_BUFFER_SEC", "20"))

# Ritardo tra fine TTS e riattivazione microfono (ms)
LISTEN_COOLDOWN_MS = int(os.environ.get("LISTEN_COOLDOWN_MS", "1000"))

# Voce TTS e dispositivi audio (opzionali)
VOICE      = os.environ.get("OPENAI_VOICE", "cedar")
MIC_DEVICE = os.environ.get("MIC_DEVICE")
SPK_DEVICE = os.environ.get("SPK_DEVICE")

# =========================
# Riconnessione / timeouts
# =========================
RECONNECT_MAX_BACKOFF = int(os.environ.get("RECONNECT_MAX_BACKOFF", "30"))
SEND_TIMEOUT = float(os.environ.get("SEND_TIMEOUT", "5.0"))
RECV_TIMEOUT = float(os.environ.get("RECV_TIMEOUT", "150.0"))
PING_INTERVAL = float(os.environ.get("PING_INTERVAL", "20.0"))
PING_TIMEOUT  = float(os.environ.get("PING_TIMEOUT",  "20.0"))
OPEN_TIMEOUT  = float(os.environ.get("OPEN_TIMEOUT",  "20.0"))
CLOSE_TIMEOUT = float(os.environ.get("CLOSE_TIMEOUT", "5.0"))

# =========================
# Memoria conversazione locale
# =========================
PERSIST_MEMORY = int(os.environ.get("PERSIST_MEMORY", "1"))
MEMORY_TURNS   = int(os.environ.get("MEMORY_TURNS", "8"))
HISTORY = collections.deque(maxlen=MEMORY_TURNS)

# =========================
# Log conversazione su Desktop/Home (persistente)
# =========================
def _get_desktop_dir() -> pathlib.Path:
    """
    Ritorna il percorso della cartella Desktop se esiste, altrimenti la home.
    Gestisce anche 'Scrivania' per alcune localizzazioni.
    """
    home = pathlib.Path.home()
    candidates = [home / "Desktop", home / "Scrivania"]
    for c in candidates:
        with contextlib.suppress(Exception):
            if c.exists():
                return c
    return home

# Migrazione automatica: rinomina l'eventuale vecchia cartella "Diego_Conversazioni" in "AI_Conversazioni"
_old_dir = _get_desktop_dir() / "Diego_Conversazioni"
_new_dir = _get_desktop_dir() / "AI_Conversazioni"
if _old_dir.exists() and not _new_dir.exists():
    with contextlib.suppress(Exception):
        os.rename(str(_old_dir), str(_new_dir))

_LOG_DIR = _new_dir
with contextlib.suppress(Exception):
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

def _log_file_path() -> pathlib.Path:
    # File giornaliero
    return _LOG_DIR / f"conversazioni_{strftime('%Y-%m-%d')}.txt"

def append_log(role: str, text: str):
    """
    Appende una riga di log con timestamp, ruolo e testo.
    Forza flush + fsync per minimizzare perdita dati in caso di spegnimento improvviso.
    """
    try:
        p = _log_file_path()
        # Apri, scrivi, flush, fsync, chiudi ad ogni chiamata
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"{strftime('[%Y-%m-%d %H:%M:%S]')} {role}: {text}\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        # Evita dipendenza da note(), non ancora definita qui
        print(f"{strftime('[%H:%M:%S]')} [LOG WARNING]: impossibile scrivere il log: {e}", flush=True)

# =========================
# Utility console
# =========================
def say_user(text):
    """Stampa il testo dell’utente in verde."""
    print(f"{strftime('[%H:%M:%S]')} {Fore.GREEN}TU: {text}{Style.RESET_ALL}", flush=True)
    append_log("TU", text)

def say_bot(text):
    """Stampa il testo dell’assistente in ciano."""
    print(f"{strftime('[%H:%M:%S]')} {Fore.CYAN}AI: {text}{Style.RESET_ALL}", flush=True)
    append_log("AI", text)

def note(msg):
    """Messaggi informativi gialli (stato sistema)."""
    print(f"{strftime('[%H:%M:%S]')} {Fore.YELLOW}[INFO]: {msg}{Style.RESET_ALL}", flush=True)

def warn(msg):
    print(f"{strftime('[%H:%M:%S]')} {Fore.MAGENTA}[ATTENZIONE]: {msg}{Style.RESET_ALL}", flush=True)

def validate_device(device, kind="input"):
    """Consente di usare nome o indice del device audio; ritorna l’indice o None."""
    if device is None: return None
    try:
        devices = sd.query_devices()
        if isinstance(device, int) and 0 <= device < len(devices):
            return device
        elif isinstance(device, str):
            want = device.strip().lower()
            # Match esatto
            for i, d in enumerate(devices):
                if d.get("name") == device:
                    return i
            # Match parziale case-insensitive
            for i, d in enumerate(devices):
                if want in (d.get("name", "").lower()):
                    return i
    except Exception as e:
        note(f"Errore validazione device {kind}: {e}")
    return None

# =========================
# MIC grezzo (con lock + VAD locale opzionale) + diagnostica
# =========================
class MicStreamer:
    """
    Cattura audio mono float32, converte a PCM16 LE e mette in coda.
    - enable()/disable(): attiva/disattiva la cattura
    - get_chunk(): ritorna un chunk (bytes) pronto da inviare
    - is_hearing(): indica se sta rilevando voce/energia
    - get_last_rms(): valore energia normalizzata per HUD
    """
    def __init__(self, device=None, max_chunks=3000):
        self.device = device
        self._lock = threading.Lock()
        self.buf = collections.deque(maxlen=max_chunks)
        self.stream = None
        self.enabled = False
        self.stopping = False

        # VAD locale opzionale (pre-filtra i chunk silenti)
        self.use_local_vad = bool(int(os.environ.get("OPENAI_LOCAL_VAD", "0")))
        self._ema_bg = 1e-6
        self._vad_on = False
        self._vad_state_chunks = 0
        self._vad_alpha = float(os.environ.get("LOCAL_VAD_ALPHA", "0.2"))
        self._thr_on = float(os.environ.get("LOCAL_VAD_THR_ON", "0.65"))
        self._thr_off = float(os.environ.get("LOCAL_VAD_THR_OFF", "0.45"))
        self._min_voice_chunks = max(1, int(int(os.environ.get("LOCAL_VAD_MIN_VOICE_MS", "120")) / CHUNK_MS))
        self._min_sil_chunks   = max(1, int(int(os.environ.get("LOCAL_VAD_MIN_SIL_MS", "250")) / CHUNK_MS))
        self._last_rms = 0.0

    def _energy_ratio(self, mono_f32: np.ndarray) -> float:
        # RMS normalizzato da uno sfondo stimato con EMA
        e = float(np.mean(mono_f32.astype(np.float32) ** 2))
        self._ema_bg = (1 - self._vad_alpha) * self._ema_bg + self._vad_alpha * e
        return e / max(self._ema_bg, 1e-9)

    def _vad_step(self, r: float) -> bool:
        if not self._vad_on:
            if r >= self._thr_on:
                self._vad_state_chunks += 1
                if self._vad_state_chunks >= self._min_voice_chunks:
                    self._vad_on = True
                    self._vad_state_chunks = 0
            else:
                self._vad_state_chunks = 0
        else:
            if r <= self._thr_off:
                self._vad_state_chunks += 1
                if self._vad_state_chunks >= self._min_sil_chunks:
                    self._vad_on = False
                    self._vad_state_chunks = 0
            else:
                self._vad_state_chunks = 0
        return self._vad_on

    def _callback(self, indata, frames, time_info, status):
        with self._lock:
            if not self.enabled or self.stopping:
                return
        mono = np.mean(indata, axis=1) if indata.shape[1] > 1 else indata[:, 0]
        # Aggiorna energia per HUD/diagnostica
        try:
            self._last_rms = float(self._energy_ratio(mono))
        except Exception:
            pass
        push = True
        if self.use_local_vad:
            r = self._energy_ratio(mono)
            push = self._vad_step(r)
        if push:
            pcm16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            with self._lock:
                if not self.stopping:
                    self.buf.append(pcm16)

    async def start(self):
        self.stream = sd.InputStream(
            device=self.device, channels=1, samplerate=SAMPLE_RATE,
            blocksize=SAMPLES_PER_CHUNK, dtype='float32', callback=self._callback
        )
        self.stream.start()
        with self._lock:
            self.enabled = True
            self.stopping = False

    async def get_chunk(self) -> bytes:
        while True:
            with self._lock:
                if self.stopping:
                    raise asyncio.CancelledError()
                if self.buf:
                    return self.buf.popleft()
            await asyncio.sleep(0.001)

    def clear(self):
        with self._lock:
            self.buf.clear()

    def set_enabled(self, val: bool):
        with self._lock:
            self.enabled = bool(val)
            if not self.enabled:
                self.buf.clear()
                # reset stato VAD per ripartenze pulite
                self._vad_on = False
                self._vad_state_chunks = 0
                self._ema_bg = 1e-6

    def is_hearing(self) -> bool:
        """
        True se il VAD è on (quando attivo) oppure l'energia supera una soglia.
        Utile per HUD e diagnosi microfono.
        """
        try:
            if self.use_local_vad:
                return bool(self._vad_on)
            return self._last_rms > 1.2  # soglia euristica
        except Exception:
            return False

    def get_last_rms(self) -> float:
        return float(self._last_rms)

    async def stop(self):
        with self._lock:
            self.stopping = True
        if self.stream is not None:
            try:
                self.stream.stop(); self.stream.close()
            finally:
                self.stream = None

# =========================
# SPEAKER ring-buffer (riproduzione PCM16)
# =========================
class Speaker:
    """
    Riproduce audio TTS (PCM16) con un piccolo ring-buffer.
    - enqueue(): aggiunge bytes PCM da suonare
    - has_audio(): indica se c’è ancora audio in coda
    """
    def __init__(self, device=None):
        self.muted = False
        self.lock = threading.Lock()
        self.buffer = bytearray()
        self.max_bytes = OUTPUT_MAX_BUFFER_SEC * SAMPLE_RATE * 2  # int16 mono
        self.stream = sd.OutputStream(
            device=device, channels=1, samplerate=SAMPLE_RATE,
            dtype='int16', blocksize=SAMPLES_PER_CHUNK, callback=self._callback
        )

    def start(self): self.stream.start()

    def _callback(self, outdata, frames, time_info, status):
        needed = frames * 2
        with self.lock:
            if self.muted or len(self.buffer) == 0:
                outdata[:] = np.zeros((frames, 1), dtype=np.int16)
                return
            if len(self.buffer) >= needed:
                chunk = self.buffer[:needed]; del self.buffer[:needed]
            else:
                chunk = self.buffer[:]; self.buffer.clear()
                chunk += b'\x00' * (needed - len(chunk))
        outdata[:] = np.frombuffer(chunk, dtype=np.int16).reshape(-1, 1)

    def enqueue(self, pcm_bytes: bytes):
        if not pcm_bytes: return
        with self.lock:
            if len(self.buffer) + len(pcm_bytes) > self.max_bytes:
                overflow = (len(self.buffer) + len(pcm_bytes)) - self.max_bytes
                if overflow > 0: del self.buffer[:overflow]
            self.buffer.extend(pcm_bytes)

    def flush(self):
        with self.lock: self.buffer.clear()

    def has_audio(self) -> bool:
        with self.lock: return len(self.buffer) > 0

    def set_muted(self, m: bool):
        with self.lock:
            self.muted = bool(m)
            if self.muted: self.buffer.clear()

    def stop(self):
        try:
            self.stream.stop(); self.stream.close()
        except Exception:
            pass

# =========================
# Prompt & session
# =========================
def load_system_prompt() -> str:
    """Carica il prompt da file se presente, altrimenti usa quello di default."""
    try:
        if SYSTEM_PROMPT_PATH.exists():
            return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except Exception as e:
        note(f"Errore lettura file prompt: {e}")
    return SYSTEM_PROMPT_DEFAULT

def build_session_payload(resume_token: Optional[str] = None) -> dict:
    """
    Configura la sessione Realtime:
    - modalities audio+text (così possiamo anche visualizzare)
    - trascrizione server-side per input
    - VAD server-side (senza barge-in)
    - supporto opzionale a resume_token, se il backend lo gestisce
    """
    instructions = load_system_prompt()
    note("Istruzioni di sessione attive:")
    note(instructions)
    session = {
        "instructions": instructions,
        "modalities": ["audio", "text"],
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "voice": VOICE,
        "input_audio_transcription": {"model": "gpt-4o-transcribe", "language": "it"},
        "turn_detection": {
            "type": "server_vad",
            "threshold": VAD_THRESHOLD,
            "prefix_padding_ms": VAD_PREFIX_MS,
            "silence_duration_ms": VAD_SILENCE_MS,
            "create_response": False
        }
    }
    if resume_token:
        session["resume"] = True
        session["resume_token"] = resume_token
        note("Tentativo di resume della sessione (token locale).")
    return {"type": "session.update", "session": session}

async def seed_history(ws):
    """(Opzionale) reinserisce la storia locale nella conversazione del server."""
    if not PERSIST_MEMORY or not HISTORY: return
    note(f"Ripristino memoria locale: {len(HISTORY)} messaggi")
    for role, text in list(HISTORY):
        payload = {
            "type":"conversation.item.create",
            "item":{"type":"message","role":role,"content":[{"type":"input_text","text":text}]}
        }
        ok = await safe_send(ws, payload)
        if not ok: break

# =========================
# Invio con timeout (con gestione eccezioni + close)
# =========================
async def safe_send(ws, payload: dict) -> bool:
    try:
        await asyncio.wait_for(ws.send(json.dumps(payload)), timeout=SEND_TIMEOUT)
        return True
    except asyncio.TimeoutError:
        note("Timeout invio su WebSocket. Forzo riconnessione.")
    except (ConnectionClosed, ConnectionClosedOK, ConnectionClosedError):
        pass
    except Exception as e:
        note(f"Errore send: {e}")
    with contextlib.suppress(Exception):
        await ws.close(code=1011, reason="send timeout")
    return False

# =========================
# === ROBOT: Bridge seriale + intent matcher ===
# =========================
try:
    import serial
    from serial.tools import list_ports
except Exception:
    serial = None
    list_ports = None

def print_serial_table(ports):
    """Stampa una tabella leggibile delle porte seriali trovate."""
    if not ports:
        warn("Nessuna porta seriale trovata.")
        return
    print(f"{strftime('[%H:%M:%S]')} {Fore.YELLOW}Porte seriali disponibili:{Style.RESET_ALL}")
    for p in ports:
        desc = [
            f"device={p.device}",
            f"manufacturer={getattr(p,'manufacturer',None) or '-'}",
            f"product={getattr(p,'product',None) or '-'}",
            f"serial={getattr(p,'serial_number',None) or '-'}",
        ]
        try:
            if p.vid is not None and p.pid is not None:
                desc.append(f"vid=0x{p.vid:04X}")
                desc.append(f"pid=0x{p.pid:04X}")
        except Exception:
            pass
        note("  - " + " | ".join(desc))

def match_arduino_candidate(p, want_vendor=None, want_product=None, want_serial=None):
    """Ritorna True se la porta 'p' sembra il nostro Arduino (manufacturer/product, VID/PID o serial)."""
    man = (getattr(p, "manufacturer", "") or "").lower()
    prod = (getattr(p, "product", "") or "").lower()
    sn   = getattr(p, "serial_number", None)

    # Filtri espliciti
    if want_serial and sn and want_serial.strip() == sn.strip():
        return True
    if want_vendor and want_vendor.lower() not in man:
        return False
    if want_product and want_product.lower() not in prod:
        return False

    # Vendor tipici
    known_makers = ("arduino", "genuino", "silicon labs", "wch.cn", "wch", "ftdi", "teensy")
    if any(k in man for k in known_makers) or any(k in prod for k in known_makers):
        return True

    # VID/PID tipici
    try:
        vid = p.vid
        pid = p.pid
        if vid is None or pid is None:
            return False
        known_pairs = {
            (0x2341, 0x0043),  # Arduino/Genuino Uno
            (0x2341, 0x0001),  # Arduino Uno (old)
            (0x2341, 0x0243),  # Arduino Uno Rev3
            (0x2A03, 0x0043),  # Arduino (alt VID)
            (0x1A86, 0x7523),  # CH340
            (0x1A86, 0x5523),  # CH341
            (0x10C4, 0xEA60),  # CP210x
            (0x0403, 0x6001),  # FT232
            (0x16C0, 0x0483),  # Teensy
        }
        if (vid, pid) in known_pairs:
            return True
    except Exception:
        pass

    return False

def find_arduino_port():
    """
    Restituisce il path della porta seriale dell'Arduino da usare.
    Priorità:
      1) ROBOT_PORT se valorizzato e presente
      2) Scansione automatica (match manufacturer/product/VID-PID/serial)
    """
    if serial is None or list_ports is None:
        warn("[RobotLink] pyserial non disponibile; impossibile auto-rilevare la porta.")
        return None

    want_port   = os.environ.get("ROBOT_PORT", "auto").strip()
    want_vendor = os.environ.get("ROBOT_VENDOR", "").strip() or None
    want_prod   = os.environ.get("ROBOT_PRODUCT", "").strip() or None
    want_sn     = os.environ.get("ROBOT_SERIAL", "").strip() or None

    ports = list(list_ports.comports())
    print_serial_table(ports)

    # 1) Se ROBOT_PORT è un device specifico e lo troviamo, usalo
    if want_port and want_port != "auto":
        for p in ports:
            if p.device == want_port:
                note(f"[RobotLink] Uso porta specificata: {want_port}")
                return want_port
        warn(f"[RobotLink] Porta specificata '{want_port}' non trovata. Procedo con auto-detect.")

    # 2) Scansione per match
    #    preferisci /dev/ttyACM* rispetto a /dev/ttyUSB* quando entrambi trovati
    acm = []
    usb = []
    for p in ports:
        if match_arduino_candidate(p, want_vendor, want_prod, want_sn):
            if p.device.startswith("/dev/ttyACM"):
                acm.append(p.device)
            elif p.device.startswith("/dev/ttyUSB"):
                usb.append(p.device)

    if acm:
        chosen = sorted(acm)[0]
        note(f"[RobotLink] Trovato Arduino su {chosen} (preferenza ttyACM*)")
        return chosen
    if usb:
        chosen = sorted(usb)[0]
        note(f"[RobotLink] Trovato Arduino su {chosen} (ttyUSB*)")
        return chosen

    # 3) fallback: se c'è un'unica seriale, usa quella
    if len(ports) == 1:
        note(f"[RobotLink] Una sola seriale trovata, uso {ports[0].device}")
        return ports[0].device

    warn("[RobotLink] Nessuna porta compatibile trovata.")
    return None

class RobotLink:
    """Bridge verso Arduino via USB-seriale (protocollo JSONL)."""
    def __init__(self, port=None, baud=115200, timeout=2.0):
        self.port = port
        self.enabled = False
        self.ser = None
        if serial is None:
            warn("[RobotLink] pyserial non disponibile; bridge disabilitato")
            return
        # Autodetect se non specificata
        if not self.port or self.port.strip().lower() == "auto":
            self.port = find_arduino_port()

        if not self.port:
            warn("[RobotLink] Porta non determinata. Disabilito il bridge.")
            return

        try:
            self.ser = serial.Serial(self.port, baudrate=baud, timeout=timeout)
            import time as _t
            _t.sleep(2.0)  # reset Arduino
            self._drain()
            self.enabled = True
            note(f"[RobotLink] Connesso a {self.port}")
        except Exception as e:
            warn(f"[RobotLink] Impossibile aprire {self.port}: {e}")

    def _drain(self):
        if not self.ser: return
        try:
            while True:
                line = self.ser.readline()
                if not line: break
        except Exception: pass

    def _send(self, payload: dict, wait_done=True, done_timeout=30.0):
        if not self.enabled: return {"status":"error","msg":"robot link disabled"}
        try:
            msg = json.dumps(payload) + "\n"
            self.ser.write(msg.encode("utf-8")); self.ser.flush()
            if not wait_done: return {"status":"sent"}
            import time as _t
            t0 = _t.monotonic(); last = None
            while _t.monotonic() - t0 < done_timeout:
                line = self.ser.readline().decode("utf-8","replace").strip()
                if not line: continue
                try: data = json.loads(line)
                except Exception: continue
                st = data.get("status"); last = st
                if st in ("done","error"): return data
            return {"status":"error","msg":f"timeout waiting done (last={last})"}
        except Exception as e:
            return {"status":"error","msg":str(e)}

    # Alcuni comandi standard (aggancia il tuo firmware Arduino)
    def home(self): return self._send({"cmd":"HOME"})
    def wave(self): return self._send({"cmd":"WAVE"})
    def open(self): return self._send({"cmd":"OPEN"})
    def close(self): return self._send({"cmd":"CLOSE"})
    def move(self, dir_="RIGHT"): return self._send({"cmd":"MOVE","dir":dir_.upper()})
    def pick_place(self, f="table", t="bin"): return self._send({"cmd":"PICK_PLACE","from":f,"to":t})
    def draw(self, ch="A"): return self._send({"cmd":"DRAW","char":ch})
    def circle(self): return self._send({"cmd":"CIRCLE"})
    def stop(self): return self._send({"cmd":"STOP"}, wait_done=False)

# Regex per riconoscere comandi vocali base
KW = {
    "HOME": re.compile(r"\b(home|casa|posizione di casa|vai a casa|riposa)\b", re.I),
    "WAVE": re.compile(r"\b(saluta|fai ciao|ciao con la mano)\b", re.I),
    "OPEN": re.compile(r"\b(rilascia|apri la pinza|lascia l'oggetto|rilascia l'oggetto)\b", re.I),
    "CLOSE":re.compile(r"\b(prendi|afferra|chiudi la pinza|stringi)\b", re.I),
    "STOP": re.compile(r"\b(stop|fermati|emergenza|basta)\b", re.I),
    "LEFT": re.compile(r"\b(sinistra|a sinistra)\b", re.I),
    "RIGHT":re.compile(r"\b(destra|a destra)\b", re.I),
    "UP":   re.compile(r"\b(alto|su|braccio alto)\b", re.I),
    "DOWN": re.compile(r"\b(basso|giù|braccio basso)\b", re.I),
    "MOVE_PHRASE": re.compile(r"\b(sposta|porta|muovi|metti)\b", re.I),
    "CIRCLE": re.compile(r"\b(cerchio|fai un cerchio)\b", re.I),
    "DRAW": re.compile(r"\b(disegna|scrivi)\b", re.I),
    "LETTER": re.compile(r"\b([A-Za-z])\b")
}

_robot = None

def _run_robot(fn, say_cb):
    """Esegue il comando su thread separato e fornisce un breve feedback a voce/testo."""
    res = fn()
    if say_cb:
        if res.get("status") == "error": say_cb("Comando robot fallito.")
        else: say_cb("Fatto.")

def process_voice_command_robot(text: str, say_cb=None) -> bool:
    """
    Semplice intent matcher basato su regex italiane.
    Se trova un comando, lo esegue e ritorna True (così evitiamo una risposta AI inutile).
    """
    global _robot
    if not _robot or not getattr(_robot, "enabled", False): return False
    t = text.lower()

    if KW["STOP"].search(t):
        threading.Thread(target=_run_robot, args=(_robot.stop, say_cb), daemon=True).start(); return True
    if KW["HOME"].search(t):
        threading.Thread(target=_run_robot, args=(_robot.home, say_cb), daemon=True).start(); return True
    if KW["WAVE"].search(t):
        threading.Thread(target=_run_robot, args=(_robot.wave, say_cb), daemon=True).start(); return True
    if KW["OPEN"].search(t):
        threading.Thread(target=_run_robot, args=(_robot.open, say_cb), daemon=True).start(); return True
    if KW["CLOSE"].search(t):
        threading.Thread(target=_run_robot, args=(_robot.close, say_cb), daemon=True).start(); return True

    if KW["MOVE_PHRASE"].search(t) or KW["LEFT"].search(t) or KW["RIGHT"].search(t) or KW["UP"].search(t) or KW["DOWN"].search(t):
        if KW["LEFT"].search(t):  threading.Thread(target=_run_robot, args=(lambda: _robot.move("LEFT"), say_cb), daemon=True).start();  return True
        if KW["RIGHT"].search(t): threading.Thread(target=_run_robot, args=(lambda: _robot.move("RIGHT"), say_cb), daemon=True).start(); return True
        if KW["UP"].search(t):    threading.Thread(target=_run_robot, args=(lambda: _robot.move("UP"), say_cb), daemon=True).start();    return True
        if KW["DOWN"].search(t):  threading.Thread(target=_run_robot, args=(lambda: _robot.move("DOWN"), say_cb), daemon=True).start();  return True

    if KW["CIRCLE"].search(t):
        threading.Thread(target=_run_robot, args=(_robot.circle, say_cb), daemon=True).start(); return True

    if KW["DRAW"].search(t):
        m = KW["LETTER"].search(text); ch = m.group(1).upper() if m else "A"
        threading.Thread(target=_run_robot, args=(lambda: _robot.draw(ch), say_cb), daemon=True).start(); return True

    if KW["CLOSE"].search(t) and KW["MOVE_PHRASE"].search(t):
        threading.Thread(target=_run_robot, args=(lambda: _robot.pick_place("table","bin"), say_cb), daemon=True).start(); return True

    return False

# =========================
# Stato connessione + HUD diagnostico (cv o console)
# =========================
class ConnState(Enum):
    OFFLINE = "offline"
    RECONNECTING = "reconnecting"
    CONNECTED = "connected"

class StatusHUD:
    """
    HUD diagnostico:
    - Modalità OpenCV (finestra grafica) se disponibile e abilitata
    - Fallback console (stampa periodica) se OpenCV non è disponibile o HUD_MODE=console
    Abilita con HUD=1 nell'ambiente.
    HUD_MODE: auto|cv|console (default: auto)
    """
    def __init__(self, enabled: Optional[bool] = None, window_name: str = "Stato Realtime"):
        self.enabled = enabled if enabled is not None else (os.getenv("HUD", "0") == "1")
        self.window = window_name
        self.mode = os.getenv("HUD_MODE", "auto").lower()
        self._last_print_blob = ""
        self._last_print_ts = 0.0
        self._print_interval = 0.5  # secondi

        if not self.enabled:
            note("HUD disattivato (setta HUD=1 per abilitarlo).")
            self.mode = "off"
            return

        if self.mode not in ("auto", "cv", "console"):
            self.mode = "auto"

        # Prova CV se richiesto/consentito
        self._cv_ok = False
        if self.mode in ("auto", "cv"):
            # Controllo ambiente grafico (Linux headless)
            if sys.platform.startswith("linux") and not os.getenv("DISPLAY"):
                if self.mode == "cv":
                    warn("HUD_MODE=cv ma DISPLAY non presente. Fallback a console.")
                self.mode = "console"
            else:
                try:
                    import cv2
                    cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
                    with contextlib.suppress(Exception):
                        cv2.moveWindow(self.window, 40, 40)
                    self._cv_ok = True
                    self.mode = "cv"
                    note("HUD attivo (modalità OpenCV).")
                except Exception as e:
                    if os.getenv("HUD_MODE", "auto").lower() == "cv":
                        warn(f"HUD_MODE=cv richiesto ma OpenCV GUI non disponibile: {e}. Fallback a console.")
                    self.mode = "console"

        if self.mode == "console":
            note("HUD attivo (modalità console).")

    def show(self, lines: list[str]):
        if not self.enabled or self.mode == "off":
            return

        if self.mode == "cv" and self._cv_ok:
            import numpy as np
            import cv2
            h, w = 240, 740
            img = np.zeros((h, w, 3), dtype=np.uint8)
            y = 28
            for line in lines:
                cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                y += 24
            cv2.imshow(self.window, img)
            cv2.waitKey(1)
            return

        # Modalità console: stampa solo se cambia o dopo un certo intervallo
        import time as _t
        now = _t.monotonic()
        blob = " | ".join(lines)
        if blob != self._last_print_blob or (now - self._last_print_ts) >= self._print_interval:
            print(f"{strftime('[%H:%M:%S]')} {Fore.YELLOW}[HUD]{Style.RESET_ALL} {blob}", flush=True)
            self._last_print_blob = blob
            self._last_print_ts = now

    def close(self):
        if not self.enabled or self.mode != "cv":
            return
        import cv2
        with contextlib.suppress(Exception):
            cv2.destroyWindow(self.window)

# =========================
# Core Realtime (mic OFF mentre parla l'AI)
# =========================
async def run_realtime(ws, mic: MicStreamer, spk: Speaker, stop_event: asyncio.Event):
    """
    Due task:
      - sender(): legge chunk dal mic e li manda al server quando il mic è attivo
      - receiver(): gestisce gli eventi del server (audio TTS, testo, fine risposta, errori)
    Durante il TTS, il mic viene disabilitato per evitare eco/loop.
    """
    ai_speaking = False

    # Buffer testuale per i transcript audio dell'AI (key = item_id)
    transcripts = {}
    printed_ids = set()

    async def sender():
        try:
            while not stop_event.is_set():
                if not mic.enabled:
                    await asyncio.sleep(0.01)
                    continue
                try:
                    chunk = await mic.get_chunk()
                except asyncio.CancelledError:
                    break
                if stop_event.is_set() or getattr(ws, "closed", False):
                    break
                ok = await safe_send(ws, {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii")
                })
                if not ok:
                    break
        finally:
            pass

    async def receiver():
        nonlocal ai_speaking, transcripts, printed_ids

        try:
            while not stop_event.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)
                except asyncio.TimeoutError:
                    note("Idle prolungata: nessun dato.")
                    continue
                except (ConnectionClosed, ConnectionClosedOK, ConnectionClosedError):
                    break
                except Exception as e:
                    note(f"Errore recv: {e}")
                    break

                # Gli eventi Realtime sono JSON
                try:
                    data = json.loads(msg)
                except Exception:
                    continue

                t = data.get("type")

                # --- Utente: trascrizione input audio completata (server-side) ---
                if t == "conversation.item.input_audio_transcription.completed":
                    user_text = (data.get("transcript", "") or "").strip()
                    if user_text and len(user_text) >= MIN_USER_CHARS:
                        say_user(user_text)
                        # Intent matcher per il braccio robotico (opzionale)
                        try:
                            handled = process_voice_command_robot(user_text, say_cb=say_bot)
                            if handled:
                                note("Comando robotico intercettato.")
                        except Exception as e:
                            note(f"Errore intent robot: {e}")

                        # Memoria locale
                        if PERSIST_MEMORY:
                            HISTORY.append(("user", user_text))

                        # Inoltra come messaggio testuale e crea risposta
                        if not await safe_send(ws, {
                            "type": "conversation.item.create",
                            "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": user_text}]}
                        }):
                            break
                        if not await safe_send(ws, {"type": "response.create"}):
                            break

                # ====== AI: testo dal canale di trascrizione dell'audio TTS ======
                elif t == "response.audio_transcript.delta":
                    item_id = data.get("item_id") or data.get("response", {}).get("item_id") or "_last_"
                    transcripts[item_id] = transcripts.get(item_id, "") + (data.get("delta") or "")

                elif t == "response.audio_transcript.done":
                    item_id = data.get("item_id") or "_last_"
                    text = (transcripts.get(item_id, "") or "").strip()
                    if text and item_id not in printed_ids:
                        say_bot(text)
                        printed_ids.add(item_id)
                        if PERSIST_MEMORY:
                            HISTORY.append(("assistant", text))

                # ====== (Extra) altre varianti testuali opzionali ======
                elif t in ("response.output_text.delta", "response.text.delta", "conversation.item.output_text.delta"):
                    transcripts["_textstream_"] = transcripts.get("_textstream_", "") + (data.get("delta") or data.get("text") or "")

                elif t in ("response.output_text.done", "response.text.done"):
                    txt = (transcripts.get("_textstream_", "") or "").strip()
                    if txt and "_textstream_" not in printed_ids:
                        say_bot(txt)
                        printed_ids.add("_textstream_")
                        if PERSIST_MEMORY:
                            HISTORY.append(("assistant", txt))

                # --- AI: audio TTS ---
                elif t == "response.audio.delta":
                    if not ai_speaking:
                        ai_speaking = True
                        mic.set_enabled(False)
                        mic.clear()
                        await safe_send(ws, {"type": "input_audio_buffer.clear"})
                        note("Microfono disattivato (AI parla)")
                    payload = data.get("delta") or data.get("audio")
                    if payload:
                        with contextlib.suppress(Exception):
                            spk.enqueue(base64.b64decode(payload))

                # --- Fine risposta ---
                elif t in ("response.done", "response.completed"):
                    for iid, txt in list(transcripts.items()):
                        if txt.strip() and iid not in printed_ids:
                            say_bot(txt.strip())
                            printed_ids.add(iid)
                            if PERSIST_MEMORY:
                                HISTORY.append(("assistant", txt.strip()))
                    transcripts = {}
                    printed_ids = set()

                    if ai_speaking:
                        ai_speaking = False
                        while spk.has_audio() and not stop_event.is_set():
                            await asyncio.sleep(0.1)
                        if not stop_event.is_set():
                            await asyncio.sleep(LISTEN_COOLDOWN_MS / 1000.0)
                            mic.clear()
                            mic.set_enabled(True)
                            await safe_send(ws, {"type": "input_audio_buffer.clear"})
                            note("Microfono riattivato")

                elif t == "error":
                    note(f"Errore API: {data.get('error', 'Sconosciuto')}")

        except asyncio.CancelledError:
            pass

    await asyncio.gather(sender(), receiver(), return_exceptions=True)

# =========================
# Helpers rete / connessione
# =========================
def network_ok(host="api.openai.com", port=443, timeout=2.0) -> bool:
    """Piccolo health-check: porta 443 raggiungibile?"""
    try:
        with socket.create_connection((host, port), timeout=timeout): return True
    except OSError: return False

async def connect_once(mic: MicStreamer, spk: Speaker, stop_event: asyncio.Event, resume_token: Optional[str] = None):
    """Apre una singola sessione WS Realtime e avvia la gestione IO (supporto resume_token opzionale)."""
    ssl_ctx = ssl.create_default_context()
    headers = [("Authorization", f"Bearer {API_KEY}"), ("OpenAI-Beta", "realtime=v1")]

    spk.set_muted(False); spk.flush(); mic.clear(); mic.set_enabled(True)

    async with websockets.connect(
        URL, additional_headers=headers, ssl=ssl_ctx,
        max_size=32*1024*1024, ping_interval=PING_INTERVAL, ping_timeout=PING_TIMEOUT,
        open_timeout=OPEN_TIMEOUT, close_timeout=CLOSE_TIMEOUT
    ) as ws:
        note("Connesso al Realtime.")
        if not await safe_send(ws, build_session_payload(resume_token=resume_token)): return
        await seed_history(ws)
        # Se il server emette un token/ID di sessione per resume, questo è il punto
        # dove potresti intercettarlo da un messaggio iniziale e conservarlo.
        await run_realtime(ws, mic, spk, stop_event)

# =========================
# Avvio con loop di auto-reconnect + HUD
# =========================
async def main():
    if not API_KEY:
        print("ERRORE: OPENAI_API_KEY mancante"); sys.exit(1)

    mic = MicStreamer(device=validate_device(MIC_DEVICE, "input"))
    spk = Speaker(device=validate_device(SPK_DEVICE, "output"))

    stop_event = asyncio.Event()
    def handle_signal(signum, frame):
        note(f"Segnale {signum} ricevuto: chiusura..."); stop_event.set()
    signal.signal(signal.SIGINT, handle_signal); signal.signal(signal.SIGTERM, handle_signal)

    # Bridge verso l’Arduino (auto-detect porta di default)
    global _robot
    robot_port_env = os.environ.get("ROBOT_PORT", "auto")
    _robot = RobotLink(port=robot_port_env)

    await mic.start(); spk.start()

    # --- HUD e stato connessione ---
    hud_enabled = (os.getenv("HUD", "0") == "1")
    hud_mode_env = os.getenv("HUD_MODE", "auto")
    note(f"Impostazione HUD: {'abilitato' if hud_enabled else 'disabilitato'} (HUD={os.getenv('HUD', '0')}, HUD_MODE={hud_mode_env})")
    hud = StatusHUD(enabled=hud_enabled)
    state = ConnState.OFFLINE
    last_error = ""
    attempts = 0
    resume_token: Optional[str] = None  # Popola se/come il server supporta un vero resume

    async def hud_loop():
        while not stop_event.is_set():
            lines = [
                f"Rete: {'OK' if network_ok() else 'KO'}",
                f"Connessione: {state.value}",
                f"Tentativi: {attempts}",
                f"Ultimo errore: {last_error or '-'}",
            ]
            try:
                hearing = mic.is_hearing()
                rms = mic.get_last_rms()
                lines.append(f"Mic: {'VOCE' if hearing else 'silenzio'} (rms={rms:.2f})")
                lines.append(f"Mic enabled: {mic.enabled}")
            except Exception:
                lines.append("Mic: n/d")
            hud.show(lines)
            await asyncio.sleep(0.25)

    hud_task = asyncio.create_task(hud_loop())

    backoff = 1
    try:
        while not stop_event.is_set():
            if not network_ok():
                state = ConnState.OFFLINE
                note("Rete non disponibile. Attendo che torni...")
                while not stop_event.is_set() and not network_ok(): await asyncio.sleep(1.5)
            if stop_event.is_set(): break

            try:
                state = ConnState.RECONNECTING
                attempts += 1
                await connect_once(mic, spk, stop_event, resume_token=resume_token)
                if stop_event.is_set(): break
                state = ConnState.RECONNECTING
                note("Connessione terminata. Preparazione a riconnettere...")
                last_error = ""
            except (ConnectionClosed, ConnectionClosedOK, ConnectionClosedError) as e:
                last_error = str(e)
                note(f"Connessione chiusa: {e}.")
                state = ConnState.RECONNECTING
            except OSError as e:
                last_error = str(e)
                note(f"Errore di rete/SSL: {e}")
                state = ConnState.RECONNECTING
            except Exception as e:
                last_error = str(e)
                note(f"Eccezione imprevista: {e}")
                state = ConnState.RECONNECTING

            spk.flush(); mic.clear(); mic.set_enabled(True)

            delay = min(backoff, RECONNECT_MAX_BACKOFF) + random.random()
            note(f"Riconnessione tra {delay:.1f}s...")
            import time as _t
            end_t = _t.monotonic() + delay
            while _t.monotonic() < end_t and not stop_event.is_set(): await asyncio.sleep(0.2)
            backoff = min(backoff * 2, RECONNECT_MAX_BACKOFF)
            state = ConnState.RECONNECTING
    finally:
        try:
            stop_event.set()
            hud_task.cancel()
            with contextlib.suppress(Exception):
                await hud_task
            hud.close()
        finally:
            try: await mic.stop()
            finally: spk.stop()
        note("Terminato.")

if __name__ == "__main__":
    asyncio.run(main())
