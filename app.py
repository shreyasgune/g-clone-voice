import os
import sys
import torch
import threading
import datetime
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import soundfile as sf

from qwen_tts import Qwen3TTSModel
from huggingface_hub import snapshot_download
import whisper
from tqdm import tqdm  # fixed snapshot_download progress

# ------------------ PATHS ------------------
VOICE_DIR = "data/voices"
OUT_DIR = "data/outputs"
QC_MODEL_DIR = "models"

os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(QC_MODEL_DIR, exist_ok=True)

# ------------------ GLOBALS ------------------
REF_AUDIO = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
audio_engine = None

# ------------------ UI ------------------
root = tk.Tk()
root.title("g-clone-voice service")

# Status & console
status_var = tk.StringVar(value="Ready")
console = tk.Text(root, height=15, state=tk.DISABLED)
console.pack(fill=tk.BOTH, expand=True)
progress = ttk.Progressbar(root, length=300)
progress.pack()

def run_on_ui_thread(fn):
    root.after(0, fn)

def log(msg):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    console.config(state=tk.NORMAL)
    console.insert(tk.END, f"[{timestamp}] {msg}\n")
    console.see(tk.END)
    console.config(state=tk.DISABLED)

def set_status(text):
    run_on_ui_thread(lambda: status_var.set(text))

def start_progress(msg):
    set_status(msg)
    progress.config(mode="indeterminate")
    progress.start(10)

def stop_progress(msg="Ready"):
    progress.stop()
    progress["value"] = 0
    set_status(msg)

# ------------------ LOGGER ------------------
class TkLogger:
    def write(self, msg):
        msg = msg.strip()
        if msg:
            run_on_ui_thread(lambda: log(msg))
    def flush(self):
        pass

sys.stdout = TkLogger()
sys.stderr = TkLogger()

# ------------------ MODEL DOWNLOAD ------------------
def download_model(repo_id: str, local_dir: str):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    log(f"Downloading model to {local_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        max_workers=1,
        tqdm_class=tqdm
    )
    log("Model download complete.")

# ------------------ QWEN TTS ------------------
class QwenTTS:
    def __init__(self, model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
        local_dir = QC_MODEL_DIR

        if not os.path.isdir(local_dir) or not os.listdir(local_dir):
            log("Model not found locally, downloading...")
            start_progress("Downloading model...")
            download_model(repo_id=model_name, local_dir=local_dir)
            stop_progress("Model downloaded")
        else:
            log("Model already available locally.")

        log("Loading Qwen TTS model...")
        self.model = Qwen3TTSModel.from_pretrained(
            local_dir,
            device_map="auto",
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        log("Qwen TTS model loaded.")

    def clone(self, text, ref_audio, ref_text):
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language="English",
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        return wavs[0], sr

# ------------------ ASR ------------------
class ASR:
    def __init__(self, model_name=None, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_name is None:
            model_name = "small" if device == "cuda" else "tiny"

        self.device = device
        log(f"Loading Whisper ASR model ({model_name}) on {device}...")
        self.model = whisper.load_model(model_name, device=device)
        log("Whisper ASR model loaded.")

    def transcribe(self, wav_path):
        result = self.model.transcribe(wav_path)
        return result["text"]

# ------------------ AUDIO ENGINE ------------------
class QwenAudio:
    def __init__(self, device=None):
        self.tts = QwenTTS()
        self.asr = ASR(device=device)

    def transcribe(self, wav_path):
        return self.asr.transcribe(wav_path)

# ------------------ DEVICE ------------------
def resolve_device(selection: str) -> str:
    return "cuda" if selection == "NVIDIA GPU" else "cpu"

def on_device_change(event=None):
    global DEVICE
    DEVICE = resolve_device(device_var.get())
    log(f"Device set to: {DEVICE}")

# ------------------ ENGINE START ------------------
def start_engine():
    global audio_engine

    def worker():
        try:
            set_status("Initializing audio engine...")
            log(f"Using device: {DEVICE}")
            start_progress("Loading model (may download if missing)...")
            audio_engine = QwenAudio(device=DEVICE)
            log("Audio engine ready")
            run_on_ui_thread(lambda: generate_btn.config(state=tk.NORMAL))
            set_status("Ready")
        except Exception as e:
            log(f"Engine load failed: {e}")
            messagebox.showerror("Fatal Error", str(e))
        finally:
            stop_progress()

    threading.Thread(target=worker, daemon=True).start()

# ------------------ UI CALLBACKS ------------------
def load_voice_file():
    global REF_AUDIO
    if audio_engine is None:
        messagebox.showwarning("Loading", "Audio engine is still loading.")
        return

    file_path = filedialog.askopenfilename(
        title="Select Reference Audio",
        filetypes=[("WAV files", "*.wav")]
    )

    if not file_path:
        return

    REF_AUDIO = file_path
    log(f"Loaded reference audio: {REF_AUDIO}")

    start_progress("Transcribing reference audio...")
    threading.Thread(target=run_transcription, daemon=True).start()

def run_transcription():
    try:
        transcription = audio_engine.transcribe(REF_AUDIO)

        def update_ui():
            ref_text_box.delete("1.0", tk.END)
            ref_text_box.insert(tk.END, transcription)
            log("Transcription completed")

        run_on_ui_thread(update_ui)
    except Exception as exc:
        run_on_ui_thread(lambda: messagebox.showerror("ASR Error", str(exc)))
        log(f"Transcription failed: {exc}")
    finally:
        run_on_ui_thread(lambda: stop_progress())

def generate():
    if audio_engine is None:
        messagebox.showwarning("Loading", "Audio engine is still loading.")
        return
    if not REF_AUDIO or not os.path.exists(REF_AUDIO):
        messagebox.showerror("Error", "Load reference audio first.")
        return
    if not ref_text_box.get("1.0", tk.END).strip():
        messagebox.showerror("Error", "Reference text is empty.")
        return
    if not target_text_box.get("1.0", tk.END).strip():
        messagebox.showerror("Error", "Target text is empty.")
        return

    threading.Thread(target=run_generation, daemon=True).start()

def run_generation():
    run_on_ui_thread(lambda: generate_btn.config(state=tk.DISABLED))
    run_on_ui_thread(lambda: start_progress("Generating voice..."))

    try:
        wav, sr = audio_engine.tts.clone(
            text=target_text_box.get("1.0", tk.END).strip(),
            ref_audio=REF_AUDIO,
            ref_text=ref_text_box.get("1.0", tk.END).strip()
        )

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"{OUT_DIR}/voice_clone_{ts}.wav"
        sf.write(out_path, wav, sr)

        log(f"Saved output: {out_path}")
        run_on_ui_thread(lambda: messagebox.showinfo("Success", out_path))
    except Exception as e:
        log(f"ERROR: {e}")
        run_on_ui_thread(lambda: messagebox.showerror("Error", str(e)))
    finally:
        run_on_ui_thread(lambda: generate_btn.config(state=tk.NORMAL))
        run_on_ui_thread(lambda: stop_progress())

# ------------------ UI LAYOUT ------------------
# UI elements like ref_text_box, target_text_box, device_var, generate_btn
# should be created here (same as your previous script)
# For brevity, you can reuse your UI setup block

log("Application started.")
start_engine()
root.mainloop()
