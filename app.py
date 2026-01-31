import os
import sys
import threading
import datetime
import tkinter as tk
from tkinter import messagebox, filedialog, ttk

import torch
import soundfile as sf
import whisper
from tqdm import tqdm
from huggingface_hub import snapshot_download

from qwen_tts import Qwen3TTSModel

from tqdm import tqdm

#PATHS
VOICE_DIR = "data/voices"
OUT_DIR = "data/outputs"
QC_MODEL_DIR = "models/qwen3-tts"

os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

#GLOBALS
REF_AUDIO = None
DEVICE = "cpu"
engine_ready = False
engine_loading = False
audio_engine = None


#UI
root = tk.Tk()
root.title("g-clone-voice service")
status_var = tk.StringVar(value="Ready")  # default text
status_label = tk.Label(root, textvariable=status_var)
status_label.pack(pady=4)

#UI HELPERS
def run_on_ui_thread(fn):
    root.after(0, fn)

def log(msg):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    console.config(state=tk.NORMAL)
    console.insert(tk.END, f"[{timestamp}] {msg}\n")
    console.see(tk.END)
    console.config(state=tk.DISABLED)

def set_status(text):
    status_var.set(text)

def start_progress(msg="Working..."):
    set_status(msg)
    progress.config(mode="indeterminate")
    progress.start(10)

def stop_progress(msg="Ready"):
    progress.stop()
    progress["value"] = 0
    set_status(msg)


#PROGRESS
progress = ttk.Progressbar(root, length=400, mode="indeterminate")
progress.pack(pady=6)


def start_progress(msg="Working..."):
    set_status(msg)  # optional, show text status
    progress.config(mode="indeterminate")
    progress.start(10)  # speed of animation

def stop_progress(msg="Ready"):
    progress.stop()
    progress["value"] = 0
    set_status(msg)

#MODEL DOWNLOAD
def download_model(repo_id: str, local_dir: str):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    log(f"Downloading model to {local_dir} ...")
    start_progress("Downloading model...")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        log("Model download complete.")
    finally:
        stop_progress()



#CUSTOM TQDM FOR TKINTER - UNUSED BUT MAY BE HANDY L8ER
class TkinterTqdm(tqdm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def display(self, msg=None, pos=None):
        # override tqdm display to redirect to your log
        if msg is None:
            msg = self.format_meter(
                n=self.n,
                total=self.total,
                elapsed=self.format_dict["elapsed"],
            )
        run_on_ui_thread(lambda: log(msg))


#TTS
class QwenTTS:
    def __init__(self, device):

        config_path = os.path.join(QC_MODEL_DIR, "config.json")
        if not os.path.isfile(config_path):
            log("Qwen TTS model not found locally. Downloading...")
            run_on_ui_thread(lambda: progress.start(10))
            download_model(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                QC_MODEL_DIR
            )
            run_on_ui_thread(progress.stop)

        log("Loading Qwen TTS model...")
        self.model = Qwen3TTSModel.from_pretrained(
            QC_MODEL_DIR,
            device_map="auto" if device == "cuda" else None,
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

#ASR
class ASR:
    def __init__(self, device):
        model_name = "small" if device == "cuda" else "tiny"
        log(f"Loading Whisper ASR ({model_name}) on {device}...")
        self.model = whisper.load_model(model_name, device=device)
        log("Whisper ASR model loaded.")

    def transcribe(self, wav):
        return self.model.transcribe(wav)["text"]

#ENGINE
class AudioEngine:
    def __init__(self, device):
        self.tts = QwenTTS(device)
        self.asr = ASR(device)

    def transcribe(self, wav):
        return self.asr.transcribe(wav)

#DEVICE
def on_device_change(*_):
    global DEVICE
    DEVICE = device_var.get()
    log(f"Device set to: {DEVICE}")

#ENGINE START
def start_engine():
    global audio_engine, engine_ready, engine_loading

    if engine_loading or engine_ready:
        log("Engine already started.")
        return

    engine_loading = True
    engine_ready = False

    def worker():
        global audio_engine, engine_ready, engine_loading
        try:
            log(f"Initializing engine on {DEVICE}...")
            run_on_ui_thread(lambda: progress.start(10))
            run_on_ui_thread(progress.stop)


            audio_engine = AudioEngine(device=DEVICE)

            engine_ready = True
            log("Audio engine ready.")

            run_on_ui_thread(lambda: load_btn.config(state=tk.NORMAL))
            run_on_ui_thread(lambda: generate_btn.config(state=tk.NORMAL))

        except Exception as e:
            err = str(e)
            log(f"Engine failed: {err}")
            run_on_ui_thread(
                lambda msg=err: messagebox.showerror("Fatal Error", msg)
            )


        finally:
            engine_loading = False
            run_on_ui_thread(progress.stop)

    threading.Thread(target=worker, daemon=True).start()


#UI CALLBACKS
def load_voice_file():
    global REF_AUDIO

    if not engine_ready:
        messagebox.showwarning(
            "Engine Not Ready",
            "Please wait until the engine finishes loading."
        )
        return


    REF_AUDIO = filedialog.askopenfilename(
        title="Select Reference Audio",
        filetypes=[("WAV files", "*.wav")]
    )

    if not REF_AUDIO:
        return

    log(f"Loaded reference audio: {REF_AUDIO}")
    progress.start(10)

    threading.Thread(target=run_transcription, daemon=True).start()


def run_transcription():
    try:
        text = audio_engine.transcribe(REF_AUDIO)
        run_on_ui_thread(lambda: ref_text_box.insert("1.0", text))
        log("Transcription completed.")
    finally:
        run_on_ui_thread(progress.stop)

def generate():
    if not engine_ready:
        messagebox.showwarning(
            "Engine Not Ready",
            "Please wait until the engine finishes loading."
        )
        return

    threading.Thread(target=run_generation, daemon=True).start()


def run_generation():
    try:
        generate_btn.config(state=tk.DISABLED)
        progress.start(10)

        wav, sr = audio_engine.tts.clone(
            target_text_box.get("1.0", tk.END).strip(),
            REF_AUDIO,
            ref_text_box.get("1.0", tk.END).strip()
        )

        out = f"{OUT_DIR}/clone_{datetime.datetime.now():%Y%m%d_%H%M%S}.wav"
        sf.write(out, wav, sr)
        log(f"Saved: {out}")

    finally:
        progress.stop()
        generate_btn.config(state=tk.NORMAL)

#UI LAYOUT
tk.Label(root, text="Device").pack()
device_var = tk.StringVar(value="cpu")
ttk.Combobox(
    root,
    textvariable=device_var,
    values=["cpu", "cuda"],
    state="readonly"
).pack()
device_var.trace_add("write", on_device_change)

tk.Button(root, text="Start Engine", command=start_engine).pack(pady=5)
load_btn = tk.Button(
    root,
    text="Load Reference Audio",
    command=load_voice_file,
    state=tk.DISABLED
)
load_btn.pack(pady=5)


tk.Label(root, text="Reference Text").pack()
ref_text_box = tk.Text(root, height=4, width=60)
ref_text_box.pack()

tk.Label(root, text="Target Text").pack()
target_text_box = tk.Text(root, height=5, width=60)
target_text_box.pack()

generate_btn = tk.Button(root, text="Generate", command=generate, state=tk.DISABLED)
generate_btn.pack(pady=8)

tk.Label(root, text="Console").pack()
console = tk.Text(root, height=10, bg="#111", fg="#0f0", state=tk.DISABLED)
console.pack(fill="both", expand=True)

#START
log("Application started.")
root.mainloop()
