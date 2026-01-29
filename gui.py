import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import soundfile as sf
import os
import threading
import datetime
import torch

from tts_load_engine import QwenAudio

VOICE_DIR = "data/voices"
OUT_DIR   = "data/outputs"
REF_AUDIO = None
DEVICE = None

os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

audio_engine = QwenAudio(device=DEVICE)


def check_cuda_support():
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        log(f"CUDA available: {device}")
        messagebox.showinfo(
            "CUDA Enabled",
            f"CUDA is available.\n\nUsing GPU:\n{device}"
        )
        return "cuda"
    else:
        log("CUDA not available. Falling back to CPU.")
        messagebox.showwarning(
            "CUDA Not Available",
            "CUDA is not available on this system.\n\n"
            "The application will use the CPU instead.\n"
            "This may be slower."
        )
        return "cpu"

def log(msg):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    console.config(state=tk.NORMAL)
    console.insert(tk.END, f"[{timestamp}] {msg}\n")
    console.see(tk.END)
    console.config(state=tk.DISABLED)

def run_on_ui_thread(fn):
    root.after(0, fn)

def load_voice_file():
    global REF_AUDIO
    file_path = filedialog.askopenfilename(
        title="Select Reference Audio",
        filetypes=[("WAV files", "*.wav")]
    )
    if not file_path:
        return

    REF_AUDIO = file_path
    log(f"Loaded reference audio: {REF_AUDIO}")

    messagebox.showinfo(
        "Audio Loaded",
        "You have loaded the WAV.\n\nI'm now going to try and transcribe it."
    )

    # Start progress bar
    progress.start(10)

    # Run transcription in background
    threading.Thread(
        target=run_transcription,
        daemon=True
    ).start()

def run_transcription():
    try:
        log("Transcribing reference audio...")

        transcription = audio_engine.transcribe(REF_AUDIO)

        def update_ui():
            ref_text_box.delete("1.0", tk.END)
            ref_text_box.insert(tk.END, transcription)
            log("Transcription completed")

        run_on_ui_thread(update_ui)

    except Exception as exc:
        error_msg = str(exc)

        def show_error():
            log(f"Transcription failed: {error_msg}")
            messagebox.showerror("ASR Error", error_msg)

        run_on_ui_thread(show_error)

    finally:
        run_on_ui_thread(progress.stop)



def generate():
    target_text = target_text_box.get("1.0", tk.END).strip()
    ref_text = ref_text_box.get("1.0", tk.END).strip()

    if not REF_AUDIO or not os.path.exists(REF_AUDIO):
        messagebox.showerror("Error", "Please load a reference audio first.")
        return

    if not ref_text:
        messagebox.showerror(
            "Error",
            "Please enter the reference text (what is spoken in the reference audio)."
        )
        return

    if not target_text:
        messagebox.showerror("Error", "Please enter target text.")
        return

    threading.Thread(target=run_generation, daemon=True).start()

def run_generation():
    try:
        generate_btn.config(state=tk.DISABLED)
        progress.start(10)

        log("Starting voice generation...")
        log("Calling TTS engine ( cloning)...")

        wav, sr = audio_engine.tts.clone(
            text=target_text_box.get("1.0", tk.END).strip(),
            ref_audio=REF_AUDIO,
            ref_text=ref_text_box.get("1.0", tk.END).strip()
    )

        out_path = f"{OUT_DIR}/output_voice_clone.wav"
        sf.write(out_path, wav, sr)

        log("Audio generated successfully")
        log(f"Saved output to {out_path}")

        messagebox.showinfo("Success", f"Saved to {out_path}")

    except Exception as e:
        log(f"ERROR: {e}")
        messagebox.showerror("Error", str(e))

    finally:
        progress.stop()
        generate_btn.config(state=tk.NORMAL)
        log("Ready.")

root = tk.Tk()
root.title("g-clone-voice service - shreyas gune")

tk.Button(root, text="Load Reference Audio", command=load_voice_file).pack(pady=6)

tk.Label(root, text="Reference Text (what is spoken in the reference audio)").pack()
ref_text_box = tk.Text(root, height=4, width=60)
ref_text_box.pack(pady=5)

tk.Label(root, text="Target Text (what you want spoken)").pack()
target_text_box = tk.Text(root, height=5, width=60)
target_text_box.pack(pady=5)

progress = ttk.Progressbar(
    root,
    orient="horizontal",
    length=400,
    mode="indeterminate"
)
progress.pack(pady=8)

generate_btn = tk.Button(root, text="Generate PLZ", command=generate)
generate_btn.pack(pady=8)

tk.Label(root, text="Console Output").pack()

console_frame = tk.Frame(root)
console_frame.pack(padx=10, pady=5, fill="both", expand=True)

scrollbar = tk.Scrollbar(console_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

console = tk.Text(
    console_frame,
    height=10,
    width=70,
    state=tk.DISABLED,
    bg="#111",
    fg="#0f0",
    insertbackground="white",
    yscrollcommand=scrollbar.set
)
console.pack(side=tk.LEFT, fill="both", expand=True)

scrollbar.config(command=console.yview)

log("Application started.")
DEVICE = check_cuda_support()


root.mainloop()
