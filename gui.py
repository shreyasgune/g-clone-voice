import tkinter as tk
from tkinter import messagebox, filedialog
import soundfile as sf
import os

from audio_utility import record_audio
from tts_load_engine import QwenTTS

VOICE_DIR = "data/voices"
OUT_DIR   = "data/outputs"

os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

tts = QwenTTS()
REF_AUDIO = f"{VOICE_DIR}/ref.wav"  # default recording path

def record_voice():
    record_audio(REF_AUDIO, duration=10)
    messagebox.showinfo("Done", f"Reference voice recorded at {REF_AUDIO}")

def load_voice_file():
    global REF_AUDIO
    file_path = filedialog.askopenfilename(
        title="Select Reference Audio",
        filetypes=[("WAV files", "*.wav")]
    )
    if file_path:
        REF_AUDIO = file_path
        messagebox.showinfo("Loaded", f"Reference audio set to:\n{REF_AUDIO}")

def generate():
    ref_text = ref_text_box.get("1.0", tk.END).strip()
    target_text = target_text_box.get("1.0", tk.END).strip()

    if not os.path.exists(REF_AUDIO):
        messagebox.showerror("Error", "Record or load a reference voice first.")
        return

    if not ref_text or not target_text:
        messagebox.showerror("Error", "Fill both text boxes.")
        return

    wav, sr = tts.clone(
        text=target_text,
        ref_audio=REF_AUDIO,
        ref_text=ref_text
    )

    out_path = f"{OUT_DIR}/output_voice_clone.wav"
    sf.write(out_path, wav, sr)
    messagebox.showinfo("Success", f"Saved to {out_path}")

root = tk.Tk()
root.title("g-clone-voice service - shreyas gune")

tk.Button(root, text="Record Reference Voice", command=record_voice).pack(pady=5)
tk.Button(root, text="Load Reference Audio", command=load_voice_file).pack(pady=5)

tk.Label(root, text="Reference Text (what the speaker says in ref audio)").pack()
ref_text_box = tk.Text(root, height=4, width=60)
ref_text_box.pack(pady=5)

tk.Label(root, text="Target Text (what you want spoken)").pack()
target_text_box = tk.Text(root, height=4, width=60)
target_text_box.pack(pady=5)

tk.Button(root, text="🔊 Generate Speech", command=generate).pack(pady=10)

root.mainloop()
