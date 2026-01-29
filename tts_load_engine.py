import os
import torch
import whisper
from qwen_tts import Qwen3TTSModel

class QwenTTS:
    def __init__(self, model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
        local_dir = os.environ.get("QC_MODEL_DIR") or os.path.join("models", model_name.split("/")[-1])
        if os.path.isdir(local_dir):
            # Load from local directory (if you have pre-downloaded the model, we will use it, else we're gonna download it from DA HUGGIN FACE HUB)
            self.model = Qwen3TTSModel.from_pretrained(
                local_dir,
                device_map="auto",
                dtype=torch.float32,
                attn_implementation="sdpa",
            )
        else:
            try:
                #downloadin from huggin face, its like 3.6GB
                self.model = Qwen3TTSModel.from_pretrained(
                    model_name,
                    device_map="auto",
                    dtype=torch.float32,
                    attn_implementation="sdpa", #I'd change this to flash attention but I don't have a NVIDIA GPU to test with :(
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model '{model_name}'. To bundle the model for an EXE, pre-download it to '{local_dir}' "
                    "and set the environment variable QC_MODEL_DIR to that path, or run the build script with the preload option."
                    f" Original error: {e}"
                )

    def clone(self, text, ref_audio, ref_text):
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language="English",
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        return wavs[0], sr

class ASR:
    def __init__(self, model_name="base", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = whisper.load_model(model_name, device=device)

    def transcribe(self, wav_path):
        result = self.model.transcribe(wav_path)
        return result["text"]


class QwenAudio:
    def __init__(self, device=None):
        self.tts = QwenTTS()
        self.asr = ASR(device=device)

    def transcribe(self, wav_path):
        return self.asr.transcribe(wav_path)

    def clone_from_wav(self, wav_path, ref_audio):
        text = self.transcribe(wav_path).strip()
        return self.tts.clone(text, ref_audio, text)
