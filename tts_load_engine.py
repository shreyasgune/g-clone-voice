import torch
from qwen_tts import Qwen3TTSModel

class QwenTTS:
    def __init__(self):
        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="auto",
            dtype=torch.float32,
            attn_implementation="flash_attention_2",
        )

    def clone(self, text, ref_audio, ref_text):
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language="English",
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        return wavs[0], sr
