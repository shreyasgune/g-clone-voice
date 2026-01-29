import sounddevice as sd
import soundfile as sf

def record_audio(filename, duration=10, samplerate=16000):
    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    sf.write(filename, audio, samplerate)
