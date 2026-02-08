import time
import torchaudio
from speechbrain.inference.interfaces import foreign_class

MODEL_SOURCE = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"

def load_model():
    return foreign_class(
        source=MODEL_SOURCE,
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
    )

def ensure_mono_16k(waveform, sr):
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        sr = 16000
    return waveform, sr

def predict(classifier, wav_path):
    waveform, sr = torchaudio.load(wav_path)  # [channels, time]
    waveform, sr = ensure_mono_16k(waveform, sr)  # still [1, time]

    # SpeechBrain custom_interface for this model expects [batch, time]
    wavs = waveform.squeeze(0).unsqueeze(0)   # [1, time]

    t0 = time.time()
    out_prob, score, index, text_lab = classifier.classify_batch(wavs)
    t1 = time.time()

    return {
        "label": str(text_lab[0]),
        "confidence": float(score[0]),
        "inference_time_sec": round(t1 - t0, 4),
    }

if __name__ == "__main__":
    clf = load_model()
    print(predict(clf, "test.wav"))
