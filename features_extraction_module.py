
import numpy as np
import librosa
import whisper

model = whisper.load_model("base")

def extract_extra_features(audio_path):
    y, sr = librosa.load(audio_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_var = np.std(pitch_values)
    pitch_diff = np.abs(np.diff(pitch_values))
    pitch_rate = np.mean(pitch_diff)
    return pitch_var, pitch_rate

def extract_fluency(audio_path):
    result = model.transcribe(audio_path, language="he")
    transcript = result['text']
    words = transcript.strip().split()
    num_words = len(words)
    y, sr = librosa.load(audio_path)
    duration_sec = librosa.get_duration(y=y, sr=sr)
    fluency_wpm = (num_words / duration_sec) * 60
    return fluency_wpm, num_words, duration_sec
