
import numpy as np
import librosa
import whisper

model = whisper.load_model("base")

def extract_features_for_boxplot(audio_path):
    y, sr = librosa.load(audio_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []

    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)

    pitch_values = np.array(pitch_values)
    pitch_var = np.std(pitch_values)
    pitch_diff = np.abs(np.diff(pitch_values))
    pitch_rate = np.mean(pitch_diff)

    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language='he')
    text = result['text']
    words = text.split()
    num_words = len(words)

    duration_sec = librosa.get_duration(y=y, sr=sr)
    fluency = num_words / (duration_sec / 60)

    return pitch_var, pitch_rate, fluency, num_words, duration_sec
