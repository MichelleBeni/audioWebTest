import librosa
import numpy as np
import whisper

def extract_features_for_boxplot(audio_path, df):
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
    result = model.transcribe(audio_path, language="he", word_timestamps=True)
    words = [seg['text'] for seg in result['segments']]
    num_words = sum(len(w.split()) for w in words)

    duration_sec = result['segments'][-1]['end'] if result['segments'] else 1
    fluency = (num_words / duration_sec) * 60 if duration_sec > 0 else 0

    return pitch_var, pitch_rate, fluency, num_words, duration_sec
