
import numpy as np
import librosa
import whisper

model = whisper.load_model("base")

def extract_extra_features(audio_path):
    y, sr = librosa.load(audio_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []

    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if 50 < pitch < 500:  # רק תחום דיבור אנושי
            pitch_values.append(pitch)

    pitch_values = np.array(pitch_values)


    pitch_var = np.std(pitch_values)
    pitch_diff = np.abs(np.diff(pitch_values))

    # סינון שינויים לא סבירים – נניח מעל 100 הרץ בין פריימים סמוכים
    pitch_diff = pitch_diff[pitch_diff < 100]
    pitch_rate = np.mean(pitch_diff) if len(pitch_diff) > 0 else 0
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
