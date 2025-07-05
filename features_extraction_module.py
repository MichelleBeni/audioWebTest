import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import whisper

def extract_features_for_boxplot(audio_path, df):
    # Load and analyze audio
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

    # Whisper transcription
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="he", word_timestamps=True)
    words = [seg['word'] for seg in result['segments']]
    num_words = len(words)
    duration_sec = librosa.get_duration(y=y, sr=sr)
    fluency_wpm = (num_words / duration_sec) * 60

    # Plot expressiveness
    plt.figure()
    plt.boxplot([df['pitch_variability']], vert=False, patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
    plt.axvline(x=pitch_var, color='red', linestyle='--', label='Your Recording')
    plt.xlabel('Pitch Variability')
    plt.title('Expressiveness vs Pitch Variability')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/plots/expressiveness_plot.png')
    plt.close()

    # Plot clarity
    plt.figure()
    plt.boxplot([df['pitch_change_rate']], vert=False, patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
    plt.axvline(x=pitch_rate, color='red', linestyle='--', label='Your Recording')
    plt.xlabel('Pitch Change Rate')
    plt.title('Clarity vs Pitch Change Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/plots/clarity_plot.png')
    plt.close()

    return pitch_var, pitch_rate, fluency_wpm, num_words, duration_sec
