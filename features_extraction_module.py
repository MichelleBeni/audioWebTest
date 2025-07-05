import librosa
import matplotlib.pyplot as plt
import numpy as np


def extract_extra_features(file_path):
    y, sr = librosa.load(file_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    pitch_var = np.std(pitch_values)
    pitch_diff = abs(pitch_values[1:] - pitch_values[:-1])
    pitch_rate = pitch_diff.mean()
    return pitch_var, pitch_rate

def generate_expressiveness_plot(pitch_var, reference_df):
    fig, ax = plt.subplots()
    ax.scatter(reference_df['pitch_variability'], reference_df['Expressiveness'], color='orange')
    ax.axvline(pitch_var, color='red', linestyle='--')
    ax.set_xlabel("pitch_variability")
    ax.set_ylabel("Expressiveness")
    ax.set_title("Expressiveness vs pitch_variability")
    return fig

def generate_clarity_plot(pitch_rate, reference_df):
    fig, ax = plt.subplots()
    ax.scatter(reference_df['pitch_change_rate'], reference_df['Clarity'], color='orange')
    ax.axvline(pitch_rate, color='red', linestyle='--')
    ax.set_xlabel("pitch_change_rate")
    ax.set_ylabel("Clarity")
    ax.set_title("Clarity vs pitch_change_rate")
    return fig
