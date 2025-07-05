import librosa
import numpy as np
import whisper
from sklearn.linear_model import LinearRegression


def extract_features_for_boxplot(audio_path, df):
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
    pitch_rate = np.mean(pitch_diff)

    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="he", word_timestamps=True)
    words = [seg['text'] for seg in result['segments']]
    num_words = sum(len(w.split()) for w in words)

    duration_sec = result['segments'][-1]['end'] if result['segments'] else 1
    fluency = (num_words / duration_sec) * 60 if duration_sec > 0 else 0

    return pitch_var, pitch_rate, fluency, num_words, duration_sec

def create_fit_plot(df, feature_column, label_column, new_value, xlabel, ylabel, filename):
    # סינון ערכים תקינים
    df_filtered = df[[feature_column, label_column]].dropna()
    X = df_filtered[[feature_column]].values
    y = df_filtered[label_column].values

    # מודל רגרסיה
    model = LinearRegression()
    model.fit(X, y)
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)

    # חיזוי עבור ההקלטה החדשה
    new_y = model.predict(np.array([[new_value]]))[0]

    # ציור
    plt.figure()
    plt.plot(x_range, y_pred, label='Linear Fit', color='navy')
    plt.scatter(new_value, new_y, color='red', label='Estimated Score', zorder=5)
    plt.axvline(x=new_value, color='red', linestyle='--')
    plt.axhline(y=new_y, color='red', linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} as Function of {xlabel}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'static/plots/{filename}')
    plt.close()
