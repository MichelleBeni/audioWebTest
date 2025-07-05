from flask import Flask, render_template, request
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import whisper
import subprocess
from werkzeug.utils import secure_filename
from features_extraction_module import extract_features_for_boxplot

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PLOTS_FOLDER = 'static/plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
df = pd.read_csv('reference_dataset.csv')

def create_boxplot_with_line(df, column, new_value, ylabel, filename):
    plt.figure()
    df.boxplot(column=column, vert=True, patch_artist=True)
    plt.axhline(y=new_value, color='red', linestyle='--', label='Your Recording')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(os.path.join(PLOTS_FOLDER, filename))
    plt.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if not (file.filename.endswith('.wav') or file.filename.endswith('.mp3') or file.filename.endswith('.mp4')):
        return 'Only .wav, .mp3, and .mp4 files are supported.', 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    if not filepath.endswith('.wav'):
        new_path = filepath.rsplit('.', 1)[0] + '.wav'
        subprocess.run(['ffmpeg', '-y', '-i', filepath, new_path])
        filepath = new_path

    pitch_var, pitch_rate, fluency_wpm, num_words, duration_sec = extract_features_for_boxplot(filepath, df)

    create_boxplot_with_line(df, 'pitch_variability', pitch_var, 'Pitch Variability', 'pitch_var_plot.png')
    create_boxplot_with_line(df, 'pitch_change_rate', pitch_rate, 'Pitch Change Rate', 'pitch_rate_plot.png')
    create_boxplot_with_line(df, 'fluency', fluency_wpm, 'Fluency (WPM)', 'fluency_plot.png')

    return render_template('index.html',
        pitch_var=round(pitch_var, 3),
        pitch_rate=round(pitch_rate, 3),
        fluency=round(fluency_wpm, 2),
        num_words=num_words,
        duration=round(duration_sec, 2),
        pitch_var_plot='plots/pitch_var_plot.png',
        pitch_rate_plot='plots/pitch_rate_plot.png',
        fluency_plot='plots/fluency_plot.png'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))



