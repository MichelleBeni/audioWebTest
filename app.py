from flask import Flask, render_template, request
import os
import subprocess
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib.pyplot as plt
from features_extraction_module import extract_extra_features, extract_fluency
from curve_plot_module import create_curve_plot

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

df = pd.read_csv('reference_dataset.csv')

def create_scatter_plot(x_ref, y_ref, new_x, xlabel, ylabel, filename):
    plt.figure()
    plt.scatter(x_ref, y_ref, alpha=0.4, label='Dataset')
    plt.axvline(x=new_x, color='red', linestyle='--', label='Your Recording')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs {xlabel}')
    plt.legend()
    plt.savefig(f'static/plots/{filename}')
    plt.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('static/plots', exist_ok=True)
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

    pitch_var, pitch_rate = extract_extra_features(filepath)
    fluency_wpm, num_words, duration_sec = extract_fluency(filepath)

    create_scatter_plot(df['pitch_variability'], df['Expressiveness'], pitch_var, 'Pitch Variability', 'Expressiveness', 'expressiveness_plot.png')
    create_scatter_plot(df['pitch_change_rate'], df['Clarity'], pitch_rate, 'Pitch Change Rate', 'Clarity', 'clarity_plot.png')

    # טען דאטה
    df_ref = pd.read_csv("reference_dataset.csv")

# גרף 1: Expressiveness מול Pitch Variability
    create_curve_plot(
        df=df_ref,
        x_col='pitch_variability',
        y_col='Expressiveness',
        new_x=pitch_var,
        xlabel='Pitch Variability (Hz)',
        ylabel='Expressiveness Score',
        filename='expressiveness_curve.png'
    )

# גרף 2: Clarity מול Pitch Change Rate
    create_curve_plot(
        df=df_ref,
        x_col='pitch_change_rate',
        y_col='Clarity',
        new_x=pitch_rate,
        xlabel='Pitch Change Rate (Hz/sec)',
        ylabel='Clarity Score',
        filename='clarity_curve.png'
    )

    return render_template('index.html',
        pitch_var=round(pitch_var, 3),
        pitch_rate=round(pitch_rate, 3),
        fluency=round(fluency_wpm, 2),
        num_words=num_words,
        duration=round(duration_sec, 2),
        expressiveness_plot='plots/expressiveness_plot.png',
        clarity_plot='plots/clarity_plot.png'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))




