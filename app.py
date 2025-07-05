
from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from features_extraction_module import extract_extra_features, extract_fluency
import subprocess

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

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

    # Convert to wav if necessary
    if not filepath.endswith('.wav'):
        new_path = filepath.rsplit('.', 1)[0] + '.wav'
        subprocess.run(['ffmpeg', '-y', '-i', filepath, new_path])
        filepath = new_path

    pitch_var, pitch_rate = extract_extra_features(filepath)
    fluency_wpm, num_words, duration_sec = extract_fluency(filepath)

    # Dummy plot file names expected to exist in static/plots
    expressiveness_plot = 'static/plots/expressiveness_plot.png'
    clarity_plot = 'static/plots/clarity_plot.png'

    return render_template('index.html',
        pitch_var=round(pitch_var, 3),
        pitch_rate=round(pitch_rate, 3),
        fluency=round(fluency_wpm, 2),
        num_words=num_words,
        duration=round(duration_sec, 2),
        expressiveness_plot=expressiveness_plot,
        clarity_plot=clarity_plot
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
