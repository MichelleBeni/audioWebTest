from flask import Flask, request, render_template, send_from_directory
import os
from features_extraction_module import extract_extra_features, generate_expressiveness_plot, generate_clarity_plot
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PLOTS_FOLDER = 'plots'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

reference_df = pd.read_csv("reference_dataset.csv", encoding='utf-8')

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
    if not file.filename.endswith('.wav'):
        return 'Only .wav files are supported.', 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    # ✅ תוודאי שהתיקייה uploads קיימת
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    file.save(filepath)


    pitch_var, pitch_rate = extract_extra_features(filepath)

    exp_fig = generate_expressiveness_plot(pitch_var, reference_df)
    clr_fig = generate_clarity_plot(pitch_rate, reference_df)

    exp_name = f"expressiveness_plot.png"
    clr_name = f"clarity_plot.png"
    exp_fig.savefig(os.path.join(PLOTS_FOLDER, exp_name), bbox_inches='tight')
    clr_fig.savefig(os.path.join(PLOTS_FOLDER, clr_name), bbox_inches='tight')

    return render_template('index.html',
                           pitch_var=round(pitch_var, 3),
                           pitch_rate=round(pitch_rate, 3),
                           expressiveness_plot=exp_name,
                           clarity_plot=clr_name)

@app.route('/plots/<filename>')
def plot_file(filename):
    return send_from_directory(PLOTS_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
