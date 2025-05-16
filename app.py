from flask import Flask, request, jsonify, render_template
from predict import predict_audio
import os
import tempfile

app = Flask(__name__)

# Page d'accueil
@app.route('/')
def index():
    return render_template("index.html")

# Route de prédiction
@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(temp_path)

    try:
        label, probabilities = predict_audio(temp_path, model_path=r"C:\Users\hp\desktop\Speacker_Identification\model_fold1.h5")
        if label is None:
            return jsonify({'error': 'Failed to predict'}), 500

        confidence = float(max(probabilities)) if probabilities else 0.0

        return jsonify({
            'predicted_label': label,
            'confidence': confidence,
            'probabilities': probabilities
        })

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Lancement de l'application
if __name__ == '__main__':
    app.run(debug=True)

