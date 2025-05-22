from flask import Flask, request, jsonify, render_template
from predict import predict_audio
import os
import tempfile
import whisper

from keybert import KeyBERT
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from codecarbon import EmissionsTracker

app = Flask(__name__)

# Charger les modèles une seule fois
whisper_model = whisper.load_model("base")  # "base", "small", "medium", "large"
#kw_model = KeyBERT()
#summarizer = LsaSummarizer()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(temp_path)

    try:
        # 🔊 Prédiction du locuteur
        label, probabilities = predict_audio(
            temp_path,
            model_path=r"C:\Users\hp\desktop\Speacker_Identification\model_fold1.h5"
        )

        if label is None:
            return jsonify({'error': 'Failed to predict'}), 500

        # 📜 Transcription avec Whisper
        whisper_result = whisper_model.transcribe(temp_path, language=None, fp16=False)
        transcription = whisper_result.get("text", "").strip()

        # 🗝️ Extraction de mots-clés avec KeyBERT
        '''if transcription:
            keywords = kw_model.extract_keywords(
                transcription,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=5
            )
            keyword_list = [kw[0] for kw in keywords]
        else:
            keyword_list = []

        # 🧠 Résumé automatique avec Sumy
        if transcription:
            parser = PlaintextParser.from_string(transcription, Tokenizer("english"))
            summary_sentences = summarizer(parser.document, sentences_count=3)
            summary = " ".join(str(sentence) for sentence in summary_sentences)
        else:
            summary = "" '''

        confidence = float(max(probabilities)) if probabilities else 0.0

        return jsonify({
            'predicted_label': label,
            'confidence': confidence,
            'probabilities': probabilities,
            'transcription': transcription,
            #'keywords': keyword_list,
            #'resume': summary
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(debug=True)
