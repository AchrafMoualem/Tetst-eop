import numpy as np
import librosa
import tensorflow as tf
import sounddevice as sd
import os

# ==== 1. PARAMETERS ====
SAMPLE_RATE = 22050
DURATION = 5
N_MFCC = 40
MAX_PAD_LEN = 100

# ==== 2. LOAD LABELS ====
label_to_index = np.load(r"C:\Users\hp\PyCharmMiscProject\.venv\label_mapping.npy", allow_pickle=True).item()
index_to_label = {v: k for k, v in label_to_index.items()}


# ==== 3. FEATURE EXTRACTION ====
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        mfcc = librosa.util.fix_length(mfcc, size=MAX_PAD_LEN, axis=1)
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def extract_features_from_array(audio):
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        mfcc = librosa.util.fix_length(mfcc, size=MAX_PAD_LEN, axis=1)
        return mfcc
    except Exception as e:
        print(f"Error processing live audio: {e}")
        return None


# ==== 4. NORMALIZATION ====
def normalize(X):
    mean = np.load(r"C:\Users\hp\PyCharmMiscProject\.venv\mean.npy")
    std = np.load(r"C:\Users\hp\PyCharmMiscProject\.venv\std.npy")
    return (X - mean) / std


# ==== 5. PREDICT FROM FILE ====
def predict_audio(file_path, model_path):
    model = tf.keras.models.load_model(model_path)
    mfcc = extract_features(file_path)
    if mfcc is None:
        return None
    X = mfcc[np.newaxis, ..., np.newaxis]
    X = normalize(X)
    preds = model.predict(X)
    pred_index = np.argmax(preds, axis=1)[0]
    return index_to_label[pred_index]


# ==== 6. PREDICT FROM LIVE MICROPHONE ====
def predict_live_microphone(model_path):
    print(f"🎙️ Speak now for {DURATION} seconds...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()

    model = tf.keras.models.load_model(model_path)
    mfcc = extract_features_from_array(audio)
    if mfcc is None:
        return None
    X = mfcc[np.newaxis, ..., np.newaxis]
    X = normalize(X)
    preds = model.predict(X)
    pred_index = np.argmax(preds, axis=1)[0]
    return index_to_label[pred_index]


# ==== 7. MAIN EXECUTION ====
if __name__ == "__main__":
    model_path = r"C:\Users\hp\PyCharmMiscProject\.venv\model_fold1.h5"

    # Option 1: Predict from file
    #audio_path = r"C:\Users\hp\Documents\identification_dataset\50_speakers_audio_data\Speaker0036\Speaker0036_029.wav"
    #prediction_file = predict_audio(audio_path, model_path)
    #if prediction_file is not None:
        #print(f"📁 Prediction from file: {prediction_file}")
    #else:
        #print("❌ Failed to predict from file.")

    # Option 2: Predict from live microphone
    prediction_live = predict_live_microphone(model_path)
    if prediction_live is not None:
        print(f"🎤 Prediction from microphone: {prediction_live}")
    else:
        print("❌ Failed to predict from microphone.")
