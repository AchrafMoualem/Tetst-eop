import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.utils.class_weight import compute_class_weight

# ==== 1. PARAMETERS ====
DATASET_PATH = r"C:\\Users\\hp\\Desktop\\dataset___splited"
SAMPLE_RATE = 22050
DURATION = 5
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MFCC = 40
MAX_PAD_LEN = 100
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4


# ==== 2. DATA AUGMENTATION ====
def augment_audio(y, sr):
    if np.random.rand() > 0.5:
        y = librosa.effects.time_stretch(y, rate=0.8 + 0.4 * np.random.rand())
    if np.random.rand() > 0.5:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.choice([-2, -1, 0, 1, 2]))
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 0.005 * np.random.rand(), y.shape)
        y += noise
    if np.random.rand() > 0.5:
        y *= np.random.uniform(0.8, 1.2)
    return y


def augment_mfcc(mfcc):
    if np.random.rand() > 0.5:
        t = np.random.randint(0, mfcc.shape[1])
        mfcc[:, t:t + np.random.randint(5, 15)] = 0
    if np.random.rand() > 0.5:
        f = np.random.randint(0, mfcc.shape[0])
        mfcc[f:f + np.random.randint(2, 6), :] = 0
    return mfcc


# ==== 3. FEATURE EXTRACTION ====
def extract_features(file_path, augment=False):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        if augment:
            audio = augment_audio(audio, sr)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        if augment:
            mfcc = augment_mfcc(mfcc)
        mfcc = librosa.util.fix_length(mfcc, size=MAX_PAD_LEN, axis=1)
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# ==== 4. LOAD DATASET ====
def load_dataset(dataset_path, augment_train=False):
    X, y = [], []
    label_to_index = {}

    for idx, label in enumerate(sorted(os.listdir(dataset_path))):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue
        label_to_index[label] = idx
        print(f"Processing {label}...")
        for file in os.listdir(label_path):
            if file.endswith(".wav"):
                file_path = os.path.join(label_path, file)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(idx)
                if augment_train:
                    features_aug = extract_features(file_path, augment=True)
                    if features_aug is not None:
                        X.append(features_aug)
                        y.append(idx)

    return np.array(X), np.array(y), label_to_index


# ==== 5. MODEL CREATION ====
def create_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ==== 6. TRAINING WITH CROSS-VALIDATION ====
def train_with_cv(X, y, num_classes, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_splits}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = create_model(X_train.shape[1:], num_classes)

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(class_weights))

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6),
            tf.keras.callbacks.ModelCheckpoint(f'model_fold{fold + 1}.h5', save_best_only=True)
        ]

        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            callbacks=callbacks,
                            class_weight=class_weights,
                            verbose=2)

        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        scores.append(val_acc)
        np.save(f"history_fold{fold + 1}.npy", history.history)

        print(f"Fold {fold + 1} - Validation Accuracy: {val_acc:.4f}")

    print(f"\nAverage Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")


# ==== 7. MAIN PIPELINE ====
def main():
    X, y, label_to_index = load_dataset(DATASET_PATH, augment_train=True)
    X = X[..., np.newaxis]

    # Normalize
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-8
    X = (X - mean) / std

    np.save('mean.npy', mean)
    np.save('std.npy', std)

    np.save('label_mapping.npy', label_to_index)

    train_with_cv(X, y, num_classes=len(label_to_index))
    print("✅ Training completed. Models and histories saved.")


if __name__ == "__main__":
    main()
