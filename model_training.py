#model_training.py
import os

# OGRANICZANIE LOGÓW TENSORFLOW:
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input
from sklearn.model_selection import train_test_split

from data_processing import (
    load_all_data_from_json,
    load_unknown_persons,
    interpolate_data,
    highpass_filter,
    trim_data,
    sliding_window_segmentation,
    prepare_data_for_model
)

from constants import (
    MODEL_PATH,
    PERSON_ID_MAP_FILE,
    UNKNOWN_PERSONS_FOLDER,
    UNKNOWN_CLASS_ID
)


def create_cnn_lstm_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),
        LSTM(100, return_sequences=True),
        LSTM(50, return_sequences=False),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(person_map_file=PERSON_ID_MAP_FILE):
    """
    Trenuje model na danych z pliku JSON (osoby znane) oraz
    dodatkowo z folderu 'unknown_persons' (osoby nieznane = klasa 9999).
    """

    print("Wczytywanie danych osób znanych...")
    raw_data_known = load_all_data_from_json(person_map_file)

    print("Wczytywanie danych osób nieznanych (UNKNOWN class)...")
    raw_data_unknown = load_unknown_persons(UNKNOWN_PERSONS_FOLDER, unknown_label=UNKNOWN_CLASS_ID)

    if not raw_data_unknown.empty:
        raw_data = pd.concat([raw_data_known, raw_data_unknown], axis=0, ignore_index=True)
    else:
        raw_data = raw_data_known

    print(f"Łączny rozmiar wczytanych danych: {raw_data.shape}")

    # interpolacja, filtracja, przycinanie, segmentacja, przygotowanie
    print("Rozpoczynam przetwarzanie danych...")
    data_interpolated = interpolate_data(raw_data, target_frequency=100)
    data_filtered = highpass_filter(data_interpolated, cutoff=0.115, fs=100)
    data_trimmed = trim_data(data_filtered, trim_seconds=4)
    segments = sliding_window_segmentation(data_trimmed, window_size_seconds=2.5, overlap=0.8, sampling_rate=100)
    X, y = prepare_data_for_model(segments, target_length=200)

    if X.shape[0] == 0:
        raise ValueError("Brak segmentów po przetwarzaniu danych. Nie można trenować.")

    unique_labels = np.unique(y)
    sorted_labels = sorted(unique_labels.tolist())  # np. [0, 1, 2, 9999] itp.
    label_to_index = {original: idx for idx, original in enumerate(sorted_labels)}
    y_mapped = np.array([label_to_index[val] for val in y], dtype=np.int32)

    print("Rozkład klas (po mapowaniu) – etykiety w 0..(n-1):")
    print(pd.Series(y_mapped).value_counts())

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_mapped, test_size=0.2, stratify=y_mapped, random_state=42
    )
    num_classes = len(sorted_labels)

    print(f"Tworzenie modelu z liczbą klas = {num_classes}")
    model = create_cnn_lstm_model(X_train.shape[1:], num_classes)

    print("Trening modelu...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32
    )
    print("Trening zakończony.")

    # Zapis modelu + label_mapping
    model.save(MODEL_PATH)
    label_mapping_path = os.path.join(os.path.dirname(MODEL_PATH), "label_mapping.json")
    with open(label_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_labels, f, ensure_ascii=False, indent=4)

    print(f"Model zapisany w {MODEL_PATH}.")
    print(f"Label mapping zapisany w {label_mapping_path}.")


# -----------------------------
#   Funkcje pomocnicze do predykcji
# -----------------------------
def compute_entropy(prob):
    return -np.sum(prob * np.log(prob + 1e-12))


def predict_person(
    file_path_or_data,
    confidence_threshold=0.9,
    top_diff_threshold=0.05,
    entropy_threshold=2,
    debug=False
):


    from data_processing import load_data
    from data_processing import (
        interpolate_data,
        highpass_filter,
        trim_data,
        sliding_window_segmentation,
        prepare_data_for_model
    )

    # 1. Wczytywanie/obróbka danych
    if isinstance(file_path_or_data, str):
        raw_data = load_data(file_path_or_data)
    else:
        raw_data = file_path_or_data

    data_interpolated = interpolate_data(raw_data, target_frequency=100)
    data_filtered = highpass_filter(data_interpolated, cutoff=0.115, fs=100)
    data_trimmed = trim_data(data_filtered, trim_seconds=4)
    segments = sliding_window_segmentation(data_trimmed, window_size_seconds=2.5, overlap=0.8, sampling_rate=100)
    X, _ = prepare_data_for_model(segments, target_length=200)

    if len(X) == 0:
        if debug:
            print("[DEBUG] Brak segmentów po preprocessing -> Unknown")
        return "Unknown"

    # 2. Wczytanie modelu + label mapping
    model = load_model(MODEL_PATH)
    label_mapping_path = os.path.join(os.path.dirname(MODEL_PATH), "label_mapping.json")
    with open(label_mapping_path, 'r', encoding='utf-8') as f:
        sorted_labels = json.load(f)

    # 3. Predykcja (bez temperature scaling)
    raw_predictions = model.predict(X)  # shape: (n_segments, n_classes)
    predicted_classes = np.argmax(raw_predictions, axis=1)
    confidences = np.max(raw_predictions, axis=1)

    from collections import Counter
    class_count = Counter(predicted_classes)

    avg_confidences = {}
    avg_entropies = {}
    for c, count in class_count.items():
        idxs = np.where(predicted_classes == c)[0]
        avg_conf = np.mean(confidences[idxs])
        ent_values = [compute_entropy(raw_predictions[i]) for i in idxs]
        avg_ent = np.mean(ent_values)
        avg_confidences[c] = avg_conf
        avg_entropies[c] = avg_ent

    if not avg_confidences:
        if debug:
            print("[DEBUG] Brak pewności (avg_confidences) -> Unknown")
        return "Unknown"

    # Sortujemy wg pewności malejąco
    sorted_by_conf = sorted(avg_confidences.items(), key=lambda x: x[1], reverse=True)

    best_class, best_conf = sorted_by_conf[0]
    best_entropy = avg_entropies[best_class]

    if len(sorted_by_conf) > 1:
        second_best_class, second_best_conf = sorted_by_conf[1]
    else:
        second_best_class, second_best_conf = None, 0.0

    # 4. Logi debug
    if debug:
        print("======================================================")
        print("[DEBUG] Szczegółowe informacje o predykcji:")
        print(f" - Najlepsza klasa (best_class)   = {best_class}")
        print(f" - Pewność najlepszej klasy       = {best_conf:.4f}")
        print(f" - Druga najlepsza klasa          = {second_best_class}")
        print(f" - Pewność drugiej klasy          = {second_best_conf:.4f}")
        print(f" - Różnica (top1 - top2)          = {best_conf - second_best_conf:.4f}")
        print(f" - Entropia (best_class)          = {best_entropy:.4f}")
        print(" - Wszystkie klasy (avg_conf, avg_ent):")
        for (c, c_conf) in sorted_by_conf:
            ent_c = avg_entropies[c]
            print(f"    klasa={c} -> conf={c_conf:.4f}, entropia={ent_c:.4f}")
        print("======================================================")

    # 5. Kryteria odrzucania
    if best_entropy > entropy_threshold:
        if debug:
            print("[DEBUG] Odrzucenie - wysoka entropia => Unknown")
        return "Unknown"

    if best_conf < confidence_threshold:
        if debug:
            print("[DEBUG] Odrzucenie - best_conf < confidence_threshold => Unknown")
        return "Unknown"

    if (best_conf - second_best_conf) < top_diff_threshold:
        if debug:
            print("[DEBUG] Odrzucenie - mała różnica top1 - top2 => Unknown")
        return "Unknown"

    original_id = sorted_labels[best_class]

    if original_id == UNKNOWN_CLASS_ID:
        if debug:
            print("[DEBUG] best_class = UNKNOWN_CLASS_ID => Unknown")
        return "Unknown"

    # 6. Rozpoznaj imię i nazwisko z pliku JSON
    with open(PERSON_ID_MAP_FILE, 'r', encoding='utf-8') as f:
        person_map = json.load(f)

    str_id = str(original_id)
    if str_id in person_map:
        fname = person_map[str_id].get("first_name", "Unknown")
        lname = person_map[str_id].get("last_name", "")
        return f"Przewidywana osoba: {fname} {lname}".strip()
    else:
        return f"Przewidywana osoba: {original_id}"
