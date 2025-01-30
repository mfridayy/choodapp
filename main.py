import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ====================================
# 1. Wczytywanie danych
# ====================================
def load_data(data_folder):
    print("Wczytywanie danych...")
    all_data = []
    for i in range(1, 35):  # Zakładamy, że mamy dane dla 34 osób
        file_path = os.path.join(data_folder, f"osoba{i}.xlsx")
        if os.path.exists(file_path):
            print(f"Wczytywanie pliku: {file_path}")
            person_data = pd.read_excel(file_path, header=None)

            # Przypisanie nazw kolumn
            person_data.columns = [
                "NodeName", "HostTimestamp", "notificationTime", "timeStamp",
                "RawData", "X", "Y", "Z"
            ]

            # Dodanie identyfikatora osoby
            person_data["PersonID"] = i
            all_data.append(person_data)
        else:
            print(f"Plik {file_path} nie istnieje!")

    if not all_data:
        raise ValueError("Nie udało się wczytać żadnych danych. Sprawdź ścieżki do plików.")

    data = pd.concat(all_data, ignore_index=True)
    print("Dane wczytane pomyślnie!")
    print(f"Liczba wczytanych wierszy: {len(data)}")
    return data


# ====================================
# 2. Interpolacja danych
# ====================================
def interpolate_data(data, target_frequency=100):
    print("Interpolacja danych do częstotliwości 100 Hz...")
    data["Time"] = pd.to_datetime(data["HostTimestamp"], unit='ms')
    data["ElapsedSeconds"] = (data["Time"] - data.groupby("PersonID")["Time"].transform("first")).dt.total_seconds()

    interpolated_data = []
    for person_id, group in data.groupby("PersonID"):
        start_time = group["ElapsedSeconds"].iloc[0]
        end_time = group["ElapsedSeconds"].iloc[-1]
        time_vector = np.arange(start_time, end_time, 1 / target_frequency)

        interpolated_group = pd.DataFrame({"ElapsedSeconds": time_vector})
        for axis in ["X", "Y", "Z"]:
            interpolated_group[axis] = np.interp(time_vector, group["ElapsedSeconds"], group[axis])
        interpolated_group["PersonID"] = person_id

        interpolated_data.append(interpolated_group)

    interpolated_data = pd.concat(interpolated_data, ignore_index=True)
    print("Interpolacja zakończona!")
    return interpolated_data


# ====================================
# 3. Filtracja górnoprzepustowa
# ====================================
def highpass_filter(data, cutoff, fs):
    print("Filtracja górnoprzepustowa...")
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='high', analog=False)

    filtered_data = data.copy()
    for axis in ["X", "Y", "Z"]:
        filtered_data[axis] = filtfilt(b, a, data[axis])

    print("Filtracja zakończona!")
    return filtered_data


# ====================================
# 4. Przycinanie pierwszych 4 sekund danych
# ====================================
def trim_data(data, trim_seconds=4):
    print("Przycinanie pierwszych 4 sekund danych...")
    trimmed_data = []
    for person_id, group in data.groupby("PersonID"):
        group = group[group["ElapsedSeconds"] >= trim_seconds]
        trimmed_data.append(group)

    trimmed_data = pd.concat(trimmed_data, ignore_index=True)
    print("Przycinanie zakończone!")
    return trimmed_data


# ====================================
# 5. Segmentacja metodą okien przesuwnych
# ====================================
def sliding_window_segmentation(data, window_size_seconds=2, overlap=0.2, sampling_rate=100):
    print("Rozpoczynam segmentację metodą okien przesuwnych...")
    window_size_samples = int(window_size_seconds * sampling_rate)
    step_size_samples = int(window_size_samples * (1 - overlap))

    segments = []
    for person_id, group in data.groupby("PersonID"):
        signal_length = len(group)
        for start in range(0, signal_length - window_size_samples + 1, step_size_samples):
            segment = group.iloc[start:start + window_size_samples][["X", "Y", "Z", "PersonID"]].copy()
            segments.append(segment)

    print("Segmentacja zakończona!")
    return segments


# ====================================
# 6. Przygotowanie danych do modelu
# ====================================
def prepare_data_for_model(segments, target_length=200, num_classes=34):
    X_data = []
    y_labels = []

    for segment in segments:
        values_x = segment["X"].values
        values_y = segment["Y"].values
        values_z = segment["Z"].values

        if len(values_x) < target_length:
            padded_x = np.pad(values_x, (0, target_length - len(values_x)), mode='constant')
            padded_y = np.pad(values_y, (0, target_length - len(values_y)), mode='constant')
            padded_z = np.pad(values_z, (0, target_length - len(values_z)), mode='constant')
        else:
            padded_x = values_x[:target_length]
            padded_y = values_y[:target_length]
            padded_z = values_z[:target_length]

        combined_values = np.stack([padded_x, padded_y, padded_z], axis=1)
        X_data.append(combined_values)
        y_labels.append(segment["PersonID"].iloc[0] - 1)

    X_data = np.array(X_data)
    y_labels = np.array(y_labels)
    X_data = (X_data - X_data.mean(axis=1, keepdims=True)) / X_data.std(axis=1, keepdims=True)

    return X_data, y_labels


# ====================================
# 7. Model CNN+LSTM
# ====================================
def create_cnn_lstm_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
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

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ====================================
# 8. Wyświetlenie liczby segmentów dla każdej klasy
# ====================================
def display_segment_counts(y_labels, num_classes):
    class_counts = pd.Series(y_labels).value_counts().sort_index()
    for class_id, count in class_counts.items():
        print(f"Klasa {class_id + 1}: {count} segmentów")


# ====================================
# Funkcja główna
# ====================================
if __name__ == "__main__":
    data_folder = r"C:\Users\Mateusz\Desktop\inżynierka\akcelerometr\Baza akcelerometr"

    # Wczytanie danych
    raw_data = load_data(data_folder)

    # Interpolacja danych
    interpolated_data = interpolate_data(raw_data)

    # Filtracja danych
    filtered_data = highpass_filter(interpolated_data, cutoff=0.115, fs=100)

    # Przycinanie pierwszych 4 sekund
    trimmed_data = trim_data(filtered_data)

    # Segmentacja danych metodą okien przesuwnych
    segments = sliding_window_segmentation(trimmed_data, window_size_seconds=2, overlap=0.2, sampling_rate=100)

    # Przygotowanie danych do modelu
    X, y = prepare_data_for_model(segments)

    # Wyświetlenie liczby segmentów dla każdej klasy
    display_segment_counts(y, num_classes=34)

    # Podział na dane treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Tworzenie modelu
    input_shape = X_train.shape[1:]
    model = create_cnn_lstm_model(input_shape, num_classes=34)

    # Trening modelu
    print("Rozpoczynam trening modelu...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

    # Ocena modelu
    print("Oceniam model na zbiorze testowym...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Loss: {loss}, Accuracy: {accuracy}")

    # Raport klasyfikacji
    y_pred = model.predict(X_test).argmax(axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
