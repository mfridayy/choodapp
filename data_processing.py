# data_processing.py

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import json

def load_data(file_or_folder, label_start=0):

    if os.path.isdir(file_or_folder):
        files = [os.path.join(file_or_folder, f) for f in os.listdir(file_or_folder) if f.endswith('.xlsx')]
    else:
        files = [file_or_folder]

    all_data = []
    person_id = label_start

    for file in files:
        print(f"Wczytywanie pliku: {file}")
        data = pd.read_excel(file, header=None)
        data.columns = ["NodeName", "HostTimestamp", "notificationTime", "timeStamp",
                        "RawData", "X", "Y", "Z"]
        data["PersonID"] = person_id
        all_data.append(data)
        person_id += 1

    if not all_data:
        raise ValueError(f"Brak danych do wczytania z {file_or_folder}")
    return pd.concat(all_data, ignore_index=True)


def load_all_data_from_json(person_map_file):
    with open(person_map_file, 'r', encoding='utf-8') as f:
        person_map = json.load(f)

    all_dataframes = []
    for str_person_id, person_info in person_map.items():
        pid = int(str_person_id)
        recordings = person_info.get("recordings", [])
        for recording_path in recordings:
            if not os.path.exists(recording_path):
                print(f"Plik {recording_path} nie istnieje, pomijam...")
                continue
            print(f"Wczytywanie pliku: {recording_path} dla PersonID={pid}")
            df = pd.read_excel(recording_path, header=None)
            df.columns = ["NodeName", "HostTimestamp", "notificationTime",
                          "timeStamp", "RawData", "X", "Y", "Z"]
            df["PersonID"] = pid
            all_dataframes.append(df)

    if not all_dataframes:
        raise ValueError("Brak danych do wczytania z pliku JSON lub nieprawidłowe ścieżki.")

    return pd.concat(all_dataframes, ignore_index=True)


def load_unknown_persons(unknown_folder, unknown_label=9999):
    if not os.path.isdir(unknown_folder):
        print(f"Folder {unknown_folder} nie istnieje!")
        return pd.DataFrame([])  # zwracamy pusty, żeby nie wysypało kodu

    files = [f for f in os.listdir(unknown_folder) if f.endswith(".xlsx")]
    all_data = []

    for f in files:
        full_path = os.path.join(unknown_folder, f)
        print(f"Wczytywanie (UNKNOWN) pliku: {full_path}")
        df = pd.read_excel(full_path, header=None)
        df.columns = ["NodeName", "HostTimestamp", "notificationTime",
                      "timeStamp", "RawData", "X", "Y", "Z"]
        df["PersonID"] = unknown_label
        all_data.append(df)

    if not all_data:
        return pd.DataFrame([])

    return pd.concat(all_data, ignore_index=True)


def interpolate_data(data, target_frequency=100):
    print("Interpolacja danych do częstotliwości 100 Hz...")
    data["Time"] = pd.to_datetime(data["HostTimestamp"], unit='ms')
    data["ElapsedSeconds"] = (data["Time"] - data.groupby("PersonID")["Time"].transform("first")).dt.total_seconds()

    interpolated_data = []
    for person_id, group in data.groupby("PersonID"):
        start_time = group["ElapsedSeconds"].iloc[0]
        end_time = group["ElapsedSeconds"].iloc[-1]
        time_vector = np.arange(start_time, end_time, 1.0 / target_frequency)

        interp_group = pd.DataFrame({"ElapsedSeconds": time_vector})
        for axis in ["X", "Y", "Z"]:
            interp_group[axis] = np.interp(time_vector, group["ElapsedSeconds"], group[axis])
        interp_group["PersonID"] = person_id
        interpolated_data.append(interp_group)

    interpolated_data = pd.concat(interpolated_data, ignore_index=True)
    print("Interpolacja zakończona!")
    return interpolated_data


def highpass_filter(data, cutoff=0.115, fs=100):

    print("Filtracja górnoprzepustowa...")
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='high', analog=False)

    filtered_data = data.copy()
    for axis in ["X", "Y", "Z"]:
        filtered_data[axis] = filtfilt(b, a, data[axis])
    print("Filtracja zakończona!")
    return filtered_data


def trim_data(data, trim_seconds=4):
    print("Przycinanie pierwszych 4 sekund...")
    trimmed_data = []
    for person_id, group in data.groupby("PersonID"):
        group = group[group["ElapsedSeconds"] >= trim_seconds]
        trimmed_data.append(group)

    trimmed_data = pd.concat(trimmed_data, ignore_index=True)
    print("Przycinanie zakończone!")
    return trimmed_data


def sliding_window_segmentation(data, window_size_seconds=2.5, overlap=0.8, sampling_rate=100):
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


def prepare_data_for_model(segments, target_length=200):
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
        y_labels.append(segment["PersonID"].iloc[0])

    X_data = np.array(X_data)
    y_labels = np.array(y_labels)

    # Normalizacja segment-po-segmencie metodą z-score:
    # (X - mean) / std dla każdego segmentu osobno
    X_mean = X_data.mean(axis=1, keepdims=True)
    X_std = X_data.std(axis=1, keepdims=True)
    # Unikamy dzielenia przez 0
    X_std[X_std < 1e-9] = 1e-9
    X_data = (X_data - X_mean) / X_std

    print(f"Kształt danych wejściowych: {X_data.shape}")
    print(f"Kształt etykiet: {y_labels.shape}")
    print(f"Unikalne klasy: {np.unique(y_labels)}")

    return X_data, y_labels
