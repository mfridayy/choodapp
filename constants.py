# constants.py
import os
MODEL_PATH = "models/cnn_lstm_model.keras"
PERSON_ID_MAP_FILE = "person_id_map.json"

# BLE ustawienia
BLE_DEVICE_ADDRESS = "AE:EC:1A:1F:CB:DE"
BLE_CORRECT_UUID = "00a00000-0001-11e1-ac36-0002a5d5c51b"
BLE_SCAN_DURATION = 120

# Parametry przetwarzania danych
TARGET_LENGTH = 200
WINDOW_SIZE_SECONDS = 2
OVERLAP = 0.8
SAMPLING_RATE = 100

# Stałe dla nieznanych osób
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNKNOWN_PERSONS_FOLDER = os.path.join(BASE_DIR, "unknown_persons")
UNKNOWN_CLASS_ID = 9999
