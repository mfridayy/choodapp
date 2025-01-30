# constants.py
# Stałe używane w aplikacji

# Ścieżki do folderów i plików
DATA_FOLDER = "C:/Users/Mateusz/PycharmProjects/inżynierka/data/known_persons"
NEW_PERSONS_FOLDER = "C:/Users/Mateusz/PycharmProjects/inżynierka/data/new_persons"
MODEL_PATH = "models/cnn_lstm_model.keras"

# Nowo dodana stała - ścieżka do pliku JSON z mapowaniem ID → {imię, nazwisko, nagrania}
PERSON_ID_MAP_FILE = "C:/Users/Mateusz/PycharmProjects/inżynierka/person_id_map.json"

# BLE ustawienia (jeśli są potrzebne)
BLE_DEVICE_ADDRESS = "AE:EC:1A:1F:CB:DE"
BLE_CORRECT_UUID = "00a00000-0001-11e1-ac36-0002a5d5c51b"
BLE_SCAN_DURATION = 120  # Czas skanowania w sekundach

# Parametry przetwarzania danych
TARGET_LENGTH = 200    # Liczba próbek w segmencie
WINDOW_SIZE_SECONDS = 2  # Długość okna w sekundach
OVERLAP = 0.8          # Nakładanie się okien w ułamku (np. 0.2 = 20%)
SAMPLING_RATE = 100    # Częstotliwość próbkowania w Hz

# *** NOWE STAŁE DLA „UNKNOWN” ***
UNKNOWN_PERSONS_FOLDER = "C:/Users/Mateusz/PycharmProjects/inżynierka/unknown_persons"
UNKNOWN_CLASS_ID = 9999
