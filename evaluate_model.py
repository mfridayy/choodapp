import os
import json

from sklearn.metrics import confusion_matrix, classification_report

from constants import UNKNOWN_CLASS_ID, PERSON_ID_MAP_FILE
from model_training import predict_person

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


TEST_KNOWN_FOLDER = os.path.join(BASE_DIR, "test_known_persons")
TEST_UNKNOWN_FOLDER = os.path.join(BASE_DIR, "test_unknown_persons")


def evaluate_model():
    """
    Główny skrypt testujący, zakładający:
      - test_known_persons: pliki .xlsx o nazwie = ID + ".xlsx" (np. "1.xlsx", "2.xlsx")
      - test_unknown_persons: pliki .xlsx o dowolnej nazwie, wszystkie traktowane jako ID=9999
    """
    y_true = []
    y_pred = []

    # --- 1. Testy osób ZNANYCH (ID w nazwie pliku: 1.xlsx, 2.xlsx, itp.) ---
    if not os.path.isdir(TEST_KNOWN_FOLDER):
        print(f"Folder testowy (known) nie istnieje: {TEST_KNOWN_FOLDER}")
        return

    known_files = [f for f in os.listdir(TEST_KNOWN_FOLDER) if f.lower().endswith(".xlsx")]

    for fname in known_files:
        file_path = os.path.join(TEST_KNOWN_FOLDER, fname)
        true_id = parse_file_name_for_id(fname)
        if true_id is None:
            print(f"UWAGA: Nie udało się wyodrębnić ID z nazwy pliku: {fname}. Pomijam ten plik.")
            continue

        # Wywołanie predykcji
        prediction_str = predict_person(
            file_path,
            confidence_threshold=0.8,
            top_diff_threshold=0.05,
            entropy_threshold=2.2,
            debug=False
        )
        pred_id = parse_prediction_to_id(prediction_str)

        y_true.append(true_id)
        y_pred.append(pred_id)

    # --- 2. Testy osób NIEZNANYCH (ID=9999) ---
    if not os.path.isdir(TEST_UNKNOWN_FOLDER):
        print(f"Folder testowy (unknown) nie istnieje: {TEST_UNKNOWN_FOLDER}")
        return

    unknown_files = [f for f in os.listdir(TEST_UNKNOWN_FOLDER) if f.lower().endswith(".xlsx")]

    for fname in unknown_files:
        file_path = os.path.join(TEST_UNKNOWN_FOLDER, fname)
        true_id = UNKNOWN_CLASS_ID

        prediction_str = predict_person(
            file_path,
            confidence_threshold=0.8,
            top_diff_threshold=0.05,
            entropy_threshold=2.2,
            debug=False
        )
        pred_id = parse_prediction_to_id(prediction_str)

        y_true.append(true_id)
        y_pred.append(pred_id)

    # --- 3. Analiza wyników ---
    all_labels = sorted(set(y_true + y_pred))
    print(f"\nEtykiety występujące w teście: {all_labels}")

    # Macierz pomyłek
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    print("\n=== Macierz pomyłek (confusion matrix) ===")
    print(cm)

    # Raport klasyfikacji (precision, recall, f1) per klasa
    print("\n=== Raport klasyfikacji ===")
    print(classification_report(y_true, y_pred, labels=all_labels))

    # Metryki open set: FAR i FRR
    compute_FAR_FRR(y_true, y_pred)


def parse_file_name_for_id(fname):
    """
    Zakładamy, że plik .xlsx w test_known_persons nazywa się np. "1.xlsx".
    Wtedy ID=1.
    Jeśli nazwa nie jest liczbą przed .xlsx, zwraca None.
    """
    base, ext = os.path.splitext(fname)  # np. "1", ".xlsx"
    try:
        return int(base)
    except ValueError:
        return None


def parse_prediction_to_id(prediction_str):
    """
    Zamienia napis zwracany przez `predict_person`
    (np. "Przewidywana osoba: Jan Kowalski" lub "Unknown")
    na ID liczbowe (np. 0,1,...,9999).
    """
    if "Unknown" in prediction_str:
        return UNKNOWN_CLASS_ID

    name_part = prediction_str.replace("Przewidywana osoba:", "").strip()

    # Wczytujemy person_id_map.json, by znaleźć ID dla danego imienia i nazwiska
    with open(PERSON_ID_MAP_FILE, "r", encoding='utf-8') as f:
        person_map = json.load(f)

    for pid_str, info in person_map.items():
        full_name = f"{info.get('first_name','')} {info.get('last_name','')}".strip()
        if full_name == name_part:
            return int(pid_str)

    # Jeśli nie znaleziono – zwracamy unknown
    return UNKNOWN_CLASS_ID


def compute_FAR_FRR(y_true, y_pred):
    """
    Oblicza metryki open set:
      - FAR = false acceptance rate (ile obcych zaklasyfikowano jako known)
      - FRR = false rejection rate (ile znanych zaklasyfikowano jako unknown)
    """
    total_unknown = 0
    false_accepted = 0

    total_known = 0
    false_rejected = 0

    for t, p in zip(y_true, y_pred):
        if t == UNKNOWN_CLASS_ID:
            total_unknown += 1
            if p != UNKNOWN_CLASS_ID:
                false_accepted += 1
        else:
            total_known += 1
            if p == UNKNOWN_CLASS_ID:
                false_rejected += 1

    FAR = false_accepted / total_unknown if total_unknown > 0 else 0.0
    FRR = false_rejected / total_known if total_known > 0 else 0.0

    print("\n=== FAR/FRR (Open Set) ===")
    print(f"  Liczba próbek unknown: {total_unknown}, false_accepted: {false_accepted}, FAR = {FAR:.4f}")
    print(f"  Liczba próbek known:   {total_known}, false_rejected: {false_rejected}, FRR = {FRR:.4f}")


if __name__ == "__main__":
    evaluate_model()
