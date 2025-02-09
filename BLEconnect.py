#BLE_connect.py
import asyncio
from bleak import BleakClient
from datetime import datetime
import time
import pandas as pd

# Adres MAC urządzenia i UUID
DEVICE_ADDRESS = "AE:EC:1A:1F:CB:DE"
CORRECT_UUID = "00a00000-0001-11e1-ac36-0002a5d5c51b"

data_list = []
REQUIRED_COLUMNS = [
    "NodeName", "HostTimestamp", "notificationTime", "timeStamp",
    "RawData", "X", "Y", "Z"
]

start_time = None
last_timestamp = None

def handle_accelerometer_data(sender, data):
    global start_time, last_timestamp

    current_timestamp = time.time()
    if last_timestamp:
        interval = (current_timestamp - last_timestamp) * 1000
        print(f"Interwał: {interval:.2f} ms")
    last_timestamp = current_timestamp

    if len(data) >= 8:
        x = int.from_bytes(data[2:4], byteorder="little", signed=True)
        y = int.from_bytes(data[4:6], byteorder="little", signed=True)
        z = int.from_bytes(data[6:8], byteorder="little", signed=True)
        current_timestamp_ms = int(time.time() * 1000)
        if start_time is None:
            start_time = current_timestamp_ms
        relative_timestamp = current_timestamp_ms - start_time
        record = {
            "NodeName": "CustomDevice",
            "HostTimestamp": relative_timestamp,
            "notificationTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timeStamp": len(data_list),
            "RawData": 0,
            "X": x,
            "Y": y,
            "Z": z
        }
        data_list.append(record)
        print(f"Dodano dane: {record}")
    else:
        print("Nieoczekiwany format danych:", data)

async def connect_and_scan(output_file_path):
    global start_time, data_list
    start_time = None
    data_list = []

    print(f"Rozpoczynam łączenie z urządzeniem {DEVICE_ADDRESS}...")
    async with BleakClient(DEVICE_ADDRESS) as client:
        print(f"Połączono z {DEVICE_ADDRESS}")
        try:
            print(f"Subskrybuję dane z UUID {CORRECT_UUID}...")
            await client.start_notify(CORRECT_UUID, handle_accelerometer_data)
            await asyncio.sleep(120)
            print(f"Zatrzymuję subskrypcję {CORRECT_UUID}")
            await client.stop_notify(CORRECT_UUID)
        except Exception as e:
            print(f"Błąd subskrypcji: {e}")

        # Zapis do Excela
        save_to_excel(data_list, output_file_path)

def save_to_excel(data, file_path):
    if data:
        print("Zapisywanie danych do pliku Excel...")
        df = pd.DataFrame(data)
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = 0
        df = df[REQUIRED_COLUMNS]
        df.to_excel(file_path, index=False, header=False)
        print(f"Dane zapisane do: {file_path}")
    else:
        print("Brak danych do zapisania!")

if __name__ == "__main__":
    # Testowe wywołanie
    output_file = "test_ble.xlsx"
    asyncio.run(connect_and_scan(output_file))

