import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox, simpledialog
import os
import threading
import asyncio
import json
import time

from constants import PERSON_ID_MAP_FILE, MODEL_PATH
from model_training import train_model, predict_person

RECORDINGS_FOLDER = "recordings"
os.makedirs(RECORDINGS_FOLDER, exist_ok=True)

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aplikacja rozpoznawania osoby")
        self.geometry("800x500")

        # ------ STYL TTK (ciemny) ------
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#2e2e2e", foreground="white")
        style.configure("TLabel", background="#2e2e2e", foreground="white")
        style.configure("TButton", background="#4e4e4e", foreground="white")
        style.map("TButton", background=[("active", "#666666")])

        self.configure(bg="#2e2e2e")

        # ------ GÓRNY PASEK (NAVBAR) ------
        self.navbar = ttk.Frame(self, height=40)
        self.navbar.pack(side="top", fill="x")

        train_btn = ttk.Button(self.navbar, text="Trenuj Model", command=self.train_model)
        train_btn.pack(side="left", padx=5, pady=5)

        test_btn = ttk.Button(self.navbar, text="Badaj Osobę", command=self.show_test_frame)
        test_btn.pack(side="left", padx=5, pady=5)

        manage_btn = ttk.Button(self.navbar, text="Zarządzaj Osobami", command=self.show_manage_frame)
        manage_btn.pack(side="left", padx=5, pady=5)

        quit_btn = ttk.Button(self.navbar, text="Zamknij", command=self.quit)
        quit_btn.pack(side="right", padx=5, pady=5)

        self.container = ttk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)

        self.status_label = ttk.Label(self, text="Status: Oczekiwanie na akcję", anchor="w")
        self.status_label.pack(side="bottom", fill="x", pady=5)

        self.home_frame = HomeFrame(self.container, self)
        self.test_person_frame = TestPersonFrame(self.container, self)
        self.manage_persons_frame = ManagePersonsFrame(self.container, self)

        for frame in (self.home_frame, self.test_person_frame, self.manage_persons_frame):
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(self.home_frame)

    def show_frame(self, frame):
        frame.tkraise()

    def show_test_frame(self):
        self.update_status("Otwieranie ekranu: Badaj Osobę")
        self.show_frame(self.test_person_frame)

    def show_manage_frame(self):
        self.update_status("Otwieranie ekranu: Zarządzaj Osobami")
        self.show_frame(self.manage_persons_frame)

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")
        self.update_idletasks()

    def train_model(self):
        self.update_status("Trwa trening modelu...")
        threading.Thread(target=self.run_training).start()

    def run_training(self):
        try:
            train_model(PERSON_ID_MAP_FILE)
            self.update_status("Trening modelu zakończony!")
        except Exception as e:
            self.update_status(f"Błąd: {str(e)}")


class HomeFrame(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        lbl_title = ttk.Label(self, text="Witaj w Aplikacji Rozpoznawania Osoby", font=("Arial", 16, "bold"))
        lbl_title.pack(pady=20)

        lbl_info = ttk.Label(self, text="Wybierz opcję z górnego paska:\n - Trenuj Model\n - Badaj Osobę\n - Zarządzaj Osobami\n", font=("Arial", 12))
        lbl_info.pack(pady=10)


class TestPersonFrame(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        ttk.Label(self, text="Badaj Osobę", font=("Arial", 14, "bold")).pack(pady=15)

        btn_load = ttk.Button(self, text="Załaduj Plik", command=self.load_test_file)
        btn_load.pack(pady=5)

        btn_live = ttk.Button(self, text="Badaj Teraz (na żywo)", command=self.start_testing)
        btn_live.pack(pady=5)

        self.test_status_label = ttk.Label(self, text="", wraplength=600)
        self.test_status_label.pack(pady=20)

    def load_test_file(self):
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if not file_path:
            return
        try:
            # Wywołujemy predict_person z łagodniejszymi progami oraz debug=True
            prediction = predict_person(
                file_path,
                confidence_threshold=0.8,
                top_diff_threshold=0.05,
                entropy_threshold=2.2,
                debug=True
            )
            self.test_status_label.config(text=f"Wynik: {prediction}")
            self.controller.update_status(f"Wynik: {prediction}")
        except Exception as e:
            self.test_status_label.config(text=f"Błąd: {str(e)}")
            self.controller.update_status(f"Błąd: {str(e)}")

    def start_testing(self):
        self.test_status_label.config(text="Trwa nagrywanie danych...")
        threading.Thread(target=self.record_and_test).start()

    def record_and_test(self):
        from BLEconnect import connect_and_scan
        import os

        temp_file = os.path.join(RECORDINGS_FOLDER, "temp_test_data.xlsx")
        try:
            connect_and_scan(temp_file)
            # Tak samo z łagodniejszymi progami + debug
            prediction = predict_person(
                temp_file,
                confidence_threshold=0.8,
                top_diff_threshold=0.05,
                entropy_threshold=2.2,
                debug=True
            )
            self.test_status_label.config(text=f"Wynik: {prediction}")
            self.controller.update_status(f"Wynik: {prediction}")
        except Exception as e:
            self.test_status_label.config(text=f"Błąd: {str(e)}")
            self.controller.update_status(f"Błąd: {str(e)}")


class ManagePersonsFrame(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.selected_person_id = None

        lbl_title = ttk.Label(self, text="Zarządzaj Osobami", font=("Arial", 14, "bold"))
        lbl_title.pack(pady=5)

        list_frame = ttk.Frame(self)
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.person_listbox = tk.Listbox(list_frame, width=40, height=15, bg="#3e3e3e", fg="white")
        self.person_listbox.pack(side="left", fill="both", expand=False)
        self.person_listbox.bind("<<ListboxSelect>>", self.on_person_select)

        self.recordings_listbox = tk.Listbox(list_frame, width=50, height=15, bg="#3e3e3e", fg="white")
        self.recordings_listbox.pack(side="right", fill="both", expand=True)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(side="bottom", pady=5)

        btn_refresh = ttk.Button(btn_frame, text="Odśwież listę", command=self.load_persons)
        btn_refresh.grid(row=0, column=0, padx=5, pady=5)

        btn_add_person = ttk.Button(btn_frame, text="Dodaj nową osobę", command=self.add_person)
        btn_add_person.grid(row=0, column=1, padx=5, pady=5)

        btn_edit_person = ttk.Button(btn_frame, text="Edytuj osobę", command=self.edit_person)
        btn_edit_person.grid(row=0, column=2, padx=5, pady=5)

        btn_delete_person = ttk.Button(btn_frame, text="Usuń osobę", command=self.delete_person)
        btn_delete_person.grid(row=0, column=3, padx=5, pady=5)

        btn_record_ble = ttk.Button(btn_frame, text="Dodaj nowe nagranie", command=self.add_new_recording)
        btn_record_ble.grid(row=0, column=4, padx=5, pady=5)

        btn_delete_recording = ttk.Button(btn_frame, text="Usuń nagranie", command=self.delete_recording)
        btn_delete_recording.grid(row=0, column=5, padx=5, pady=5)

        self.load_persons()

    def delete_recording(self):
        pid = self.get_selected_pid()
        if pid is None:
            messagebox.showwarning("Brak wyboru", "Wybierz osobę z listy.")
            return

        selected_recording = self.recordings_listbox.curselection()
        if not selected_recording:
            messagebox.showwarning("Brak wyboru", "Wybierz nagranie do usunięcia.")
            return

        recording_index = selected_recording[0]
        recording_name = self.recordings_listbox.get(recording_index)

        confirm = messagebox.askyesno(
            "Potwierdzenie",
            f"Czy na pewno chcesz usunąć nagranie:\n{recording_name}?"
        )
        if not confirm:
            return

        try:
            with open(PERSON_ID_MAP_FILE, 'r', encoding='utf-8') as f:
                person_map = json.load(f)

            recordings = person_map[pid].get("recordings", [])
            if recording_name in recordings:
                recordings.remove(recording_name)
                person_map[pid]["recordings"] = recordings

                with open(PERSON_ID_MAP_FILE, 'w', encoding='utf-8') as f:
                    json.dump(person_map, f, ensure_ascii=False, indent=4)

                self.load_persons()
                messagebox.showinfo("Sukces", f"Nagranie \"{recording_name}\" zostało usunięte.")
            else:
                messagebox.showerror("Błąd", "Nie znaleziono nagrania w danych osoby.")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się usunąć nagrania: {e}")

    def add_new_recording(self):
        pid = self.get_selected_pid()
        if pid is None:
            messagebox.showwarning("Brak wyboru", "Wybierz osobę, aby dodać nowe nagranie.")
            return

        response = messagebox.askyesno(
            "Dodaj nowe nagranie",
            "Czy chcesz utworzyć NOWE nagranie z czujnika BLE?\n"
            "Tak – rozpocznie się nagrywanie.\n"
            "Nie – wybierzesz gotowy plik z dysku."
        )

        if response:
            dialog = tk.Toplevel(self)
            dialog.title("Dodaj nowe nagranie")
            dialog.geometry("300x200")
            dialog.transient(self)
            dialog.grab_set()

            tk.Label(dialog, text="Podaj imię:").pack(pady=(20, 5))
            first_name_entry = ttk.Entry(dialog)
            first_name_entry.configure(foreground="black", background="white")
            first_name_entry.pack(pady=5)

            tk.Label(dialog, text="Podaj nazwisko:").pack(pady=(10, 5))
            last_name_entry = ttk.Entry(dialog)
            last_name_entry.configure(foreground="black", background="white")
            last_name_entry.pack(pady=5)

            def submit():
                first_name = first_name_entry.get().strip()
                last_name = last_name_entry.get().strip()
                if not first_name or not last_name:
                    messagebox.showerror("Błąd", "Musisz podać zarówno imię, jak i nazwisko.")
                    return

                filename = f"{first_name}_{last_name}.xlsx".replace(" ", "_")
                file_path = os.path.join(RECORDINGS_FOLDER, filename)

                dialog.destroy()
                threading.Thread(target=self.run_ble_recording, args=(pid, file_path)).start()

                messagebox.showinfo(
                    "Nagrywanie",
                    f"Rozpoczynam nagrywanie do pliku: {file_path}\n\nCzekaj na zakończenie..."
                )

            ttk.Button(dialog, text="OK", command=submit).pack(pady=(20, 10))
            ttk.Button(dialog, text="Anuluj", command=dialog.destroy).pack()

        else:
            from tkinter import filedialog
            file_path = filedialog.askopenfilename(
                filetypes=[("Excel files", "*.xlsx")]
            )
            if not file_path:
                return

            try:
                with open(PERSON_ID_MAP_FILE, 'r', encoding='utf-8') as f:
                    person_map = json.load(f)

                person_map[pid].setdefault("recordings", []).append(file_path)

                with open(PERSON_ID_MAP_FILE, 'w', encoding='utf-8') as f:
                    json.dump(person_map, f, ensure_ascii=False, indent=4)

                self.load_persons()
                messagebox.showinfo(
                    "Sukces",
                    f"Plik '{file_path}' został dodany do nagrań osoby o ID {pid}."
                )
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się zapisać pliku: {e}")

    def run_ble_recording(self, pid, file_path):
        try:
            from BLEconnect import connect_and_scan
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(connect_and_scan(file_path))

            with open(PERSON_ID_MAP_FILE, 'r', encoding='utf-8') as f:
                person_map = json.load(f)

            person_map[pid].setdefault("recordings", []).append(file_path)

            with open(PERSON_ID_MAP_FILE, 'w', encoding='utf-8') as f:
                json.dump(person_map, f, ensure_ascii=False, indent=4)

            self.load_persons()
            messagebox.showinfo("Sukces", f"Nagranie zapisane: {file_path}")
        except Exception as e:
            messagebox.showerror("Błąd", f"Błąd nagrywania BLE: {str(e)}")

    def load_persons(self):
        self.selected_person_id = None
        self.person_listbox.delete(0, tk.END)
        self.recordings_listbox.delete(0, tk.END)
        try:
            with open(PERSON_ID_MAP_FILE, 'r', encoding='utf-8') as f:
                person_map = json.load(f)
            sorted_keys = sorted(person_map.keys(), key=lambda x: int(x))
            for pid in sorted_keys:
                info = person_map[pid]
                fname = info.get("first_name", "Unknown")
                lname = info.get("last_name", "")
                display = f"ID: {pid} - {fname} {lname}"
                self.person_listbox.insert(tk.END, display)
        except FileNotFoundError:
            messagebox.showerror("Błąd", "Nie znaleziono pliku person_id_map.json")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się wczytać danych: {e}")

    def on_person_select(self, event):
        selection = self.person_listbox.curselection()
        if not selection:
            return

        self.recordings_listbox.delete(0, tk.END)
        index = selection[0]
        person_entry = self.person_listbox.get(index)
        pid = person_entry.split(" ")[1]  # np. "ID: 1 - John Doe" -> "1"
        self.selected_person_id = pid

        try:
            with open(PERSON_ID_MAP_FILE, 'r', encoding='utf-8') as f:
                person_map = json.load(f)
            recs = person_map[pid].get("recordings", [])
            if not recs:
                self.recordings_listbox.insert(tk.END, "(Brak nagrań)")
            else:
                for r in recs:
                    self.recordings_listbox.insert(tk.END, r)
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można wczytać nagrań: {e}")

    def get_selected_pid(self):
        return self.selected_person_id

    def add_person(self):
        fname = simpledialog.askstring("Imię", "Podaj imię:")
        lname = simpledialog.askstring("Nazwisko", "Podaj nazwisko:")
        if not fname or not lname:
            return
        try:
            with open(PERSON_ID_MAP_FILE, 'r', encoding='utf-8') as f:
                person_map = json.load(f)
        except FileNotFoundError:
            person_map = {}

        if person_map:
            all_ids = list(map(int, person_map.keys()))
            new_id = max(all_ids) + 1
        else:
            new_id = 0

        person_map[str(new_id)] = {
            "first_name": fname,
            "last_name": lname,
            "recordings": []
        }

        try:
            with open(PERSON_ID_MAP_FILE, 'w', encoding='utf-8') as f:
                json.dump(person_map, f, ensure_ascii=False, indent=4)
            self.load_persons()
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się zapisać: {e}")

    def edit_person(self):
        pid = self.get_selected_pid()
        if pid is None:
            return
        with open(PERSON_ID_MAP_FILE, 'r', encoding='utf-8') as f:
            person_map = json.load(f)
        info = person_map[pid]

        new_fname = simpledialog.askstring("Edytuj imię", "Podaj nowe imię:", initialvalue=info.get("first_name", ""))
        new_lname = simpledialog.askstring("Edytuj nazwisko", "Podaj nowe nazwisko:", initialvalue=info.get("last_name", ""))
        if not new_fname or not new_lname:
            return

        person_map[pid]["first_name"] = new_fname
        person_map[pid]["last_name"] = new_lname

        try:
            with open(PERSON_ID_MAP_FILE, 'w', encoding='utf-8') as f:
                json.dump(person_map, f, ensure_ascii=False, indent=4)
            self.load_persons()
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się zapisać: {e}")

    def delete_person(self):
        pid = self.get_selected_pid()
        if pid is None:
            return
        confirm = messagebox.askyesno("Potwierdzenie", f"Czy na pewno chcesz usunąć osobę ID {pid}?")
        if not confirm:
            return
        with open(PERSON_ID_MAP_FILE, 'r', encoding='utf-8') as f:
            person_map = json.load(f)
        if pid not in person_map:
            return
        del person_map[pid]
        try:
            with open(PERSON_ID_MAP_FILE, 'w', encoding='utf-8') as f:
                json.dump(person_map, f, ensure_ascii=False, indent=4)
            self.load_persons()
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się zapisać: {e}")


if __name__ == "__main__":
    app = Application()
    app.mainloop()
