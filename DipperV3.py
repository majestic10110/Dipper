import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, simpledialog
import pyaudio
import numpy as np
from scipy.fft import fft
import threading
import time
import os
from reedsolo import RSCodec, ReedSolomonError

CHAR_SOUNDS = {
    "A": [(2000, 2000, "tone")], "B": [(3000, 3000, "trill")], "C": [(4000, 2000, "slide")],
    "D": [(1500, 1500, "tone")], "E": [(5000, 5000, "tone")], "F": [(2500, 3500, "slide")],
    "G": [(1000, 1000, "trill")], "H": [(3500, 3500, "tone")], "I": [(4500, 4500, "tone")],
    "J": [(2000, 4000, "slide")], "K": [(3000, 3000, "tone")], "L": [(4000, 4000, "trill")],
    "M": [(1500, 2500, "slide")], "N": [(5000, 3000, "slide")], "O": [(1000, 1000, "tone")],
    "P": [(3500, 3500, "trill")], "Q": [(2500, 2500, "tone")], "R": [(2000, 2000, "trill")],
    "S": [(4500, 1500, "slide")], "T": [(3000, 5000, "slide")], "U": [(4000, 4000, "tone")],
    "V": [(1500, 1500, "trill")], "W": [(1000, 2000, "slide")], "X": [(5000, 5000, "trill")],
    "Y": [(3500, 2500, "slide")], "Z": [(2500, 2500, "trill")],
    "0": [(1000, 1500, "slide")], "1": [(1500, 2000, "slide")], "2": [(2000, 2500, "slide")],
    "3": [(2500, 3000, "slide")], "4": [(3000, 3500, "slide")], "5": [(3500, 4000, "slide")],
    "6": [(4000, 4500, "slide")], "7": [(4500, 5000, "slide")], "8": [(5000, 4500, "slide")],
    "9": [(2000, 1500, "slide")],
    "!": [(5000, 1000, "slide")], "/": [(3000, 4000, "slide")], "-": [(2000, 2000, "trill")],
    ".": [(4500, 4500, "tone")], " ": [(1000, 1000, "tone")]
}

WORD_SOUNDS = {
    "CQ": [(2000, 4000, "slide")],
    "HTTPS://WWW.": [(1000, 1000, "tone"), (3000, 3000, "tone"), (5000, 5000, "trill")],
    "DE": [(1500, 2500, "slide")],
    "the": [(1100, 1100, "tone")],
    "and": [(1150, 1150, "trill")],
    "to": [(1200, 2200, "slide")],
    "of": [(1250, 1250, "tone")],
    "you": [(1300, 1300, "tone")],
    "I": [(1350, 1350, "tone")],
    "in": [(1400, 1400, "trill")],
    "that": [(1450, 1450, "tone")],
    "it": [(1550, 1550, "tone")],
    "he": [(1600, 2600, "slide")],
    "for": [(1650, 1650, "tone")],
    "with": [(1700, 2700, "slide")],
    "on": [(1750, 1750, "trill")],
    "they": [(1800, 1800, "tone")],
    "have": [(1850, 2850, "slide")],
    "from": [(1900, 1900, "tone")],
    "who": [(1950, 1950, "trill")],
    "this": [(2050, 2050, "tone")],
    "what": [(2100, 3100, "slide")],
    "say": [(2150, 2150, "tone")],
    "K": [(2200, 2200, "tone")],
    "QTH": [(2250, 3250, "slide")],
    "73": [(2300, 2300, "trill")],
    "QSL": [(2350, 2350, "tone")],
    "DX": [(2400, 3400, "slide")],
    "SSB": [(2450, 2450, "tone")],
    "FM": [(2550, 2550, "trill")],
    "AM": [(2600, 2600, "tone")],
    "RST": [(2650, 3650, "slide")],
    "QSO": [(2700, 2700, "trill")],
    "OM": [(2750, 2750, "tone")],
    "YL": [(2800, 2800, "tone")],
    "Antenna": [(2850, 3850, "slide")],
    "Frequency": [(2900, 2900, "tone")],
    "Power": [(2950, 2950, "trill")],
    "Weather": [(3050, 2050, "slide")],
    "Conditions": [(3100, 3100, "tone")],
    "Hello": [(3150, 3150, "trill")],
    "<html>": [(3200, 4200, "slide")],
    "<head>": [(3250, 3250, "tone")],
    "<title>": [(3300, 4300, "trill")],
    "<body>": [(3350, 3350, "tone")],
    "<meta>": [(3400, 3400, "trill")],
    "<link>": [(3450, 4450, "slide")],
    "<script>": [(3500, 3500, "tone")],
    "<style>": [(3550, 3550, "trill")],
    "<div>": [(3600, 4600, "slide")],
    "<span>": [(3650, 3650, "tone")],
    "<p>": [(3700, 3700, "trill")],
    "<a>": [(3750, 4750, "slide")],
    "<img>": [(3800, 3800, "tone")],
    "<ul>": [(3850, 3850, "trill")],
    "<li>": [(3900, 4900, "slide")],
    "<table>": [(3950, 3950, "tone")],
    "<tr>": [(4050, 4050, "trill")],
    "<td>": [(4100, 4100, "tone")],
    "<form>": [(4150, 3150, "slide")],
    "<input>": [(4200, 4200, "trill")]
}

SAMPLE_RATE = 44100
CHUNK = 1024
CALLSIGN_FILE = os.path.join(os.path.dirname(__file__), "mycallsign.txt")

SYMBOL_MAP = {**{char: i for i, char in enumerate(CHAR_SOUNDS.keys())},
              **{word: i + len(CHAR_SOUNDS) for i, word in enumerate(WORD_SOUNDS.keys())}}
REVERSE_MAP = {i: char_or_word for char_or_word, i in SYMBOL_MAP.items()}
RS_CODEC = RSCodec(8)  # RS(31,23) in GF(2^8)

class DipperModeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DIPPER V3 by M0OLI")
        self.root.geometry("800x600")

        self.light_colors = {"bg": "#FFFFFF", "fg": "#000000", "entry_bg": "#F0F0F0", "button_bg": "#D0D0D0"}
        self.dark_colors = {"bg": "#000000", "fg": "#00FF00", "entry_bg": "#1A1A1A", "button_bg": "#333333"}
        self.current_colors = self.light_colors

        self.my_callsign_value = self.load_callsign()
        if not self.my_callsign_value:
            self.my_callsign_value = self.prompt_callsign()
            self.save_callsign(self.my_callsign_value)

        self.p = pyaudio.PyAudio()
        self.input_devices, self.output_devices = self.get_audio_devices()
        self.input_device_index = tk.IntVar(value=-1)
        self.output_device_index = tk.IntVar(value=-1)
        self.input_volume = tk.DoubleVar(value=50.0)
        self.speed_var = tk.StringVar(value="slow")  # Default to slow
        self.stream_out = None
        self.stream_in = None
        self.running = False
        self.rx_thread = None

        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)
        self.settings_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Settings", menu=self.settings_menu)

        self.input_menu = tk.Menu(self.settings_menu, tearoff=0)
        self.settings_menu.add_cascade(label="Audio Input", menu=self.input_menu)
        for index, name in self.input_devices.items():
            self.input_menu.add_radiobutton(label=name, variable=self.input_device_index, value=index, command=self.update_audio)

        self.output_menu = tk.Menu(self.settings_menu, tearoff=0)
        self.settings_menu.add_cascade(label="Audio Output", menu=self.output_menu)
        for index, name in self.output_devices.items():
            self.output_menu.add_radiobutton(label=name, variable=self.output_device_index, value=index, command=self.update_audio)

        self.settings_menu.add_command(label="User Callsign", command=self.edit_callsign)
        self.settings_menu.add_command(label="Audio Input Volume", command=self.show_volume_slider)

        self.main_frame = tk.Frame(root, bg=self.current_colors["bg"])
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.callsign_frame = tk.Frame(self.main_frame, bg=self.current_colors["bg"])
        self.callsign_frame.pack(fill="x", pady=(0, 20))
        
        tk.Label(self.callsign_frame, text="My Callsign:", fg=self.current_colors["fg"], bg=self.current_colors["bg"]).grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.my_callsign = tk.Entry(self.callsign_frame, width=15, fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], insertbackground=self.current_colors["fg"])
        self.my_callsign.grid(row=0, column=1, padx=5, pady=5)
        if self.my_callsign_value:
            self.my_callsign.insert(0, self.my_callsign_value)
            self.my_callsign.config(state="disabled")

        tk.Label(self.callsign_frame, text="To Callsign:", fg=self.current_colors["fg"], bg=self.current_colors["bg"]).grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.to_callsign = tk.Entry(self.callsign_frame, width=15, fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], insertbackground=self.current_colors["fg"])
        self.to_callsign.grid(row=0, column=3, padx=5, pady=5)

        self.dark_mode_var = tk.BooleanVar()
        self.dark_mode_check = tk.Checkbutton(self.callsign_frame, text="Dark Mode", variable=self.dark_mode_var, command=self.toggle_dark_mode, fg=self.current_colors["fg"], bg=self.current_colors["bg"], selectcolor=self.current_colors["entry_bg"])
        self.dark_mode_check.grid(row=0, column=4, padx=20, pady=5)

        self.tx_frame = tk.Frame(self.main_frame, bg=self.current_colors["bg"])
        self.tx_frame.pack(fill="x", pady=(0, 20))
        
        tk.Label(self.tx_frame, text="Transmit Message:", fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(anchor="w", pady=(0, 5))
        self.tx_input = tk.Entry(self.tx_frame, width=60, fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], insertbackground=self.current_colors["fg"])
        self.tx_input.pack(fill="x", pady=(0, 5))
        
        self.button_frame = tk.Frame(self.tx_frame, bg=self.current_colors["bg"])
        self.button_frame.pack(anchor="w", pady=(0, 5))
        self.tx_button = tk.Button(self.button_frame, text="Send", command=self.transmit, fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
        self.tx_button.pack(side="left", padx=5)
        self.cq_button = tk.Button(self.button_frame, text="CQ", command=self.send_cq, fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
        self.cq_button.pack(side="left", padx=5)

        self.speed_frame = tk.LabelFrame(self.tx_frame, text="Speed", fg=self.current_colors["fg"], bg=self.current_colors["bg"])
        self.speed_frame.pack(anchor="w", pady=5)
        tk.Radiobutton(self.speed_frame, text="Slow (~6-13 WPM)", variable=self.speed_var, value="slow", fg=self.current_colors["fg"], bg=self.current_colors["bg"], selectcolor=self.current_colors["entry_bg"]).pack(side="left", padx=5)
        tk.Radiobutton(self.speed_frame, text="Medium (~10-20 WPM)", variable=self.speed_var, value="medium", fg=self.current_colors["fg"], bg=self.current_colors["bg"], selectcolor=self.current_colors["entry_bg"]).pack(side="left", padx=5)
        tk.Radiobutton(self.speed_frame, text="Fast (~15-30 WPM)", variable=self.speed_var, value="fast", fg=self.current_colors["fg"], bg=self.current_colors["bg"], selectcolor=self.current_colors["entry_bg"]).pack(side="left", padx=5)
        tk.Radiobutton(self.speed_frame, text="Turbo (~40+ WPM)", variable=self.speed_var, value="turbo", fg=self.current_colors["fg"], bg=self.current_colors["bg"], selectcolor=self.current_colors["entry_bg"]).pack(side="left", padx=5)

        self.rx_frame = tk.Frame(self.main_frame, bg=self.current_colors["bg"])
        self.rx_frame.pack(fill="both", expand=True)
        
        tk.Label(self.rx_frame, text="Received Messages:", fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(anchor="w", pady=(0, 5))
        self.rx_output = scrolledtext.ScrolledText(self.rx_frame, width=70, height=20, fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], insertbackground=self.current_colors["fg"])
        self.rx_output.pack(fill="both", expand=True)
        self.rx_output.tag_config("sent", foreground="red")
        self.rx_output.tag_config("underline", underline=True)

        self.clear_button = tk.Button(self.rx_frame, text="Clear", command=self.clear_receive, fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
        self.clear_button.pack(side="left", padx=5, pady=5)

        self.start_audio()

    def get_audio_devices(self):
        input_devices = {}
        output_devices = {}
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            name = dev["name"]
            index = dev["index"]
            if dev["maxInputChannels"] > 0:
                input_devices[index] = name
            if dev["maxOutputChannels"] > 0:
                output_devices[index] = name
        return input_devices, output_devices

    def start_audio(self):
        try:
            if self.stream_out:
                self.stream_out.stop_stream()
                self.stream_out.close()
            if self.stream_in:
                self.stream_in.stop_stream()
                self.stream_in.close()
            
            input_idx = self.input_device_index.get() if self.input_device_index.get() >= 0 else None
            output_idx = self.output_device_index.get() if self.output_device_index.get() >= 0 else None
            
            self.stream_out = self.p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, output=True, output_device_index=output_idx)
            self.stream_in = self.p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK, input_device_index=input_idx)
            self.running = True
            if not self.rx_thread or not self.rx_thread.is_alive():
                self.rx_thread = threading.Thread(target=self.receive_loop, daemon=True)
                self.rx_thread.start()
            self.rx_output.insert(tk.END, "Audio started successfully\n")
        except Exception as e:
            self.rx_output.insert(tk.END, f"Audio failed to start: {str(e)}\n")
            self.running = False
            messagebox.showwarning("Audio Warning", f"Audio setup failed: {str(e)}. Continuing without audio.")

    def update_audio(self):
        self.start_audio()

    def edit_callsign(self):
        new_callsign = simpledialog.askstring("User Callsign", "Enter your callsign:", initialvalue=self.my_callsign_value, parent=self.root)
        if new_callsign:
            self.my_callsign_value = new_callsign.upper()
            self.save_callsign(self.my_callsign_value)
            self.my_callsign.config(state="normal")
            self.my_callsign.delete(0, tk.END)
            self.my_callsign.insert(0, self.my_callsign_value)
            self.my_callsign.config(state="disabled")
            self.rx_output.insert(tk.END, f"Callsign updated to: {self.my_callsign_value}\n")

    def show_volume_slider(self):
        volume_window = tk.Toplevel(self.root)
        volume_window.title("Audio Input Volume")
        volume_window.geometry("300x100")
        volume_window.configure(bg=self.current_colors["bg"])

        tk.Label(volume_window, text="Input Volume:", fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(pady=5)
        slider = tk.Scale(volume_window, from_=0, to=100, orient="horizontal", variable=self.input_volume, fg=self.current_colors["fg"], bg=self.current_colors["bg"])
        slider.pack(pady=5)

    def clear_receive(self):
        self.rx_output.delete(1.0, tk.END)

    def load_callsign(self):
        if os.path.exists(CALLSIGN_FILE):
            with open(CALLSIGN_FILE, "r") as f:
                return f.read().strip()
        return ""

    def save_callsign(self, callsign):
        with open(CALLSIGN_FILE, "w") as f:
            f.write(callsign)

    def prompt_callsign(self):
        callsign = simpledialog.askstring("Setup", "What is your callsign?", parent=self.root)
        return callsign.upper() if callsign else ""

    def toggle_dark_mode(self):
        if self.dark_mode_var.get():
            self.current_colors = self.dark_colors
        else:
            self.current_colors = self.light_colors
        
        self.root.configure(bg=self.current_colors["bg"])
        self.main_frame.configure(bg=self.current_colors["bg"])
        self.callsign_frame.configure(bg=self.current_colors["bg"])
        self.tx_frame.configure(bg=self.current_colors["bg"])
        self.rx_frame.configure(bg=self.current_colors["bg"])
        self.speed_frame.configure(bg=self.current_colors["bg"])
        
        for widget in (self.callsign_frame.winfo_children() + 
                       self.tx_frame.winfo_children() + 
                       self.rx_frame.winfo_children() + 
                       self.speed_frame.winfo_children()):
            if isinstance(widget, (tk.Label, tk.Checkbutton, tk.Radiobutton)):
                widget.configure(fg=self.current_colors["fg"], bg=self.current_colors["bg"])
            elif isinstance(widget, tk.Entry):
                widget.configure(fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], insertbackground=self.current_colors["fg"])
            elif isinstance(widget, tk.Button):
                widget.configure(fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
        self.rx_output.configure(fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], insertbackground=self.current_colors["fg"])
        self.clear_button.configure(fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
        self.dark_mode_check.configure(selectcolor=self.current_colors["entry_bg"])

    def generate_sound(self, pattern):
        speed = self.speed_var.get()
        duration_map = {"slow": 0.2, "medium": 0.15, "fast": 0.1, "turbo": 0.03}
        duration = duration_map[speed]
        audio = np.array([], dtype=np.float32)
        if not pattern:
            return np.zeros(int(SAMPLE_RATE * duration), dtype=np.float32)
        for start_freq, end_freq, sound_type in pattern:
            t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
            if sound_type == "tone":
                signal = np.sin(2 * np.pi * start_freq * t)
            elif sound_type == "slide":
                freqs = np.linspace(start_freq, end_freq, len(t))
                signal = np.sin(2 * np.pi * freqs * t)
            elif sound_type == "trill":
                signal = np.sin(2 * np.pi * start_freq * t) * np.sin(2 * np.pi * 20 * t)
            audio = np.concatenate((audio, signal))
        return audio / np.max(np.abs(audio))

    def encode_fec(self, text):
        symbol_list = []
        words = text.split()
        for word in words:
            if word in WORD_SOUNDS:
                symbol_list.append(SYMBOL_MAP[word])
            else:
                for char in word:
                    symbol_list.append(SYMBOL_MAP.get(char, SYMBOL_MAP[" "]))
        
        speed = self.speed_var.get()
        if speed in ["slow", "medium"]:
            while len(symbol_list) % 23 != 0:
                symbol_list.append(SYMBOL_MAP[" "])
            blocks = [symbol_list[i:i+23] for i in range(0, len(symbol_list), 23)]
            encoded_blocks = []
            for block in blocks:
                encoded = RS_CODEC.encode(block)
                encoded_blocks.extend(encoded)
            return encoded_blocks
        return symbol_list  # No FEC for fast or turbo

    def decode_fec(self, symbols):
        speed = self.speed_var.get()
        if speed in ["slow", "medium"]:
            blocks = [symbols[i:i+31] for i in range(0, len(symbols), 31)]
            decoded_text = ""
            for block in blocks:
                if len(block) == 31:
                    try:
                        decoded = RS_CODEC.decode(block)[0]
                        decoded_text += "".join(REVERSE_MAP.get(sym, " ") for sym in decoded)
                    except ReedSolomonError:
                        decoded_text += "[ERROR] "
            return decoded_text.strip()
        return "".join(REVERSE_MAP.get(sym, " ") for sym in symbols).strip()

    def transmit(self):
        if not self.stream_out:
            messagebox.showwarning("Warning", "Audio not started!")
            return
        to_call = self.to_callsign.get().upper()
        my_call = self.my_callsign.get().upper()
        if not to_call or not my_call:
            messagebox.showwarning("Warning", "Enter both callsigns!")
            return
        if not self.my_callsign_value:
            self.my_callsign_value = my_call
            self.save_callsign(my_call)
            self.my_callsign.config(state="disabled")
        text = self.tx_input.get().strip().upper()
        preamble = f"{to_call} DE {my_call}"
        full_text = preamble + " " + text if text else preamble

        encoded_symbols = self.encode_fec(full_text)
        audio_data = np.array([], dtype=np.float32)
        speed = self.speed_var.get()
        gap_map = {"slow": 0.1, "medium": 0.075, "fast": 0.05, "turbo": 0.015}
        gap_duration = gap_map[speed]
        
        for symbol in encoded_symbols:
            char = REVERSE_MAP.get(symbol, " ")
            if char in WORD_SOUNDS:
                pattern = WORD_SOUNDS[char]
            elif char in CHAR_SOUNDS:
                pattern = CHAR_SOUNDS[char]
            else:
                pattern = []
            sound = self.generate_sound(pattern)
            audio_data = np.concatenate((audio_data, sound, np.zeros(int(SAMPLE_RATE * gap_duration))))
            audio_data = np.concatenate((audio_data, sound, np.zeros(int(SAMPLE_RATE * gap_duration))))

        self.stream_out.write(audio_data.astype(np.float32).tobytes())
        
        self.rx_output.insert(tk.END, "Sent: ")
        words = full_text.split()
        for word in words:
            start_idx = self.rx_output.index(tk.END)
            self.rx_output.insert(tk.END, word + " ")
            if word in WORD_SOUNDS:
                end_idx = self.rx_output.index(tk.END)
                self.rx_output.tag_add("underline", start_idx, f"{end_idx} - 1c")
        self.rx_output.insert(tk.END, "\n", "sent")
        self.rx_output.see(tk.END)
        self.tx_input.delete(0, tk.END)

    def send_cq(self):
        if not self.stream_out:
            messagebox.showwarning("Warning", "Audio not started!")
            return
        my_call = self.my_callsign.get().upper()
        if not my_call:
            messagebox.showwarning("Warning", "Enter your callsign!")
            return
        if not self.my_callsign_value:
            self.my_callsign_value = my_call
            self.save_callsign(my_call)
            self.my_callsign.config(state="disabled")
        
        single_cq = f"CQ CQ CQ DE {my_call}"
        cq_text = f"{single_cq} {single_cq} {single_cq}"
        
        encoded_symbols = self.encode_fec(cq_text)
        audio_data = np.array([], dtype=np.float32)
        speed = self.speed_var.get()
        gap_map = {"slow": 0.1, "medium": 0.075, "fast": 0.05, "turbo": 0.015}
        gap_duration = gap_map[speed]
        
        for i, symbol in enumerate(encoded_symbols):
            char = REVERSE_MAP.get(symbol, " ")
            if char in WORD_SOUNDS:
                pattern = WORD_SOUNDS[char]
            elif char in CHAR_SOUNDS:
                pattern = CHAR_SOUNDS[char]
            else:
                pattern = []
            sound = self.generate_sound(pattern)
            audio_data = np.concatenate((audio_data, sound, np.zeros(int(SAMPLE_RATE * gap_duration))))
            audio_data = np.concatenate((audio_data, sound, np.zeros(int(SAMPLE_RATE * gap_duration))))
            if (i + 1) % 5 == 0 and i < len(encoded_symbols) - 1 and speed in ["slow", "medium"]:
                audio_data = np.concatenate((audio_data, np.zeros(int(SAMPLE_RATE * 0.2))))

        self.stream_out.write(audio_data.astype(np.float32).tobytes())
        
        self.rx_output.insert(tk.END, "Sent: ")
        words = cq_text.split()
        for word in words:
            start_idx = self.rx_output.index(tk.END)
            self.rx_output.insert(tk.END, word + " ")
            if word in WORD_SOUNDS:
                end_idx = self.rx_output.index(tk.END)
                self.rx_output.tag_add("underline", start_idx, f"{end_idx} - 1c")
        self.rx_output.insert(tk.END, "\n", "sent")
        self.rx_output.see(tk.END)

    def decode_audio(self, data):
        freqs = np.abs(fft(data))
        freq_axis = np.linspace(0, SAMPLE_RATE, len(freqs))
        peak_idx = np.argmax(freqs[:len(freqs)//2])
        peak_freq = freq_axis[peak_idx]
        for word, pattern in WORD_SOUNDS.items():
            start_freq = pattern[0][0]
            if abs(peak_freq - start_freq) < 200:
                return SYMBOL_MAP[word]
        for char, pattern in CHAR_SOUNDS.items():
            start_freq = pattern[0][0]
            if abs(peak_freq - start_freq) < 200:
                return SYMBOL_MAP[char]
        return None

    def receive_loop(self):
        buffer = []
        gap_map = {"slow": 0.1, "medium": 0.075, "fast": 0.05, "turbo": 0.015}
        duration_map = {"slow": 0.2, "medium": 0.15, "fast": 0.1, "turbo": 0.03}
        while self.running:
            try:
                speed = self.speed_var.get()
                duration = duration_map[speed]
                chunk_size = int(SAMPLE_RATE * (duration + gap_map[speed] * 2))
                data = self.stream_in.read(chunk_size, exception_on_overflow=False)
                data = np.frombuffer(data, dtype=np.float32)
                gain = self.input_volume.get() / 100.0
                data = data * gain
                symbol = self.decode_audio(data)
                if symbol is not None:
                    buffer.append(symbol)
                    block_size = 31 if speed in ["slow", "medium"] else 1
                    if len(buffer) >= block_size:
                        decoded_text = self.decode_fec(buffer[:block_size])
                        self.rx_output.insert(tk.END, decoded_text + "\n")
                        self.rx_output.see(tk.END)
                        buffer = buffer[block_size:]
                time.sleep(0.005)  # Reduced for turbo responsiveness
            except Exception as e:
                if self.running:
                    self.rx_output.insert(tk.END, f"Receive error: {str(e)}\n")
                break

    def on_closing(self):
        self.running = False
        if self.stream_out:
            try:
                self.stream_out.stop_stream()
                self.stream_out.close()
            except Exception as e:
                print(f"Error closing output stream: {e}")
            self.stream_out = None
        if self.stream_in:
            try:
                self.stream_in.stop_stream()
                self.stream_in.close()
            except Exception as e:
                print(f"Error closing input stream: {e}")
            self.stream_in = None
        if self.rx_thread and self.rx_thread.is_alive():
            self.rx_thread.join(timeout=1.0)
            if self.rx_thread.is_alive():
                print("Receive thread did not terminate in time")
        self.rx_thread = None
        if self.p:
            try:
                self.p.terminate()
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")
            self.p = None
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DipperModeApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()