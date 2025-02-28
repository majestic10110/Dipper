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
    "A": [(2000, 2000, 0.2, "tone")], "B": [(3000, 3000, 0.2, "trill")], "C": [(4000, 2000, 0.2, "slide")],
    "D": [(1500, 1500, 0.2, "tone")], "E": [(5000, 5000, 0.1, "tone")], "F": [(2500, 3500, 0.2, "slide")],
    "G": [(1000, 1000, 0.2, "trill")], "H": [(3500, 3500, 0.2, "tone")], "I": [(4500, 4500, 0.1, "tone")],
    "J": [(2000, 4000, 0.2, "slide")], "K": [(3000, 3000, 0.2, "tone")], "L": [(4000, 4000, 0.2, "trill")],
    "M": [(1500, 2500, 0.2, "slide")], "N": [(5000, 3000, 0.2, "slide")], "O": [(1000, 1000, 0.2, "tone")],
    "P": [(3500, 3500, 0.2, "trill")], "Q": [(2500, 2500, 0.2, "tone")], "R": [(2000, 2000, 0.2, "trill")],
    "S": [(4500, 1500, 0.2, "slide")], "T": [(3000, 5000, 0.2, "slide")], "U": [(4000, 4000, 0.2, "tone")],
    "V": [(1500, 1500, 0.2, "trill")], "W": [(1000, 2000, 0.2, "slide")], "X": [(5000, 5000, 0.2, "trill")],
    "Y": [(3500, 2500, 0.2, "slide")], "Z": [(2500, 2500, 0.2, "trill")],
    "0": [(1000, 1500, 0.2, "slide")], "1": [(1500, 2000, 0.2, "slide")], "2": [(2000, 2500, 0.2, "slide")],
    "3": [(2500, 3000, 0.2, "slide")], "4": [(3000, 3500, 0.2, "slide")], "5": [(3500, 4000, 0.2, "slide")],
    "6": [(4000, 4500, 0.2, "slide")], "7": [(4500, 5000, 0.2, "slide")], "8": [(5000, 4500, 0.2, "slide")],
    "9": [(2000, 1500, 0.2, "slide")],
    "!": [(5000, 1000, 0.2, "slide")], "/": [(3000, 4000, 0.2, "slide")], "-": [(2000, 2000, 0.2, "trill")],
    ".": [(4500, 4500, 0.1, "tone")], " ": [(1000, 1000, 0.1, "tone")]
}

WORD_SOUNDS = {
    "CQ": [(2000, 4000, 0.2, "slide")],
    "HTTPS://WWW.": [(1000, 1000, 0.1, "tone"), (3000, 3000, 0.1, "tone"), (5000, 5000, 0.2, "trill")],
    "DE": [(1500, 2500, 0.2, "slide")],
    "the": [(1100, 1100, 0.1, "tone")],
    "and": [(1150, 1150, 0.2, "trill")],
    "to": [(1200, 2200, 0.2, "slide")],
    "of": [(1250, 1250, 0.1, "tone")],
    "you": [(1300, 1300, 0.2, "tone")],
    "I": [(1350, 1350, 0.1, "tone")],
    "in": [(1400, 1400, 0.2, "trill")],
    "that": [(1450, 1450, 0.2, "tone")],
    "it": [(1550, 1550, 0.1, "tone")],
    "he": [(1600, 2600, 0.2, "slide")],
    "for": [(1650, 1650, 0.2, "tone")],
    "with": [(1700, 2700, 0.2, "slide")],
    "on": [(1750, 1750, 0.2, "trill")],
    "they": [(1800, 1800, 0.1, "tone")],
    "have": [(1850, 2850, 0.2, "slide")],
    "from": [(1900, 1900, 0.2, "tone")],
    "who": [(1950, 1950, 0.1, "trill")],
    "this": [(2050, 2050, 0.2, "tone")],
    "what": [(2100, 3100, 0.2, "slide")],
    "say": [(2150, 2150, 0.1, "tone")],
    "K": [(2200, 2200, 0.1, "tone")],
    "QTH": [(2250, 3250, 0.2, "slide")],
    "73": [(2300, 2300, 0.2, "trill")],
    "QSL": [(2350, 2350, 0.1, "tone")],
    "DX": [(2400, 3400, 0.2, "slide")],
    "SSB": [(2450, 2450, 0.2, "tone")],
    "FM": [(2550, 2550, 0.2, "trill")],
    "AM": [(2600, 2600, 0.1, "tone")],
    "RST": [(2650, 3650, 0.2, "slide")],
    "QSO": [(2700, 2700, 0.2, "trill")],
    "OM": [(2750, 2750, 0.1, "tone")],
    "YL": [(2800, 2800, 0.2, "tone")],
    "Antenna": [(2850, 3850, 0.2, "slide")],
    "Frequency": [(2900, 2900, 0.2, "tone")],
    "Power": [(2950, 2950, 0.1, "trill")],
    "Weather": [(3050, 2050, 0.2, "slide")],
    "Conditions": [(3100, 3100, 0.2, "tone")],
    "Hello": [(3150, 3150, 0.1, "trill")],
    "<html>": [(3200, 4200, 0.2, "slide")],
    "<head>": [(3250, 3250, 0.2, "tone")],
    "<title>": [(3300, 4300, 0.2, "trill")],
    "<body>": [(3350, 3350, 0.1, "tone")],
    "<meta>": [(3400, 3400, 0.2, "trill")],
    "<link>": [(3450, 4450, 0.2, "slide")],
    "<script>": [(3500, 3500, 0.2, "tone")],
    "<style>": [(3550, 3550, 0.1, "trill")],
    "<div>": [(3600, 4600, 0.2, "slide")],
    "<span>": [(3650, 3650, 0.2, "tone")],
    "<p>": [(3700, 3700, 0.1, "trill")],
    "<a>": [(3750, 4750, 0.2, "slide")],
    "<img>": [(3800, 3800, 0.2, "tone")],
    "<ul>": [(3850, 3850, 0.1, "trill")],
    "<li>": [(3900, 4900, 0.2, "slide")],
    "<table>": [(3950, 3950, 0.2, "tone")],
    "<tr>": [(4050, 4050, 0.1, "trill")],
    "<td>": [(4100, 4100, 0.2, "tone")],
    "<form>": [(4150, 3150, 0.2, "slide")],
    "<input>": [(4200, 4200, 0.2, "trill")]
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
        self.root.title("DIPPER V2.2_with_FEC by M0OLI")
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
        self.use_fec_var = tk.BooleanVar(value=True)  # FEC on by default
        self.stream_out = None
        self.stream_in = None
        self.running = False

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
        self.tx_input.pack(fill="x", pady=(0, 10))
        
        self.tx_button = tk.Button(self.tx_frame, text="Send", command=self.transmit, fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
        self.tx_button.pack(side="left", padx=5)
        self.cq_button = tk.Button(self.tx_frame, text="CQ", command=self.send_cq, fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
        self.cq_button.pack(side="left", padx=5)
        self.fec_check = tk.Checkbutton(self.tx_frame, text="Use FEC", variable=self.use_fec_var, fg=self.current_colors["fg"], bg=self.current_colors["bg"], selectcolor=self.current_colors["entry_bg"])
        self.fec_check.pack(side="left", padx=5)

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
            if not hasattr(self, "rx_thread") or not self.rx_thread.is_alive():
                self.rx_thread = threading.Thread(target=self.receive_loop)
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
        
        for widget in self.callsign_frame.winfo_children() + self.tx_frame.winfo_children() + self.rx_frame.winfo_children():
            if isinstance(widget, (tk.Label, tk.Checkbutton)):
                widget.configure(fg=self.current_colors["fg"], bg=self.current_colors["bg"])
            elif isinstance(widget, tk.Entry):
                widget.configure(fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], insertbackground=self.current_colors["fg"])
            elif isinstance(widget, tk.Button):
                widget.configure(fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
        self.rx_output.configure(fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], insertbackground=self.current_colors["fg"])
        self.clear_button.configure(fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
        self.dark_mode_check.configure(selectcolor=self.current_colors["entry_bg"])
        self.fec_check.configure(fg=self.current_colors["fg"], bg=self.current_colors["bg"], selectcolor=self.current_colors["entry_bg"])

    def generate_sound(self, pattern):
        audio = np.array([], dtype=np.float32)
        if not pattern:
            return np.zeros(int(SAMPLE_RATE * 0.2), dtype=np.float32)
        for start_freq, end_freq, duration, sound_type in pattern:
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
        
        if self.use_fec_var.get():  # Apply FEC if checkbox is ticked
            while len(symbol_list) % 23 != 0:
                symbol_list.append(SYMBOL_MAP[" "])
            blocks = [symbol_list[i:i+23] for i in range(0, len(symbol_list), 23)]
            encoded_blocks = []
            for block in blocks:
                encoded = RS_CODEC.encode(block)
                encoded_blocks.extend(encoded)
            return encoded_blocks
        else:  # No FEC, return raw symbols
            return symbol_list

    def decode_fec(self, symbols):
        if self.use_fec_var.get():  # Decode with FEC if enabled
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
        else:  # No FEC, decode directly
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
        for symbol in encoded_symbols:
            char = REVERSE_MAP.get(symbol, " ")
            if char in WORD_SOUNDS:
                pattern = WORD_SOUNDS[char]
            elif char in CHAR_SOUNDS:
                pattern = CHAR_SOUNDS[char]
            else:
                pattern = []
            sound = self.generate_sound(pattern)
            audio_data = np.concatenate((audio_data, sound, np.zeros(int(SAMPLE_RATE * 0.1))))
            audio_data = np.concatenate((audio_data, sound, np.zeros(int(SAMPLE_RATE * 0.1))))

        self.stream_out.write(audio_data.astype(np.float32).tobytes())
        
        # Insert with underlining in receive box
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
        for i, symbol in enumerate(encoded_symbols):
            char = REVERSE_MAP.get(symbol, " ")
            if char in WORD_SOUNDS:
                pattern = WORD_SOUNDS[char]
            elif char in CHAR_SOUNDS:
                pattern = CHAR_SOUNDS[char]
            else:
                pattern = []
            sound = self.generate_sound(pattern)
            audio_data = np.concatenate((audio_data, sound, np.zeros(int(SAMPLE_RATE * 0.1))))
            audio_data = np.concatenate((audio_data, sound, np.zeros(int(SAMPLE_RATE * 0.1))))
            if (i + 1) % 5 == 0 and i < len(encoded_symbols) - 1 and self.use_fec_var.get():
                audio_data = np.concatenate((audio_data, np.zeros(int(SAMPLE_RATE * 0.2))))

        self.stream_out.write(audio_data.astype(np.float32).tobytes())
        
        # Insert with underlining in receive box
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
        while self.running:
            try:
                data = np.frombuffer(self.stream_in.read(CHUNK, exception_on_overflow=False), dtype=np.float32)
                gain = self.input_volume.get() / 100.0
                data = data * gain
                symbol = self.decode_audio(data)
                if symbol is not None:
                    buffer.append(symbol)
                    block_size = 31 if self.use_fec_var.get() else 1
                    if len(buffer) >= block_size:
                        decoded_text = self.decode_fec(buffer[:block_size])
                        self.rx_output.insert(tk.END, decoded_text + "\n")
                        self.rx_output.see(tk.END)
                        buffer = buffer[block_size:]
                time.sleep(0.01)
            except Exception as e:
                self.rx_output.insert(tk.END, f"Receive error: {str(e)}\n")
                break

    def on_closing(self):
        self.running = False
        if self.rx_thread and self.rx_thread.is_alive():
            self.rx_thread.join()
        if self.stream_out:
            self.stream_out.stop_stream()
            self.stream_out.close()
        if self.stream_in:
            self.stream_in.stop_stream()
            self.stream_in.close()
        if self.p:
            self.p.terminate()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DipperModeApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()