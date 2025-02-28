import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, simpledialog
import pyaudio
import numpy as np
from scipy.fft import fft
from scipy.signal import butter, lfilter
import threading
import time
import os
from reedsolo import RSCodec, ReedSolomonError
import crcmod
import sys

print("Imports completed")

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
    "9": [(2000, 1500, "slide")], "!": [(5000, 1000, "slide")], "/": [(3000, 4000, "slide")],
    "-": [(2000, 2000, "trill")], ".": [(4500, 4500, "tone")], " ": [(1000, 1000, "tone")]
}

WORD_SOUNDS = {
    "CQ": [(2000, 4000, "slide")], "DE": [(1500, 2500, "slide")],
}

SAMPLE_RATE = 44100
CHUNK = 1024
CALLSIGN_FILE = os.path.join(os.path.dirname(__file__), "mycallsign.txt")

SYMBOL_MAP = {**{char: i for i, char in enumerate(CHAR_SOUNDS.keys())},
              **{word: i + len(CHAR_SOUNDS) for i, word in enumerate(WORD_SOUNDS.keys())}}
REVERSE_MAP = {i: char_or_word for char_or_word, i in SYMBOL_MAP.items()}
RS_CODEC = RSCodec(8)

V4_SYMBOLS = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
    8: "I", 9: "K", 10: "L", 11: "M", 12: "N", 13: "O", 14: "P", 15: "Q"
}
V4_REVERSE_MAP = {v: k for k, v in V4_SYMBOLS.items()}

CONV_TABLE = {
    (0, 0): [0, 0], (0, 1): [1, 1], (1, 0): [1, 0], (1, 1): [0, 1]
}

crc16 = crcmod.mkCrcFun(0x11021, initCrc=0, xorOut=0xFFFF)

class DipperModeApp:
    def __init__(self, root):
        print("Initializing DipperModeApp")
        self.root = root
        try:
            self.root.title("DIPPER V4 by M0OLI")
            self.root.geometry("800x750")
            print("Root window created")
        except Exception as e:
            print(f"Root window error: {e}")
            raise

        self.light_colors = {"bg": "#FFFFFF", "fg": "#000000", "entry_bg": "#F0F0F0", "button_bg": "#D0D0D0"}
        self.current_colors = self.light_colors

        self.my_callsign_value = ""
        try:
            self.my_callsign_value = self.load_callsign()
            if not self.my_callsign_value:
                self.my_callsign_value = self.prompt_callsign()
                self.save_callsign(self.my_callsign_value)
        except Exception as e:
            print(f"Callsign load error: {e}")

        self.p = None
        self.input_devices = {}
        self.output_devices = {}
        self.input_device_index = tk.IntVar(value=-1)
        self.output_device_index = tk.IntVar(value=-1)
        self.input_volume = tk.DoubleVar(value=50.0)
        self.speed_var = tk.StringVar(value="slow")
        self.filter_var = tk.StringVar(value="none")
        self.stream_out = None
        self.stream_in = None
        self.running = False
        self.rx_thread = None
        self.tx_thread = None
        self.is_v4_mode = False
        self.packet_buffer = {}
        self.last_packet_id = -1
        self.tx_queue = []
        self.rx_ack = None

        print("Building GUI")
        try:
            self.main_frame = tk.Frame(root, bg=self.current_colors["bg"])
            self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

            self.callsign_frame = tk.Frame(self.main_frame, bg=self.current_colors["bg"])
            self.callsign_frame.pack(fill="x", pady=(0, 10))
            
            tk.Label(self.callsign_frame, text="My Callsign:", fg=self.current_colors["fg"], 
                     bg=self.current_colors["bg"]).grid(row=0, column=0, padx=5, pady=5, sticky="e")
            self.my_callsign = tk.Entry(self.callsign_frame, width=15, fg=self.current_colors["fg"], 
                                       bg=self.current_colors["entry_bg"], insertbackground=self.current_colors["fg"])
            self.my_callsign.grid(row=0, column=1, padx=5, pady=5)
            if self.my_callsign_value:
                self.my_callsign.insert(0, self.my_callsign_value)
                self.my_callsign.config(state="disabled")

            tk.Label(self.callsign_frame, text="To Callsign:", fg=self.current_colors["fg"], 
                     bg=self.current_colors["bg"]).grid(row=0, column=2, padx=5, pady=5, sticky="e")
            self.to_callsign = tk.Entry(self.callsign_frame, width=15, fg=self.current_colors["fg"], 
                                       bg=self.current_colors["entry_bg"], insertbackground=self.current_colors["fg"])
            self.to_callsign.grid(row=0, column=3, padx=5, pady=5)

            self.dark_mode_var = tk.BooleanVar()
            self.dark_mode_check = tk.Checkbutton(self.callsign_frame, text="Dark Mode", 
                                                 variable=self.dark_mode_var, command=self.toggle_dark_mode, 
                                                 fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                                                 selectcolor=self.current_colors["entry_bg"])
            self.dark_mode_check.grid(row=0, column=4, padx=20, pady=5)

            self.v4_indicator_frame = tk.Frame(self.callsign_frame, bg=self.current_colors["bg"])
            self.v4_indicator_frame.grid(row=0, column=5, padx=10, pady=5, sticky="e")
            tk.Label(self.v4_indicator_frame, text="Robust RX/TX:", fg=self.current_colors["fg"], 
                     bg=self.current_colors["bg"]).pack(side="left")
            self.v4_indicator = tk.Canvas(self.v4_indicator_frame, width=20, height=20, 
                                         bg=self.current_colors["bg"], highlightthickness=0)
            self.v4_indicator.create_rectangle(0, 0, 20, 20, fill="grey", tags="mode_light")
            self.v4_indicator.pack(side="left")

            self.tx_frame = tk.Frame(self.main_frame, bg=self.current_colors["bg"])
            self.tx_frame.pack(fill="x", pady=(0, 10))
            
            tk.Label(self.tx_frame, text="Transmit Message:", fg=self.current_colors["fg"], 
                     bg=self.current_colors["bg"]).pack(anchor="w", pady=(0, 5))
            self.tx_input = tk.Entry(self.tx_frame, width=60, fg=self.current_colors["fg"], 
                                    bg=self.current_colors["entry_bg"], insertbackground=self.current_colors["fg"])
            self.tx_input.pack(fill="x", pady=(0, 5))
            
            self.button_frame = tk.Frame(self.tx_frame, bg=self.current_colors["bg"])
            self.button_frame.pack(anchor="w", pady=(0, 5))
            self.tx_button = tk.Button(self.button_frame, text="Send", command=self.transmit, 
                                      fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
            self.tx_button.pack(side="left", padx=5)
            self.cq_button = tk.Button(self.button_frame, text="CQ", command=self.send_cq, 
                                      fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
            self.cq_button.pack(side="left", padx=5)

            self.speed_frame = tk.LabelFrame(self.tx_frame, text="Speed", fg=self.current_colors["fg"], 
                                            bg=self.current_colors["bg"])
            self.speed_frame.pack(anchor="w", pady=5)
            tk.Radiobutton(self.speed_frame, text="Slow (~6-13 WPM)", variable=self.speed_var, value="slow", 
                          fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                          selectcolor=self.current_colors["entry_bg"]).pack(side="left", padx=5)
            tk.Radiobutton(self.speed_frame, text="Medium (~10-20 WPM)", variable=self.speed_var, value="medium", 
                          fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                          selectcolor=self.current_colors["entry_bg"]).pack(side="left", padx=5)
            tk.Radiobutton(self.speed_frame, text="Fast (~15-30 WPM)", variable=self.speed_var, value="fast", 
                          fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                          selectcolor=self.current_colors["entry_bg"]).pack(side="left", padx=5)
            tk.Radiobutton(self.speed_frame, text="Turbo (~40+ WPM)", variable=self.speed_var, value="turbo", 
                          fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                          selectcolor=self.current_colors["entry_bg"]).pack(side="left", padx=5)
            tk.Radiobutton(self.speed_frame, text="Robust (~20-40 WPM)", variable=self.speed_var, value="robust", 
                          fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                          selectcolor=self.current_colors["entry_bg"]).pack(side="left", padx=5)

            self.filter_frame = tk.LabelFrame(self.tx_frame, text="Frequency Filter", fg=self.current_colors["fg"], 
                                             bg=self.current_colors["bg"])
            self.filter_frame.pack(anchor="w", pady=5)
            tk.Radiobutton(self.filter_frame, text="No Filter", variable=self.filter_var, value="none", 
                          fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                          selectcolor=self.current_colors["entry_bg"]).pack(side="left", padx=5)
            tk.Radiobutton(self.filter_frame, text="900-5100 Hz", variable=self.filter_var, value="900-5100", 
                          fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                          selectcolor=self.current_colors["entry_bg"]).pack(side="left", padx=5)
            tk.Radiobutton(self.filter_frame, text="800-5200 Hz", variable=self.filter_var, value="800-5200", 
                          fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                          selectcolor=self.current_colors["entry_bg"]).pack(side="left", padx=5)

            self.rx_frame = tk.Frame(self.main_frame, bg=self.current_colors["bg"])
            self.rx_frame.pack(fill="both", expand=True, pady=(10, 0))
            
            self.rx_header_frame = tk.Frame(self.rx_frame, bg=self.current_colors["bg"])
            self.rx_header_frame.pack(fill="x")
            tk.Label(self.rx_header_frame, text="Received Messages:", fg=self.current_colors["fg"], 
                     bg=self.current_colors["bg"]).pack(side="left", pady=(0, 5))
            self.clear_button = tk.Button(self.rx_header_frame, text="Clear", command=self.clear_receive, 
                                         fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
            self.clear_button.pack(side="right", padx=5, pady=5)

            self.rx_output = scrolledtext.ScrolledText(self.rx_frame, width=70, height=20, 
                                                      fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], 
                                                      insertbackground=self.current_colors["fg"])
            self.rx_output.pack(fill="both", expand=True, pady=(0, 10))
            self.rx_output.tag_config("sent", foreground="red")
            self.rx_output.tag_config("underline", underline=True)

            print("GUI built successfully")
        except Exception as e:
            print(f"GUI build error: {e}")
            self.rx_output.insert(tk.END, f"GUI error: {e}\n")

        print("Scheduling audio start")
        self.root.after(100, self.start_audio)

    def clear_receive(self):
        """Clears the received messages text box."""
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
        return simpledialog.askstring("Setup", "What is your callsign?", parent=self.root) or ""

    def get_audio_devices(self):
        if not self.p:
            return {}, {}
        try:
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
        except Exception as e:
            print(f"Error getting audio devices: {e}")
            return {}, {}

    def start_audio(self):
        print("Starting audio")
        try:
            self.p = pyaudio.PyAudio()
            self.input_devices, self.output_devices = self.get_audio_devices()
            input_idx = self.input_device_index.get() if self.input_device_index.get() >= 0 else None
            output_idx = self.output_device_index.get() if self.output_device_index.get() >= 0 else None
            
            self.stream_out = self.p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, output=True, 
                                        output_device_index=output_idx)
            self.stream_in = self.p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True, 
                                       frames_per_buffer=CHUNK, input_device_index=input_idx)
            self.running = True
            self.rx_thread = threading.Thread(target=self.receive_loop, daemon=True)
            self.rx_thread.start()
            self.tx_thread = threading.Thread(target=self.transmit_loop, daemon=True)
            self.tx_thread.start()
            self.rx_output.insert(tk.END, "Audio started successfully\n")
            print("Audio threads started")
        except Exception as e:
            self.rx_output.insert(tk.END, f"Audio failed to start: {str(e)}\n")
            self.running = False
            print(f"Audio startup error: {e}")
            self.root.after(0, lambda: messagebox.showwarning("Audio Warning", 
                            f"Audio setup failed: {str(e)}. Continuing without audio."))

    def toggle_dark_mode(self):
        if self.dark_mode_var.get():
            self.current_colors = {"bg": "#000000", "fg": "#00FF00", "entry_bg": "#1A1A1A", "button_bg": "#333333"}
        else:
            self.current_colors = self.light_colors
        self.root.configure(bg=self.current_colors["bg"])
        self.main_frame.configure(bg=self.current_colors["bg"])
        self.callsign_frame.configure(bg=self.current_colors["bg"])
        self.tx_frame.configure(bg=self.current_colors["bg"])
        self.rx_frame.configure(bg=self.current_colors["bg"])
        self.rx_header_frame.configure(bg=self.current_colors["bg"])
        self.speed_frame.configure(bg=self.current_colors["bg"])
        self.filter_frame.configure(bg=self.current_colors["bg"])
        self.v4_indicator_frame.configure(bg=self.current_colors["bg"])
        
        for widget in (self.callsign_frame.winfo_children() + 
                       self.tx_frame.winfo_children() + 
                       self.rx_frame.winfo_children() + 
                       self.rx_header_frame.winfo_children() +
                       self.speed_frame.winfo_children() +
                       self.filter_frame.winfo_children()):
            if isinstance(widget, (tk.Label, tk.Checkbutton, tk.Radiobutton)):
                widget.configure(fg=self.current_colors["fg"], bg=self.current_colors["bg"])
            elif isinstance(widget, tk.Entry):
                widget.configure(fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], 
                                insertbackground=self.current_colors["fg"])
            elif isinstance(widget, tk.Button):
                widget.configure(fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
        self.rx_output.configure(fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], 
                                insertbackground=self.current_colors["fg"])
        self.clear_button.configure(fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
        self.dark_mode_check.configure(selectcolor=self.current_colors["entry_bg"])

    def generate_sound(self, pattern, duration=0.05):
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

    def convolutional_encode(self, bits):
        encoded = []
        state = 0
        for bit in bits:
            encoded.extend(CONV_TABLE[(state, bit)])
            state = bit
        return encoded

    def viterbi_decode(self, bits):
        states = {0: (0, []), 1: (float('inf'), [])}
        for i in range(0, len(bits), 2):
            r1, r2 = bits[i], bits[i+1]
            new_states = {}
            for s in [0, 1]:
                for b in [0, 1]:
                    next_s = b
                    o1, o2 = CONV_TABLE[(s, b)]
                    cost = states[s][0] + (r1 ^ o1) + (r2 ^ o2)
                    path = states[s][1] + [b]
                    if next_s not in new_states or cost < new_states[next_s][0]:
                        new_states[next_s] = (cost, path)
            states = new_states
        return min(states.values(), key=lambda x: x[0])[1]

    def interleave(self, bits):
        block_size = 16
        padded = bits + [0] * (block_size - len(bits) % block_size) if len(bits) % block_size else bits
        return [padded[i // 4 + (i % 4) * 4] for i in range(len(padded))]

    def deinterleave(self, bits):
        block_size = 16
        deinterleaved = [0] * len(bits)
        for i in range(len(bits)):
            deinterleaved[i // 4 + (i % 4) * 4] = bits[i]
        return deinterleaved

    def detect_v4_preamble(self, data):
        preamble_freqs = [(1500, 2000), (2500, 3000), (3500, 4000), (4500, 5000), 
                         (2000, 1500), (2000, 2500), (3000, 3500)]
        segment_size = int(SAMPLE_RATE * 0.05)
        freq_axis = np.linspace(0, SAMPLE_RATE // 2, segment_size // 2)
        detected = True
        for i, (start_freq, end_freq) in enumerate(preamble_freqs):
            chunk = data[i * segment_size:(i + 1) * segment_size]
            if len(chunk) < segment_size:
                return False
            freqs = np.abs(fft(chunk)[:segment_size // 2])
            peak_idx = np.argmax(freqs)
            peak_freq = freq_axis[peak_idx]
            if not (start_freq - 200 < peak_freq < end_freq + 200):
                detected = False
                break
        return detected

    def encode_v4_packet(self, text, packet_id):
        bits = [int(b) for char in text for b in bin(ord(char))[2:].zfill(8)]
        encoded_bits = self.convolutional_encode(bits)
        interleaved_bits = self.interleave(encoded_bits)
        symbols = []
        for i in range(0, len(interleaved_bits), 4):
            chunk = interleaved_bits[i:i+4]
            value = sum(b << (3-j) for j, b in enumerate(chunk[:4]))
            symbols.append(V4_SYMBOLS.get(value, " "))
        
        header = [V4_SYMBOLS[packet_id % 16]]
        crc = crc16(text.encode('utf-8'))
        crc_symbols = [V4_SYMBOLS[(crc >> 12) & 0xF], V4_SYMBOLS[(crc >> 8) & 0xF], 
                      V4_SYMBOLS[(crc >> 4) & 0xF], V4_SYMBOLS[crc & 0xF]]
        return header + symbols + crc_symbols

    def decode_v4_packet(self, symbols):
        if len(symbols) < 5:
            return None, None
        packet_id = V4_REVERSE_MAP.get(symbols[0], 0)
        payload_symbols = symbols[1:-4]
        crc_symbols = symbols[-4:]
        bits = [int(b) for sym in payload_symbols 
                for b in bin(V4_REVERSE_MAP.get(sym, 0))[2:].zfill(4)]
        deinterleaved_bits = self.deinterleave(bits)
        decoded_bits = self.viterbi_decode(deinterleaved_bits)
        text = "".join(chr(sum(b << (7-j) for j, b in enumerate(decoded_bits[i:i+8]))) 
                      for i in range(0, len(decoded_bits), 8) if len(decoded_bits[i:i+8]) == 8)
        crc = crc16(text.encode('utf-8'))
        received_crc = sum(V4_REVERSE_MAP.get(s, 0) << (12 - 4*i) for i, s in enumerate(crc_symbols))
        return packet_id, text if crc == received_crc else None

    def send_ack_nack(self, packet_id, success):
        if not self.stream_out:
            return
        self.set_v4_tx_indicator()
        ack_sound = self.generate_sound(CHAR_SOUNDS["K" if success else "N"], 0.05)
        audio_data = np.concatenate((ack_sound, np.zeros(int(SAMPLE_RATE * 0.025))))
        self.stream_out.write(audio_data.astype(np.float32).tobytes())
        self.reset_indicator()

    def transmit(self):
        if not self.stream_out:
            messagebox.showwarning("Warning", "Audio not started!")
            return
        to_call = self.to_callsign.get().upper()
        my_call = self.my_callsign.get().upper()
        if not to_call or not my_call:
            messagebox.showwarning("Warning", "Enter both callsigns!")
            return
        text = self.tx_input.get().strip().upper()
        full_text = f"{to_call} DE {my_call} {text}" if text else f"{to_call} DE {my_call}"
        speed = self.speed_var.get()
        if speed == "robust":
            packet_id = (self.last_packet_id + 1) % 16
            self.last_packet_id = packet_id
            self.tx_queue.append((full_text, packet_id))
        else:
            self.tx_queue.append((full_text, None))
        self.tx_input.delete(0, tk.END)

    def transmit_loop(self):
        while self.running:
            if self.tx_queue and self.stream_out:
                full_text, packet_id = self.tx_queue.pop(0)
                speed = self.speed_var.get()
                duration_map = {"slow": 0.2, "medium": 0.15, "fast": 0.1, "turbo": 0.03, "robust": 0.05}
                duration = duration_map[speed]
                gap_duration = duration / 2

                audio_data = np.array([], dtype=np.float32)
                if speed == "robust":
                    self.set_v4_tx_indicator()
                    preamble_audio = np.concatenate([self.generate_sound(CHAR_SOUNDS[num], 0.05) 
                                                    for num in "1357924"])
                    symbols = self.encode_v4_packet(full_text, packet_id)
                    audio_data = preamble_audio
                    for sym in symbols:
                        pattern = CHAR_SOUNDS.get(sym, CHAR_SOUNDS[" "])
                        sound = self.generate_sound(pattern, 0.05)
                        audio_data = np.concatenate((audio_data, sound, np.zeros(int(SAMPLE_RATE * gap_duration))))
                    self.stream_out.write(audio_data.astype(np.float32).tobytes())
                    self.rx_output.insert(tk.END, "Robust Sent: " + full_text + "\n", "sent")
                    self.reset_indicator()
                    time.sleep(0.2)  # Wait for ACK/NACK
                    if self.rx_ack == "N":  # Only re-queue on NACK
                        self.tx_queue.append((full_text, packet_id))
                    self.rx_ack = None
                else:
                    encoded_symbols = self.encode_fec(full_text)
                    for symbol in encoded_symbols:
                        char = REVERSE_MAP.get(symbol, " ")
                        pattern = WORD_SOUNDS.get(char, CHAR_SOUNDS.get(char, []))
                        sound = self.generate_sound(pattern, duration)
                        audio_data = np.concatenate((audio_data, sound, np.zeros(int(SAMPLE_RATE * gap_duration))))
                    self.stream_out.write(audio_data.astype(np.float32).tobytes())
                    self.rx_output.insert(tk.END, "Sent: " + full_text + "\n", "sent")
            time.sleep(0.001)

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
        return symbol_list

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

    def send_cq(self):
        if not self.stream_out:
            messagebox.showwarning("Warning", "Audio not started!")
            return
        my_call = self.my_callsign.get().upper()
        if not my_call:
            messagebox.showwarning("Warning", "Enter your callsign!")
            return
        speed = self.speed_var.get()
        single_cq = f"CQ CQ CQ DE {my_call}"
        if speed == "robust":
            packet_id = (self.last_packet_id + 1) % 16
            self.last_packet_id = packet_id
            self.tx_queue.append((single_cq, packet_id))  # Single CQ for Robust
        else:
            cq_text = f"{single_cq} {single_cq}"  # Two repeats for old modes
            self.tx_queue.append((cq_text, None))

    def decode_audio(self, data):
        freqs = np.abs(fft(data)[:len(data)//2])
        freq_axis = np.linspace(0, SAMPLE_RATE // 2, len(freqs))
        peak_idx = np.argmax(freqs)
        peak_freq = freq_axis[peak_idx]
        if self.is_v4_mode:
            for sym, value in V4_REVERSE_MAP.items():
                pattern = CHAR_SOUNDS[sym]
                start_freq = pattern[0][0]
                if abs(peak_freq - start_freq) < 200:
                    return sym
        else:
            for word, pattern in WORD_SOUNDS.items():
                start_freq = pattern[0][0]
                if abs(peak_freq - start_freq) < 200:
                    return SYMBOL_MAP[word]
            for char, pattern in CHAR_SOUNDS.items():
                start_freq = pattern[0][0]
                if abs(peak_freq - start_freq) < 200:
                    return SYMBOL_MAP[char]
        return None

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_filter(self, data):
        filter_type = self.filter_var.get()
        if filter_type == "none":
            return data
        elif filter_type == "900-5100":
            b, a = self.butter_bandpass(900, 5100, SAMPLE_RATE)
            return lfilter(b, a, data)
        elif filter_type == "800-5200":
            b, a = self.butter_bandpass(800, 5200, SAMPLE_RATE)
            return lfilter(b, a, data)
        return data

    def set_v4_rx_indicator(self):
        self.v4_indicator.itemconfig("mode_light", fill="green")
        if hasattr(self, "indicator_timeout"):
            self.root.after_cancel(self.indicator_timeout)
        self.indicator_timeout = self.root.after(3000, self.reset_indicator)

    def set_v4_tx_indicator(self):
        if hasattr(self, "indicator_timeout"):
            self.root.after_cancel(self.indicator_timeout)
        self.v4_indicator.itemconfig("mode_light", fill="red")

    def reset_indicator(self):
        self.v4_indicator.itemconfig("mode_light", fill="grey")

    def receive_loop(self):
        buffer = []
        gap_map = {"slow": 0.1, "medium": 0.075, "fast": 0.05, "turbo": 0.015, "robust": 0.025}
        duration_map = {"slow": 0.2, "medium": 0.15, "fast": 0.1, "turbo": 0.03, "robust": 0.05}
        preamble_detected = False
        while self.running:
            try:
                speed = self.speed_var.get()
                duration = duration_map[speed]
                chunk_size = int(SAMPLE_RATE * (duration + gap_map[speed] * 2))
                data = self.stream_in.read(chunk_size, exception_on_overflow=False)
                data = np.frombuffer(data, dtype=np.float32)
                gain = self.input_volume.get() / 100.0
                data = data * gain
                data = self.apply_filter(data)

                if not preamble_detected and len(buffer) == 0:
                    if self.detect_v4_preamble(data[:int(SAMPLE_RATE * 0.4)]):
                        self.is_v4_mode = True
                        self.set_v4_rx_indicator()
                        preamble_detected = True
                        continue

                symbol = self.decode_audio(data)
                if symbol is not None:
                    buffer.append(symbol)
                    if self.is_v4_mode:
                        if len(buffer) >= 10:
                            packet_id, decoded_text = self.decode_v4_packet(buffer)
                            if decoded_text and packet_id not in self.packet_buffer:
                                self.packet_buffer[packet_id] = decoded_text
                                self.rx_output.insert(tk.END, "Robust: " + decoded_text + "\n")
                                self.rx_output.see(tk.END)
                                self.send_ack_nack(packet_id, True)
                                self.rx_ack = "K"
                            elif not decoded_text:
                                self.send_ack_nack(packet_id, False)
                                self.rx_ack = "N"
                            buffer.clear()
                            preamble_detected = False
                            self.is_v4_mode = False
                    else:
                        block_size = 31 if speed in ["slow", "medium"] else 1
                        if len(buffer) >= block_size:
                            decoded_text = self.decode_fec(buffer[:block_size])
                            self.rx_output.insert(tk.END, decoded_text + "\n")
                            self.rx_output.see(tk.END)
                            buffer = buffer[block_size:]
                time.sleep(0.001)
            except Exception as e:
                if self.running:
                    self.rx_output.insert(tk.END, f"Receive error: {str(e)}\n")
                    print(f"Receive loop error: {e}")
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
        if self.tx_thread and self.tx_thread.is_alive():
            self.tx_thread.join(timeout=1.0)
        self.rx_thread = None
        self.tx_thread = None
        if self.p:
            try:
                self.p.terminate()
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")
            self.p = None
        self.root.destroy()
        print("Application closed")

if __name__ == "__main__":
    try:
        print("Starting main")
        root = tk.Tk()
        print("Tk root created")
        app = DipperModeApp(root)
        print("App instance created")
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
        print("Mainloop exited")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        input("Press Enter to exit...")