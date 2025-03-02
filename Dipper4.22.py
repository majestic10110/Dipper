import tkinter as tk
from tkinter import messagebox, ttk, simpledialog, Menu
import tkinter.font
import pyaudio
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import butter, lfilter
import threading
import time
import os
import serial
from reedsolo import RSCodec, ReedSolomonError
import crcmod
import sys
import serial.tools.list_ports

print("Script starting...")
print("Imports completed")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
script_path = os.path.abspath(__file__)
print(f"Script path: {script_path}")

try:
    print("Testing dependencies...")
    p = pyaudio.PyAudio()
    p.terminate()
    np.array([1, 2, 3])
    RSCodec(8)
    crc16 = crcmod.mkCrcFun(0x11021, initCrc=0, xorOut=0xFFFF)
    serial.tools.list_ports.comports()
    print("Dependencies loaded successfully")
except Exception as e:
    print(f"Dependency error: {e}")
    sys.exit(1)

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
    "-": [(2000, 2000, "trill")], ".": [(4500, 4500, "tone")], " ": [(1000, 1000, "tone")],
    "@": [(6000, 7000, "slide")]
}

WORD_SOUNDS = {
    "CQ": [(2000, 4000, "slide")], "DE": [(1500, 2500, "slide")],
}

SAMPLE_RATE = 44100
CHUNK = 1024
CALLSIGN_FILE = os.path.join(os.path.dirname(__file__), "mycallsign.txt")
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "radio_settings.txt")

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
            self.root.title("DIPPER V4.2 Robust+ by M0OLI")
            self.root.geometry("1000x750")
            print("Root window created")
        except Exception as e:
            print(f"Root window error: {e}")
            raise

        self.settings = self.load_settings()
        self.radio = tk.StringVar(value=self.settings.get("radio", "Icom IC-703"))
        self.usb_address = tk.StringVar(value=self.settings.get("usb_address", ""))
        self.mode_usb = tk.BooleanVar(value=self.settings.get("mode_usb", False))
        self.mode_usb_digital = tk.BooleanVar(value=self.settings.get("mode_usb_digital", False))
        self.mode_fm = tk.BooleanVar(value=self.settings.get("mode_fm", False))
        self.serial_port = tk.StringVar(value=self.settings.get("serial_port", "NONE"))
        self.baud_rate = tk.StringVar(value=self.settings.get("baud_rate", ""))
        self.rts = tk.BooleanVar(value=self.settings.get("rts", False))
        self.dtr = tk.BooleanVar(value=self.settings.get("dtr", False))
        self.ptt_port = tk.StringVar(value=self.settings.get("ptt_port", "NONE"))
        self.ptt_rts = tk.BooleanVar(value=self.settings.get("ptt_rts", False))
        self.ptt_dtr = tk.BooleanVar(value=self.settings.get("ptt_dtr", False))

        self.input_device_index = tk.IntVar(value=self.settings.get("input_device", -1))
        self.output_device_index = tk.IntVar(value=self.settings.get("output_device", -1))
        self.input_volume = tk.DoubleVar(value=self.settings.get("input_volume", 50.0))
        self.sensitivity = tk.DoubleVar(value=self.settings.get("sensitivity", 50.0))
        self.input_devices, self.output_devices = self.get_audio_devices()

        valid_speeds = ["normal", "robust", "robust_plus"]
        loaded_speed = self.settings.get("speed", "normal")
        self.speed_var = tk.StringVar(value=loaded_speed if loaded_speed in valid_speeds else "normal")
        self.filter_var = tk.StringVar(value=self.settings.get("filter", "none"))
        self.receive_buffer = []
        self.temp_receive_buffer = []

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
        self.radio_serial = None
        self.indicator_timeout = None

        self.rx_output = None

        print("Building GUI")
        try:
            self.main_frame = tk.Frame(root, bg=self.current_colors["bg"])
            self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

            menubar = Menu(self.root)
            self.root.config(menu=menubar)
            settings_menu = Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Settings", menu=settings_menu)
            # Add User submenu at the top
            user_menu = Menu(settings_menu, tearoff=0)
            settings_menu.add_cascade(label="User", menu=user_menu)
            user_menu.add_command(label="My Callsign", command=self.show_user_settings_window)
            # Existing menu items below User
            settings_menu.add_command(label="Configure Radio", command=self.show_radio_settings_window)
            settings_menu.add_command(label="Audio Settings", command=self.show_audio_settings_window)

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
            self.v4_label = tk.Label(self.v4_indicator_frame, text="Robust RX/TX:", fg=self.current_colors["fg"], 
                                     bg=self.current_colors["bg"])
            self.v4_label.pack(side="left")
            self.v4_indicator = tk.Canvas(self.v4_indicator_frame, width=20, height=20, 
                                         bg=self.current_colors["bg"], highlightthickness=0)
            self.v4_indicator.create_rectangle(0, 0, 20, 20, fill="grey", tags="robust_mode_light")
            self.v4_indicator.pack(side="left")

            self.robust_plus_indicator_frame = tk.Frame(self.callsign_frame, bg=self.current_colors["bg"])
            self.robust_plus_indicator_frame.grid(row=1, column=5, padx=10, pady=(0, 5), sticky="e")
            self.robust_plus_label = tk.Label(self.robust_plus_indicator_frame, text="Robust+ RX/TX:", fg=self.current_colors["fg"], 
                                              bg=self.current_colors["bg"])
            self.robust_plus_label.pack(side="left")
            self.robust_plus_indicator = tk.Canvas(self.robust_plus_indicator_frame, width=20, height=20, 
                                                  bg=self.current_colors["bg"], highlightthickness=0)
            self.robust_plus_indicator.create_rectangle(0, 0, 20, 20, fill="grey", tags="robust_plus_mode_light")
            self.robust_plus_indicator.pack(side="left")

# Part 1
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
            self.speed_normal_rb = tk.Radiobutton(self.speed_frame, text="Normal (~15-30 WPM)", variable=self.speed_var, value="normal", 
                                                 fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                                                 selectcolor=self.current_colors["entry_bg"])
            self.speed_normal_rb.pack(side="left", padx=5)
            self.speed_robust_rb = tk.Radiobutton(self.speed_frame, text="Robust (~20-40 WPM)", variable=self.speed_var, value="robust", 
                                                 fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                                                 selectcolor=self.current_colors["entry_bg"])
            self.speed_robust_rb.pack(side="left", padx=5)
            self.speed_robust_plus_rb = tk.Radiobutton(self.speed_frame, text="Robust+ (~40-100 WPM)", variable=self.speed_var, value="robust_plus", 
                                                      fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                                                      selectcolor=self.current_colors["entry_bg"])
            self.speed_robust_plus_rb.pack(side="left", padx=5)

            self.filter_frame = tk.LabelFrame(self.tx_frame, text="Frequency Filter", fg=self.current_colors["fg"], 
                                             bg=self.current_colors["bg"])
            self.filter_frame.pack(anchor="w", pady=5)
            self.filter_none_rb = tk.Radiobutton(self.filter_frame, text="No Filter", variable=self.filter_var, value="none", 
                                                fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                                                selectcolor=self.current_colors["entry_bg"])
            self.filter_none_rb.pack(side="left", padx=5)
            self.filter_900_5100_rb = tk.Radiobutton(self.filter_frame, text="900-5100 Hz", variable=self.filter_var, value="900-5100", 
                                                    fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                                                    selectcolor=self.current_colors["entry_bg"])
            self.filter_900_5100_rb.pack(side="left", padx=5)
            self.filter_800_5200_rb = tk.Radiobutton(self.filter_frame, text="800-5200 Hz", variable=self.filter_var, value="800-5200", 
                                                    fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                                                    selectcolor=self.current_colors["entry_bg"])
            self.filter_800_5200_rb.pack(side="left", padx=5)

            self.rx_frame = tk.Frame(self.main_frame, bg=self.current_colors["bg"])
            self.rx_frame.pack(fill="both", expand=True, pady=(10, 0))
            
            self.rx_header_frame = tk.Frame(self.rx_frame, bg=self.current_colors["bg"])
            self.rx_header_frame.pack(fill="x")
            tk.Label(self.rx_header_frame, text="Received Messages:", fg=self.current_colors["fg"], 
                     bg=self.current_colors["bg"]).pack(side="left", pady=(0, 5))
            self.clear_button = tk.Button(self.rx_header_frame, text="Clear", command=self.clear_receive, 
                                         fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
            self.clear_button.pack(side="right", padx=5, pady=5)

            self.rx_output_frame = tk.Frame(self.rx_frame, bg=self.current_colors["bg"])
            self.rx_output_frame.pack(fill="both", expand=True, pady=(0, 10))

            v_scrollbar = tk.Scrollbar(self.rx_output_frame, orient="vertical")
            v_scrollbar.pack(side="right", fill="y")

            h_scrollbar = tk.Scrollbar(self.rx_output_frame, orient="horizontal")
            h_scrollbar.pack(side="bottom", fill="x")

            self.rx_output = tk.Text(self.rx_output_frame, 
                                    font=tkinter.font.Font(family="Courier", size=10), 
                                    fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], 
                                    insertbackground=self.current_colors["fg"], wrap="none", 
                                    yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            self.rx_output.pack(side="left", fill="both", expand=True)

            v_scrollbar.config(command=self.rx_output.yview)
            h_scrollbar.config(command=self.rx_output.xview)

            self.rx_output.tag_config("sent", foreground="red")
            self.rx_output.tag_config("received", foreground="blue")
            self.rx_output.tag_config("underline", underline=True)

            self.root.bind("<Configure>", self.update_dynamic_width)

            print("GUI built successfully")
        except Exception as e:
            print(f"GUI build error: {e}")
            if not hasattr(self, 'rx_output'):
                self.rx_output = tk.Text(self.root, 
                                       font=tkinter.font.Font(family="Courier", size=10), 
                                       fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], 
                                       insertbackground=self.current_colors["fg"])
                self.rx_output.pack_forget()
            if self.rx_output:
                self.rx_output.insert(tk.END, f"GUI error: {e}\n")

        print("Scheduling audio start")
        self.root.after(100, self.start_audio)

    def toggle_dark_mode(self):
        if self.dark_mode_var.get():
            self.current_colors = {"bg": "#000000", "fg": "#00FF00", "entry_bg": "#1A1A1A", "button_bg": "#000000"}
            self.tx_button.config(fg="#00FF00", bg=self.current_colors["button_bg"], activeforeground="#00FF00", 
                                  activebackground="#444444", highlightbackground="#00FF00", highlightcolor="#00FF00", 
                                  highlightthickness=1)
            self.cq_button.config(fg="#00FF00", bg=self.current_colors["button_bg"], activeforeground="#00FF00", 
                                  activebackground="#444444", highlightbackground="#00FF00", highlightcolor="#00FF00", 
                                  highlightthickness=1)
            self.v4_label.config(fg="#00FF00", bg=self.current_colors["bg"])
            self.robust_plus_label.config(fg="#00FF00", bg=self.current_colors["bg"])
            self.speed_frame.config(fg="#00FF00")
            self.filter_frame.config(fg="#00FF00")
            # Radio buttons: white when not selected, black when selected
            for rb in [self.speed_normal_rb, self.speed_robust_rb, self.speed_robust_plus_rb,
                       self.filter_none_rb, self.filter_900_5100_rb, self.filter_800_5200_rb]:
                rb.config(fg="#FFFFFF", selectcolor="#444444", activeforeground="#000000")
        else:
            self.current_colors = self.light_colors
            self.tx_button.config(fg=self.current_colors["fg"], bg=self.current_colors["button_bg"], 
                                  activeforeground=self.current_colors["fg"], activebackground="#E0E0E0", 
                                  highlightthickness=0)
            self.cq_button.config(fg=self.current_colors["fg"], bg=self.current_colors["button_bg"], 
                                  activeforeground=self.current_colors["fg"], activebackground="#E0E0E0", 
                                  highlightthickness=0)
            self.v4_label.config(fg=self.current_colors["fg"], bg=self.current_colors["bg"])
            self.robust_plus_label.config(fg=self.current_colors["fg"], bg=self.current_colors["bg"])
            self.speed_frame.config(fg=self.current_colors["fg"])
            self.filter_frame.config(fg=self.current_colors["fg"])
            # Revert radio buttons to default light mode colors
            for rb in [self.speed_normal_rb, self.speed_robust_rb, self.speed_robust_plus_rb,
                       self.filter_none_rb, self.filter_900_5100_rb, self.filter_800_5200_rb]:
                rb.config(fg=self.current_colors["fg"], selectcolor=self.current_colors["entry_bg"], 
                          activeforeground=self.current_colors["fg"])

        self.root.configure(bg=self.current_colors["bg"])
        self.main_frame.configure(bg=self.current_colors["bg"])
        self.callsign_frame.configure(bg=self.current_colors["bg"])
        self.tx_frame.configure(bg=self.current_colors["bg"])
        self.rx_frame.configure(bg=self.current_colors["bg"])
        self.rx_header_frame.configure(bg=self.current_colors["bg"])
        self.speed_frame.configure(bg=self.current_colors["bg"])
        self.filter_frame.configure(bg=self.current_colors["bg"])
        self.v4_indicator_frame.configure(bg=self.current_colors["bg"])
        self.robust_plus_indicator_frame.configure(bg=self.current_colors["bg"])
        self.rx_output_frame.configure(bg=self.current_colors["bg"])
        self.button_frame.configure(bg=self.current_colors["bg"])
        
        for widget in (self.callsign_frame.winfo_children() + 
                       self.tx_frame.winfo_children() + 
                       self.rx_frame.winfo_children() + 
                       self.rx_header_frame.winfo_children() +
                       self.speed_frame.winfo_children() +
                       self.filter_frame.winfo_children() +
                       self.rx_output_frame.winfo_children() +
                       self.robust_plus_indicator_frame.winfo_children()):
            if isinstance(widget, (tk.Label, tk.Checkbutton, tk.Radiobutton)):
                widget.configure(fg=self.current_colors["fg"], bg=self.current_colors["bg"])
            elif isinstance(widget, tk.Entry):
                widget.configure(fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], 
                                insertbackground=self.current_colors["fg"])
            elif isinstance(widget, tk.Button) and widget not in [self.tx_button, self.cq_button]:
                widget.configure(fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
        if self.rx_output:
            self.rx_output.configure(fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], 
                                    insertbackground=self.current_colors["fg"])
        self.clear_button.configure(fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
        self.dark_mode_check.configure(selectcolor=self.current_colors["entry_bg"])

# Part 2
    def update_dynamic_width(self, event):
        if hasattr(self, 'rx_output'):
            pixel_width = self.rx_output.winfo_width()
            if pixel_width > 0:
                font = tkinter.font.Font(family="Courier", size=10)
                char_width = font.measure("0")
                if char_width > 0:
                    chars_per_line = pixel_width // char_width
                    print(f"Dynamic width updated - Pixel width: {pixel_width}, Characters per line: {chars_per_line}")
                    self.dynamic_width = max(1, chars_per_line)
                else:
                    print("Warning: Invalid font width detected, using fallback width of 200 characters.")
                    self.dynamic_width = 200
            else:
                self.dynamic_width = 200

    def clear_receive(self):
        if self.rx_output:
            self.rx_output.delete(1.0, tk.END)
        self.receive_buffer = []
        self.temp_receive_buffer = []

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

    def load_settings(self):
        settings = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r") as f:
                    for line in f:
                        if line.strip():
                            key, value = line.strip().split("=", 1)
                            if key == "baud_rate":
                                settings[key] = value
                            elif key == "input_volume":
                                settings[key] = float(value)
                            elif key == "sensitivity":
                                settings[key] = float(value) if value else 50.0
                            elif key in ["rts", "dtr", "mode_usb", "mode_usb_digital", "mode_fm", "ptt_rts", "ptt_dtr"]:
                                settings[key] = value.lower() == "true"
                            elif key in ["input_device", "output_device"]:
                                settings[key] = int(value) if value != "-1" else -1
                            else:
                                settings[key] = value
            except Exception as e:
                print(f"Error loading settings: {e}")
        return settings

    def save_settings(self):
        settings = {
            "radio": self.radio.get(),
            "usb_address": self.usb_address.get(),
            "mode_usb": self.mode_usb.get(),
            "mode_usb_digital": self.mode_usb_digital.get(),
            "mode_fm": self.mode_fm.get(),
            "serial_port": self.serial_port.get(),
            "baud_rate": self.baud_rate.get(),
            "rts": self.rts.get(),
            "dtr": self.dtr.get(),
            "ptt_port": self.ptt_port.get(),
            "ptt_rts": self.ptt_rts.get(),
            "ptt_dtr": self.ptt_dtr.get(),
            "input_device": self.input_device_index.get(),
            "output_device": self.output_device_index.get(),
            "input_volume": self.input_volume.get(),
            "sensitivity": self.sensitivity.get(),
            "speed": self.speed_var.get(),
            "filter": self.filter_var.get(),
        }
        try:
            with open(SETTINGS_FILE, "w") as f:
                for key, value in settings.items():
                    if key in ["rts", "dtr", "mode_usb", "mode_usb_digital", "mode_fm", "ptt_rts", "ptt_dtr"]:
                        f.write(f"{key}={value}\n")
                    else:
                        f.write(f"{key}={value}\n")
            print("Radio settings saved successfully")
            if self.rx_output:
                self.rx_output.insert(tk.END, "Radio settings saved successfully\n")
        except Exception as e:
            if self.rx_output:
                self.rx_output.insert(tk.END, f"Error saving settings: {str(e)}\n")
            print(f"Error saving settings: {e}")

    def get_audio_devices(self):
        p = pyaudio.PyAudio()
        input_devices = {}
        output_devices = {}
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            name = dev["name"]
            index = dev["index"]
            if dev["maxInputChannels"] > 0:
                input_devices[index] = name
            if dev["maxOutputChannels"] > 0:
                output_devices[index] = name
        p.terminate()
        print(f"Available input devices: {input_devices}")
        print(f"Available output devices: {output_devices}")
        if not input_devices or not output_devices:
            raise Exception("No audio devices available. Check hardware or close other audio programs.")
        return input_devices, output_devices

    def get_com_ports(self):
        return ["NONE"] + [port.device for port in serial.tools.list_ports.comports()]

    def get_default_address(self, radio):
        default_addresses = {
            "Icom IC-703": "68",
            "Icom IC-705": "94",
            "Generic": ""
        }
        return default_addresses.get(radio, "")

    def start_audio(self):
        print("Starting audio")
        try:
            self.p = pyaudio.PyAudio()
            self.input_devices, self.output_devices = self.get_audio_devices()
            input_idx = self.input_device_index.get() if self.input_device_index.get() >= 0 else None
            output_idx = self.output_device_index.get() if self.output_device_index.get() >= 0 else None
            
            print(f"Attempting to open audio streams - Input device: {input_idx}, Output device: {output_idx}")
            self.stream_out = self.p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, output=True, 
                                        output_device_index=output_idx)
            self.stream_in = self.p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True, 
                                       frames_per_buffer=CHUNK, input_device_index=input_idx)
            self.running = True
            self.rx_thread = threading.Thread(target=self.receive_loop, daemon=True)
            self.rx_thread.start()
            self.tx_thread = threading.Thread(target=self.transmit_loop, daemon=True)
            self.tx_thread.start()
            if self.rx_output:
                self.rx_output.insert(tk.END, "Audio started successfully\n")
            print("Audio threads started")
        except pyaudio.PyError as e:
            if self.rx_output:
                self.rx_output.insert(tk.END, f"Audio error: Another program may have control of the audio. Error: {str(e)}\n")
            self.running = False
            print(f"Audio startup error: {str(e)}. Please close other audio programs and try again.")
            if self.root and self.rx_output:
                self.root.after(0, lambda: messagebox.showwarning("Audio Error", 
                                f"Audio setup failed: {str(e)}. Another program may have control of the audio. Close other audio programs and try again."))
            if self.p:
                self.p.terminate()
                self.p = None
        except Exception as e:
            if self.rx_output:
                self.rx_output.insert(tk.END, f"Audio failed to start: {str(e)}\n")
            self.running = False
            print(f"Audio startup error: {e}")
            if self.root and self.rx_output:
                self.root.after(0, lambda: messagebox.showwarning("Audio Warning", 
                                f"Audio setup failed: {str(e)}. Continuing without audio."))
            if self.p:
                self.p.terminate()
                self.p = None

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
        if not bits or len(bits) < block_size:
            print(f"deinterleave: Input too short ({len(bits)} bits), returning unchanged")
            return bits  # Return as-is if empty or too short
        num_blocks = (len(bits) + block_size - 1) // block_size  # Ceiling division
        padded_length = num_blocks * block_size
        padded_bits = bits + [0] * (padded_length - len(bits))  # Pad with zeros if needed
        deinterleaved = [0] * padded_length
        for i in range(len(padded_bits)):
            new_index = (i // 4 + (i % 4) * 4) % padded_length  # Ensure index stays in bounds
            deinterleaved[new_index] = padded_bits[i]
        print(f"deinterleave: Processed {len(bits)} bits into {len(deinterleaved)} deinterleaved bits")
        return deinterleaved[:len(bits)]  # Trim back to original length

    def detect_v4_preamble(self, data):
        preamble_freqs = [(1500, 2000), (3000, 3500), (5000, 4500), (7000, 6500), 
                         (2000, 2500), (4000, 4500), (6000, 5500)]
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

    def detect_robust_plus_preamble(self, data):
        preamble_freqs = [(2000, 2500), (4000, 3500), (6000, 5500), (8000, 7500), 
                         (1000, 1500), (3000, 2500), (5000, 4500)]
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

    def decode_ofdm_frequency(self, data):
        freqs = np.abs(fft(data)[:len(data)//2])
        freq_axis = np.linspace(0, SAMPLE_RATE // 2, len(freqs))
        peak_idx = np.argmax(freqs)
        peak_freq = freq_axis[peak_idx]
        tones = 16
        if np.max(freqs) > self.sensitivity.get() / 100.0 * np.max(freqs):
            for freq_idx in range(tones):
                expected_freq = 1200 + freq_idx * 150
                if abs(peak_freq - expected_freq) < 100:
                    return expected_freq
        return None

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

# Part 3
    def send_ack_nack(self, packet_id, success):
        if not self.stream_out:
            return
        self.set_v4_tx_indicator() if self.speed_var.get() == "robust" else self.set_robust_plus_tx_indicator()
        ack_sound = self.generate_sound(CHAR_SOUNDS["K" if success else "N"], 0.05)
        audio_data = np.concatenate((ack_sound, np.zeros(int(SAMPLE_RATE * 0.025))))
        self.stream_out.write(audio_data.astype(np.float32).tobytes())
        self.reset_robust_plus_indicator() if self.speed_var.get() == "robust_plus" else self.reset_indicator()

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
            self.tx_queue.append((full_text, packet_id, False))
        elif speed == "robust_plus":
            packet_id = (self.last_packet_id + 1) % 16
            self.last_packet_id = packet_id
            self.tx_queue.append((full_text, packet_id, True))
        else:
            self.tx_queue.append((full_text, None, False))
        self.tx_input.delete(0, tk.END)
        self.start_radio_transmission()
        self.tx_button.config(bg="red")

    def transmit_loop(self):
        while self.running:
            if self.tx_queue and self.stream_out:
                self.tx_button.config(bg="red")
                self.cq_button.config(bg="red")
                full_text, packet_id, is_robust_plus = self.tx_queue.pop(0)
                speed = self.speed_var.get()
                duration_map = {"normal": 0.1, "robust": 0.05, "robust_plus": 0.01}
                duration = duration_map[speed]
                gap_duration = duration / 2

                self.start_radio_transmission()
                audio_data = np.array([], dtype=np.float32)
                if speed in ["robust", "robust_plus"]:
                    if is_robust_plus:
                        preamble_audio = np.concatenate([self.generate_sound(CHAR_SOUNDS[num], 0.05) 
                                                        for num in "2468135"])
                        self.set_robust_plus_tx_indicator()
                        ofdm_symbols = self.encode_ofdm_packet(full_text, packet_id)
                        audio_data = preamble_audio
                        for freq, duration in ofdm_symbols:
                            sound = self.generate_ofdm_sound(freq, duration)
                            audio_data = np.concatenate((audio_data, sound, np.zeros(int(SAMPLE_RATE * gap_duration / 10))))
                        if self.rx_output:
                            self.rx_output.insert(tk.END, f"Robust+ TX (Preamble: 2468135): " + full_text + "\n", "sent")
                    else:
                        preamble_audio = np.concatenate([self.generate_sound(CHAR_SOUNDS[num], 0.05) 
                                                        for num in "1357924"])
                        self.set_v4_tx_indicator()
                        symbols = self.encode_v4_packet(full_text, packet_id)
                        audio_data = preamble_audio
                        for sym in symbols:
                            pattern = CHAR_SOUNDS.get(sym, CHAR_SOUNDS[" "])
                            sound = self.generate_sound(pattern, 0.05)
                            audio_data = np.concatenate((audio_data, sound, np.zeros(int(SAMPLE_RATE * gap_duration))))
                        if self.rx_output:
                            self.rx_output.insert(tk.END, f"Robust TX (Preamble: 1357924): " + full_text + "\n", "sent")
                    self.stream_out.write(audio_data.astype(np.float32).tobytes())
                    self.reset_robust_plus_indicator() if is_robust_plus else self.reset_indicator()
                    self.stop_radio_transmission()
                    time.sleep(0.2)
                    if self.rx_ack == "N":
                        self.tx_queue.append((full_text, packet_id, is_robust_plus))
                    self.rx_ack = None
                else:
                    encoded_symbols = self.encode_fec(full_text)
                    for symbol in encoded_symbols:
                        char = REVERSE_MAP.get(symbol, " ")
                        pattern = WORD_SOUNDS.get(char, CHAR_SOUNDS.get(char, []))
                        sound = self.generate_sound(pattern, duration)
                        audio_data = np.concatenate((audio_data, sound, np.zeros(int(SAMPLE_RATE * gap_duration))))
                    self.stream_out.write(audio_data.astype(np.float32).tobytes())
                    if self.rx_output:
                        self.rx_output.insert(tk.END, "Sent: " + full_text + "\n", "sent")
                    self.stop_radio_transmission()
                self.tx_button.config(bg=self.current_colors["button_bg"])
                self.cq_button.config(bg=self.current_colors["button_bg"])
            time.sleep(0.001)  # Properly indented inside while loop

    def send_cq(self):  # Correctly indented at class level
        if not self.stream_out:
            messagebox.showwarning("Warning", "Audio not started!")
            return
        my_call = self.my_callsign.get().upper()
        if not my_call:
            messagebox.showwarning("Warning", "Enter your callsign!")
            return
        speed = self.speed_var.get()
        single_cq = f"CQ CQ CQ DE {my_call}"
        
        self.start_radio_transmission()
        self.cq_button.config(bg="red")
        audio_data = np.array([], dtype=np.float32)
        duration_map = {"normal": 0.1, "robust": 0.05, "robust_plus": 0.01}
        duration = duration_map[speed]
        gap_duration = duration / 2

        if speed == "robust":
            packet_id = (self.last_packet_id + 1) % 16
            self.last_packet_id = packet_id
            preamble_audio = np.concatenate([self.generate_sound(CHAR_SOUNDS[num], 0.05) 
                                            for num in "1357924"])
            self.set_v4_tx_indicator()
            symbols = self.encode_v4_packet(single_cq, packet_id)
            audio_data = preamble_audio
            for sym in symbols:
                pattern = CHAR_SOUNDS.get(sym, CHAR_SOUNDS[" "])
                sound = self.generate_sound(pattern, 0.05)
                audio_data = np.concatenate((audio_data, sound, np.zeros(int(SAMPLE_RATE * gap_duration))))
            if self.rx_output:
                self.rx_output.insert(tk.END, f"Robust TX (Preamble: 1357924): " + single_cq + "\n", "sent")
        elif speed == "robust_plus":
            packet_id = (self.last_packet_id + 1) % 16
            self.last_packet_id = packet_id
            preamble_audio = np.concatenate([self.generate_sound(CHAR_SOUNDS[num], 0.05) 
                                            for num in "2468135"])
            self.set_robust_plus_tx_indicator()
            ofdm_symbols = self.encode_ofdm_packet(single_cq, packet_id)
            audio_data = preamble_audio
            for freq, duration in ofdm_symbols:
                sound = self.generate_ofdm_sound(freq, duration)
                audio_data = np.concatenate((audio_data, sound, np.zeros(int(SAMPLE_RATE * gap_duration / 10))))
            if self.rx_output:
                self.rx_output.insert(tk.END, f"Robust+ TX (Preamble: 2468135): " + single_cq + "\n", "sent")
        else:
            cq_text = f"{single_cq} {single_cq}"
            encoded_symbols = self.encode_fec(cq_text)
            for symbol in encoded_symbols:
                char = REVERSE_MAP.get(symbol, " ")
                pattern = WORD_SOUNDS.get(char, CHAR_SOUNDS.get(char, []))
                sound = self.generate_sound(pattern, duration)
                audio_data = np.concatenate((audio_data, sound, np.zeros(int(SAMPLE_RATE * gap_duration))))
            if self.rx_output:
                self.rx_output.insert(tk.END, "Sent: " + cq_text + "\n", "sent")

        self.stream_out.write(audio_data.astype(np.float32).tobytes())
        if speed in ["robust", "robust_plus"]:
            self.reset_robust_plus_indicator() if speed == "robust_plus" else self.reset_indicator()
        self.stop_radio_transmission()
        self.cq_button.config(bg=self.current_colors["button_bg"])

    def test_ptt_connection(self):
        if self.serial_port.get() != "NONE" and self.baud_rate.get():
            if not self.baud_rate.get().isdigit():
                messagebox.showwarning("Radio Warning", "Baud rate must be a valid number.")
                return
            self.start_radio_transmission()
            self.root.after(1000, lambda: [self.stop_radio_transmission(), self.tx_button.config(bg=self.current_colors["button_bg"]), self.cq_button.config(bg=self.current_colors["button_bg"])])
        else:
            messagebox.showwarning("Radio Warning", "No serial port or baud rate selected for radio communication.")

    def start_radio_transmission(self):
        if self.serial_port.get() != "NONE" and self.baud_rate.get():
            try:
                radio = self.radio.get()
                raw_address = self.usb_address.get().replace('H', '').replace('h', '')
                if not raw_address:
                    raise ValueError("No address specified for radio communication. Please set a valid hex address (e.g., '68' for IC-703).")
                if len(raw_address) != 2 or not all(c in '0123456789ABCDEFabcdef' for c in raw_address):
                    raise ValueError(f"Invalid address '{raw_address}'. Must be a 2-character hexadecimal value (e.g., '68', '94').")
                address = raw_address.upper()

                if not self.radio_serial or not self.radio_serial.is_open:
                    self.radio_serial = serial.Serial(
                        port=self.serial_port.get(),
                        baudrate=int(self.baud_rate.get()) if self.baud_rate.get().isdigit() else 9600,
                        timeout=0.1,
                        rtscts=self.rts.get(),
                        dsrdtr=self.dtr.get()
                    )
                    print(f"Radio serial connection established on {self.serial_port.get()} at {self.baud_rate.get()} baud")
                    if self.rx_output:
                        self.rx_output.insert(tk.END, f"Radio connection established on {self.serial_port.get()}\n")

                print(f"Using address: {address}")
                if self.rx_output:
                    self.rx_output.insert(tk.END, f"Using address: {address}\n")

                if radio in ["Icom IC-703", "Icom IC-705"]:
                    cmd = bytes.fromhex(f"FE FE {address} E0 1C 00 FD")
                else:
                    cmd = bytes.fromhex(f"FE FE {address} E0 1C 00 FD")
                self.radio_serial.write(cmd)
                print(f"Sent PTT ON command: {cmd.hex()}")
                if self.rx_output:
                    self.rx_output.insert(tk.END, f"Sent PTT ON command: {cmd.hex()}\n")
            except (serial.SerialException, ValueError) as e:
                print(f"Failed to start radio transmission: {str(e)}")
                if self.rx_output:
                    self.rx_output.insert(tk.END, f"Failed to start radio transmission: {str(e)}\n")
                if self.radio_serial:
                    self.radio_serial.close()
                    self.radio_serial = None
            except Exception as e:
                print(f"Unexpected error in start_radio_transmission: {str(e)}")
                if self.rx_output:
                    self.rx_output.insert(tk.END, f"Unexpected error in start_radio_transmission: {str(e)}\n")
                if self.radio_serial:
                    self.radio_serial.close()
                    self.radio_serial = None

    def stop_radio_transmission(self):
        if self.radio_serial and self.radio_serial.is_open:
            try:
                radio = self.radio.get()
                address = self.usb_address.get().replace('H', '').replace('h', '')
                if not address:
                    raise ValueError("No address specified for radio communication.")
                if len(address) != 2 or not all(c in '0123456789ABCDEFabcdef' for c in address):
                    raise ValueError(f"Invalid address '{address}'. Must be a 2-character hexadecimal value.")
                address = address.upper()

                if radio in ["Icom IC-703", "Icom IC-705"]:
                    cmd = bytes.fromhex(f"FE FE {address} E0 1C 01 FD")
                else:
                    cmd = bytes.fromhex(f"FE FE {address} E0 1C 01 FD")
                self.radio_serial.write(cmd)
                print(f"Sent PTT OFF command: {cmd.hex()}")
                if self.rx_output:
                    self.rx_output.insert(tk.END, f"Sent PTT OFF command: {cmd.hex()}\n")
            except serial.SerialException as e:
                print(f"Failed to stop radio transmission: {str(e)}")
                if self.rx_output:
                    self.rx_output.insert(tk.END, f"Failed to stop radio transmission: {str(e)}\n")
            finally:
                if self.radio_serial:
                    self.radio_serial.close()
                    self.radio_serial = None

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
        if speed == "robust":
            while len(symbol_list) % 23 != 0:
                symbol_list.append(SYMBOL_MAP[" "])
            blocks = [symbol_list[i:i+23] for i in range(0, len(symbol_list), 23)]
            encoded_blocks = []
            for block in blocks:
                encoded = RSCodec.encode(data=block)
                encoded_blocks.extend(encoded)
            return encoded_blocks
        elif speed == "robust_plus":
            ofdm_symbols = self.encode_ofdm_packet(text, 0)
            return ofdm_symbols
        return symbol_list

    def decode_fec(self, symbols):
        speed = self.speed_var.get()
        if speed == "robust":
            blocks = [symbols[i:i+31] for i in range(0, len(symbols), 31)]
            decoded_text = ""
            for block in blocks:
                if len(block) == 31:
                    try:
                        if not all(isinstance(s, int) for s in block):
                            print(f"Invalid data in block: {block}")
                            decoded_text += "[ERROR] Invalid symbol data "
                            continue
                        decoded = RSCodec.decode(data=block)[0]
                        decoded_text += "".join(REVERSE_MAP.get(sym, " ") for sym in decoded)
                    except ReedSolomonError:
                        decoded_text += "[ERROR] "
            return decoded_text.strip()
        elif speed == "robust_plus":
            if not isinstance(symbols, list) or not all(isinstance(s, tuple) and len(s) == 2 for s in symbols):
                print(f"Invalid OFDM symbols: {symbols}")
                return "[ERROR] Invalid OFDM data"
            packet_id, decoded_text = self.decode_ofdm_packet(symbols)
            return decoded_text.strip()
        return "".join(REVERSE_MAP.get(sym, " ") for sym in symbols).strip()

    def encode_ofdm_packet(self, text, packet_id):
        bits = [int(b) for char in text for b in bin(ord(char))[2:].zfill(8)]
        encoded_bits = self.convolutional_encode(bits)
        interleaved_bits = self.interleave(encoded_bits)
        rs_encoded = RSCodec(16).encode(data=interleaved_bits)
        ofdm_symbols = []
        tones = 16
        symbol_rate = 41.6
        for i in range(0, len(rs_encoded), int(symbol_rate)):
            chunk = rs_encoded[i:i+int(symbol_rate)]
            if chunk:
                freq_idx = sum(chunk) % tones
                freq = 1200 + freq_idx * 150
                duration = 0.01
                ofdm_symbols.append((freq, duration))
        crc = crc16(text.encode('utf-8'))
        crc_bits = [int(b) for b in bin(crc)[2:].zfill(16)]
        for i in range(0, len(crc_bits), int(symbol_rate)):
            chunk = crc_bits[i:i+int(symbol_rate)]
            if chunk:
                freq_idx = sum(chunk) % tones
                freq = 1200 + freq_idx * 150
                duration = 0.01
                ofdm_symbols.append((freq, duration))
        return ofdm_symbols

    def decode_ofdm_packet(self, symbols):
        if len(symbols) < 10:
            print(f"decode_ofdm_packet: Not enough symbols ({len(symbols)}), skipping")
            return (0, "[ERROR] Insufficient data")
        decoded_bits = []
        tones = 16
        symbol_rate = 41.6
        for freq, _ in symbols:
            freq_idx = int((freq - 1200) / 150) % tones
            bits = [int(b) for b in bin(freq_idx)[2:].zfill(int(np.log2(tones)))]
            decoded_bits.extend(bits[:int(symbol_rate)])
        deinterleaved_bits = self.deinterleave(decoded_bits)
        viterbi_decoded = self.viterbi_decode(deinterleaved_bits)
        try:
            rs_decoded = RSCodec(16).decode(data=viterbi_decoded)[0]
            text = ""
            for i in range(0, len(rs_decoded), 8):
                byte = rs_decoded[i:i+8]
                if len(byte) == 8:
                    char = chr(sum(b << (7-j) for j, b in enumerate(byte)))
                    text += char
            print(f"decode_ofdm_packet: Decoded text before CRC check: '{text}'")
            crc_pos = len(text) - 2
            if crc_pos <= 0 or not text[crc_pos:].strip():
                print(f"decode_ofdm_packet: Invalid CRC position or content (crc_pos={crc_pos}, text='{text}')")
                return (0, "[ERROR] Invalid CRC data")
            crc_text = text[crc_pos:]
            if not all(c in '0123456789ABCDEFabcdef' for c in crc_text):
                print(f"decode_ofdm_packet: Invalid hex characters in CRC: '{crc_text}'")
                return (0, "[ERROR] Invalid CRC format")
            crc_received = int(crc_text, 16)
            crc_calculated = crc16(text[:crc_pos].encode('utf-8'))
            if crc_received != crc_calculated:
                print(f"decode_ofdm_packet: CRC mismatch (received={crc_received}, calculated={crc_calculated})")
                return (0, "[ERROR] CRC mismatch")
            return (0, text)
        except ReedSolomonError:
            print("decode_ofdm_packet: ReedSolomonError during decoding")
            return (0, "[ERROR] Decoding failed")
        except ValueError as e:
            print(f"decode_ofdm_packet: ValueError during CRC conversion: {str(e)}")
            return (0, "[ERROR] Invalid CRC format")

# Part 4
    def generate_ofdm_sound(self, freq, duration):
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
        signal = np.sin(2 * np.pi * freq * t)
        return signal / np.max(np.abs(signal))

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
            b, a = self.butter_bandpass(900, 2400, SAMPLE_RATE)
            return lfilter(b, a, data)
        elif filter_type == "800-5200":
            b, a = self.butter_bandpass(800, 2400, SAMPLE_RATE)
            return lfilter(b, a, data)
        return data

    def decode_audio(self, data):
        freqs = np.abs(fft(data)[:len(data)//2])
        freq_axis = np.linspace(0, SAMPLE_RATE // 2, len(freqs))
        peak_idx = np.argmax(freqs)
        peak_freq = freq_axis[peak_idx]
        
        sensitivity_value = self.sensitivity.get() / 100.0
        min_amplitude = 0.01 + (sensitivity_value * 0.99)
        print(f"Decode audio - Sensitivity: {sensitivity_value}, Min Amplitude: {min_amplitude}, Peak Amplitude: {np.max(freqs)}")
        
        if np.max(freqs) < min_amplitude * np.max(freqs):
            return None

        tolerance = 300
        speed = self.speed_var.get()
        if speed in ["robust", "robust_plus"]:
            if speed == "robust":
                for sym in V4_REVERSE_MAP:
                    pattern = CHAR_SOUNDS[sym]
                    start_freq = pattern[0][0]
                    if abs(peak_freq - start_freq) < tolerance:
                        return sym
            else:  # robust_plus
                tones = 16
                duration = 0.01  # Match encode_ofdm_packet duration
                for freq_idx in range(tones):
                    expected_freq = 1200 + freq_idx * 150
                    if abs(peak_freq - expected_freq) < 100:
                        return (expected_freq, duration)
        else:  # normal mode
            for word, pattern in WORD_SOUNDS.items():
                start_freq = pattern[0][0]
                if abs(peak_freq - start_freq) < tolerance:
                    return SYMBOL_MAP[word]
            for char, pattern in CHAR_SOUNDS.items():
                start_freq = pattern[0][0]
                if abs(peak_freq - start_freq) < tolerance:
                    return SYMBOL_MAP[char]
        return None

    def set_v4_tx_indicator(self):
        if self.indicator_timeout is not None:
            self.root.after_cancel(self.indicator_timeout)
        self.v4_indicator.itemconfig("robust_mode_light", fill="red")

    def set_v4_rx_indicator(self):
        self.v4_indicator.itemconfig("robust_mode_light", fill="green")
        if self.indicator_timeout is not None:
            self.root.after_cancel(self.indicator_timeout)
        self.indicator_timeout = self.root.after(3000, self.reset_indicator)

    def set_robust_plus_tx_indicator(self):
        if self.indicator_timeout is not None:
            self.root.after_cancel(self.indicator_timeout)
        self.robust_plus_indicator.itemconfig("robust_plus_mode_light", fill="orange")

    def set_robust_plus_rx_indicator(self):
        self.robust_plus_indicator.itemconfig("robust_plus_mode_light", fill="yellow")
        if self.indicator_timeout is not None:
            self.root.after_cancel(self.indicator_timeout)
        self.indicator_timeout = self.root.after(3000, self.reset_robust_plus_indicator)

    def reset_indicator(self):
        self.v4_indicator.itemconfig("robust_mode_light", fill="grey")
        self.indicator_timeout = None

    def reset_robust_plus_indicator(self):
        self.robust_plus_indicator.itemconfig("robust_plus_mode_light", fill="grey")
        self.indicator_timeout = None

    def receive_loop(self):
        buffer = []
        gap_map = {"normal": 0.05, "robust": 0.025, "robust_plus": 0.005}
        duration_map = {"normal": 0.1, "robust": 0.05, "robust_plus": 0.01}
        preamble_detected = False
        last_update_time = time.time()
        update_interval = 2.0

        while self.running:
            try:
                speed = self.speed_var.get()
                if speed not in duration_map:
                    raise ValueError(f"Invalid speed setting: '{speed}'. Valid options are 'normal', 'robust', 'robust_plus'.")
                
                duration = duration_map[speed]
                chunk_size = int(SAMPLE_RATE * (duration + gap_map[speed] * 2))
                data = self.stream_in.read(chunk_size, exception_on_overflow=False)
                if not data:
                    print("No audio data received, skipping chunk.")
                    time.sleep(0.01)
                    continue
                data = np.frombuffer(data, dtype=np.float32)
                gain = self.input_volume.get() / 100.0
                print(f"Receive loop - Input volume: {gain}, Peak amplitude: {np.max(np.abs(data))}")
                data = data * gain
                data = self.apply_filter(data)

                if not preamble_detected and len(buffer) == 0:
                    if self.detect_v4_preamble(data[:int(SAMPLE_RATE * 0.4)]):
                        self.is_v4_mode = True
                        self.set_v4_rx_indicator()
                        preamble_detected = True
                        continue
                    elif self.detect_robust_plus_preamble(data[:int(SAMPLE_RATE * 0.4)]):
                        self.is_v4_mode = True
                        self.set_robust_plus_rx_indicator()
                        preamble_detected = True
                        continue

                symbol = self.decode_audio(data)
                if symbol is not None:
                    if speed in ["robust", "robust_plus"]:
                        buffer.append(symbol)
                        expected_buffer_size = 31 if speed == "robust" else 30  # Increased for Robust+
                        print(f"Receive loop - Buffer size: {len(buffer)}/{expected_buffer_size}")
                        if len(buffer) >= expected_buffer_size:
                            result = (self.decode_v4_packet(buffer) if speed == "robust" 
                                     else self.decode_ofdm_packet(buffer))
                            if result is None or len(result) != 2:
                                print(f"Receive loop - Invalid decode result: {result}")
                                buffer.clear()
                                preamble_detected = False
                                self.is_v4_mode = False
                                self.reset_robust_plus_indicator() if speed == "robust_plus" else self.reset_indicator()
                                continue
                            packet_id, decoded_text = result
                            if decoded_text and packet_id not in self.packet_buffer:
                                self.packet_buffer[packet_id] = decoded_text
                                preamble = "1357924" if speed == "robust" else "2468135"
                                self.display_received_text(f"{'Robust' if speed == 'robust' else 'Robust+'} RX (Preamble: {preamble}): " + decoded_text + "\n", "received")
                                self.send_ack_nack(packet_id, True)
                                self.rx_ack = "K"
                            elif not decoded_text:
                                self.send_ack_nack(packet_id, False)
                                self.rx_ack = "N"
                            buffer.clear()
                            preamble_detected = False
                            self.is_v4_mode = False
                            self.reset_robust_plus_indicator() if speed == "robust_plus" else self.reset_indicator()
                    else:  # normal mode
                        buffer.append(symbol)
                        if len(buffer) >= 1:
                            decoded_text = self.decode_fec(buffer[:1])
                            self.temp_receive_buffer.extend(decoded_text)
                            print(f"Decoded text accumulated: {decoded_text}, Buffer length: {len(''.join(self.temp_receive_buffer))}")
                            buffer = buffer[1:]
                        current_time = time.time()
                        if not hasattr(self, 'dynamic_width'):
                            self.update_dynamic_width(None)
                        buffer_text = "".join(self.temp_receive_buffer)
                        if (len(buffer_text) >= self.dynamic_width or 
                            current_time - last_update_time >= update_interval or not self.running):
                            if self.temp_receive_buffer:
                                self.display_received_text(buffer_text + "\n")
                                self.temp_receive_buffer = []
                            last_update_time = current_time
                time.sleep(0.01)
            except Exception as e:
                if self.running:
                    if self.rx_output:
                        self.rx_output.insert(tk.END, f"Receive loop error: {str(e)}\n")
                    print(f"Receive loop error: {e}")
                break

# Part 5
    def display_received_text(self, text, tag=""):
        if not self.rx_output:
            return

        if not hasattr(self, 'dynamic_width'):
            self.update_dynamic_width(None)
        widget_width = self.dynamic_width

        print(f"Display received text - widget_width: {widget_width}, text length: {len(text)}, text: {text[:50]}...")

        for char in text:
            self.receive_buffer.append(char)
            if len(self.receive_buffer) >= widget_width:
                self.rx_output.insert(tk.END, ''.join(self.receive_buffer) + "\n", tag)
                self.receive_buffer = []

        if self.receive_buffer:
            remaining_text = ''.join(self.receive_buffer)
            if remaining_text:
                self.rx_output.insert(tk.END, remaining_text + "\n", tag)
            self.receive_buffer = []

        self.rx_output.see(tk.END)

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
        if self.radio_serial and self.radio_serial.is_open:
            self.stop_radio_transmission()
        if self.p:
            try:
                self.p.terminate()
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")
            self.p = None
        self.root.destroy()
        print("Application closed")

    def show_user_settings_window(self):
        user_window = tk.Toplevel(self.root)
        user_window.title("User Settings")
        user_window.geometry("300x150")
        user_window.resizable(False, False)

        tk.Label(user_window, text="My Callsign:", fg=self.current_colors["fg"], 
                 bg=self.current_colors["bg"]).pack(pady=5)
        callsign_entry = tk.Entry(user_window, width=15, fg=self.current_colors["fg"], 
                                 bg=self.current_colors["entry_bg"], insertbackground=self.current_colors["fg"])
        callsign_entry.pack(pady=5)
        callsign_entry.insert(0, self.my_callsign_value)  # Pre-fill with current callsign

        def save_user_settings():
            new_callsign = callsign_entry.get().strip().upper()
            if new_callsign:
                self.my_callsign_value = new_callsign
                self.save_callsign(new_callsign)
                self.my_callsign.config(state="normal")  # Enable to update
                self.my_callsign.delete(0, tk.END)
                self.my_callsign.insert(0, new_callsign)
                self.my_callsign.config(state="disabled")  # Disable again
                if self.rx_output:
                    self.rx_output.insert(tk.END, f"Callsign updated to: {new_callsign}\n")
                user_window.destroy()
            else:
                messagebox.showwarning("Warning", "Callsign cannot be empty!", parent=user_window)

        tk.Button(user_window, text="Save", command=save_user_settings, 
                  fg=self.current_colors["fg"], bg=self.current_colors["button_bg"]).pack(pady=10)

    def show_audio_settings_window(self):
        audio_window = tk.Toplevel(self.root)
        audio_window.title("Audio Settings")
        audio_window.geometry("600x400")
        audio_window.resizable(False, False)

        tk.Label(audio_window, text="Audio Input:", fg=self.current_colors["fg"], 
                 bg=self.current_colors["bg"]).pack(pady=5)
        input_options = list(self.input_devices.values())
        self.input_device = tk.StringVar(value=list(self.input_devices.values())[self.input_device_index.get()] 
                                        if self.input_device_index.get() >= 0 else "Default")
        tk.OptionMenu(audio_window, self.input_device, "Default", *input_options, 
                     command=self.update_audio_devices).pack(pady=5)

        tk.Label(audio_window, text="Audio Output:", fg=self.current_colors["fg"], 
                 bg=self.current_colors["bg"]).pack(pady=5)
        output_options = list(self.output_devices.values())
        self.output_device = tk.StringVar(value=list(self.output_devices.values())[self.output_device_index.get()] 
                                         if self.output_device_index.get() >= 0 else "Default")
        tk.OptionMenu(audio_window, self.output_device, "Default", *output_options, 
                     command=self.update_audio_devices).pack(pady=5)

        tk.Label(audio_window, text="Input Volume (%):", fg=self.current_colors["fg"], 
                 bg=self.current_colors["bg"]).pack(pady=5)
        tk.Scale(audio_window, from_=0, to=100, variable=self.input_volume, orient=tk.HORIZONTAL, 
                 length=400, fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(pady=5)

        tk.Label(audio_window, text="Sensitivity/Squelch (0–100):", fg=self.current_colors["fg"], 
                 bg=self.current_colors["bg"]).pack(pady=5)
        tk.Scale(audio_window, from_=0, to=100, variable=self.sensitivity, orient=tk.HORIZONTAL, 
                 length=400, resolution=1, fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(pady=5)

        tk.Label(audio_window, text="Higher values reduce sensitivity (increase squelch), lower values increase sensitivity.", 
                 fg=self.current_colors["fg"], bg=self.current_colors["bg"], wraplength=500).pack(pady=5)

        tk.Button(audio_window, text="Save", command=lambda: [self.save_audio_settings(), audio_window.destroy()], 
                  fg=self.current_colors["fg"], bg=self.current_colors["button_bg"]).pack(pady=10)

    def update_audio_devices(self, *args):
        if self.p and self.running:
            try:
                input_idx = self.input_device_index.get() if self.input_device_index.get() >= 0 else None
                output_idx = self.output_device_index.get() if self.output_device_index.get() >= 0 else None
                
                if self.stream_in:
                    self.stream_in.stop_stream()
                    self.stream_in.close()
                if self.stream_out:
                    self.stream_out.stop_stream()
                    self.stream_out.close()

                self.stream_out = self.p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, output=True, 
                                            output_device_index=output_idx)
                self.stream_in = self.p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True, 
                                           frames_per_buffer=CHUNK, input_device_index=input_idx)
                if self.rx_output:
                    self.rx_output.insert(tk.END, f"Audio devices updated to Input: {input_idx}, Output: {output_idx}\n")
                print(f"Audio devices updated - Input device: {input_idx}, Output device: {output_idx}")
            except pyaudio.PyError as e:
                if self.rx_output:
                    self.rx_output.insert(tk.END, f"Audio update error: {str(e)}\n")
                print(f"Audio update error: {str(e)}. Please close other audio programs and try again.")
                if self.root and self.rx_output:
                    self.root.after(0, lambda: messagebox.showwarning("Audio Error", 
                                    f"Failed to update audio devices: {str(e)}. Close other audio programs and try again."))
                if self.p:
                    self.p.terminate()
                    self.p = None
                    self.running = False
            except Exception as e:
                if self.rx_output:
                    self.rx_output.insert(tk.END, f"Audio update failed: {str(e)}\n")
                print(f"Audio update error: {e}")
                if self.root and self.rx_output:
                    self.root.after(0, lambda: messagebox.showwarning("Audio Warning", 
                                    f"Failed to update audio devices: {str(e)}. Continuing without audio."))
                if self.p:
                    self.p.terminate()
                    self.p = None
                    self.running = False

    def save_audio_settings(self):
        input_value = self.input_device.get()
        output_value = self.output_device.get()
        
        try:
            if input_value in self.input_devices.values():
                self.input_device_index.set(list(self.input_devices.keys())[list(self.input_devices.values()).index(input_value)])
            elif input_value == "Default":
                self.input_device_index.set(-1)
            else:
                self.input_device_index.set(-1)

            if output_value in self.output_devices.values():
                self.output_device_index.set(list(self.output_devices.keys())[list(self.output_devices.values()).index(output_value)])
            elif output_value == "Default":
                self.output_device_index.set(-1)
            else:
                self.output_device_index.set(-1)

            self.save_settings()
            self.update_audio_devices()
        except Exception as e:
            if self.rx_output:
                self.rx_output.insert(tk.END, f"Error saving audio settings: {str(e)}\n")
            print(f"Error saving audio settings: {e}")

    def show_radio_settings_window(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Radio Settings")
        settings_window.geometry("600x500")
        settings_window.resizable(False, False)

        def validate_hex(P):
            if len(P) > 2:
                return False
            return all(c in '0123456789ABCDEFabcdef' for c in P) or P == ""

        vcmd = (self.root.register(validate_hex), '%P')

        radio_frame = tk.LabelFrame(settings_window, text="Radio Selection", fg=self.current_colors["fg"], 
                                   bg=self.current_colors["bg"])
        radio_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(radio_frame, text="Radio:", fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(side="left", padx=5)
        radio_options = ["Icom IC-703", "Icom IC-705", "Generic"]
        tk.OptionMenu(radio_frame, self.radio, *radio_options, command=self.update_radio_fields).pack(side="left", padx=5)

        tk.Label(radio_frame, text="Address (2-digit hex):", fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(side="left", padx=5)
        self.usb_entry = tk.Entry(radio_frame, textvariable=self.usb_address, width=5, 
                                 fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], 
                                 insertbackground=self.current_colors["fg"], validate="key", validatecommand=vcmd)
        self.usb_entry.pack(side="left", padx=5)

        tk.Label(radio_frame, text="Mode:", fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(side="left", padx=5)
        self.mode_frame = tk.Frame(radio_frame, bg=self.current_colors["bg"])
        self.mode_frame.pack(side="left", padx=2)
        self.mode_usb_check = tk.Checkbutton(self.mode_frame, text="USB", variable=self.mode_usb, 
                                           fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                                           selectcolor=self.current_colors["entry_bg"])
        self.mode_usb_check.pack(side="left", padx=2)
        self.mode_usb_digital_check = tk.Checkbutton(self.mode_frame, text="USB Digital", variable=self.mode_usb_digital, 
                                                   fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                                                   selectcolor=self.current_colors["entry_bg"])
        self.mode_usb_digital_check.pack(side="left", padx=2)
        self.mode_fm_check = tk.Checkbutton(self.mode_frame, text="FM", variable=self.mode_fm, 
                                          fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                                          selectcolor=self.current_colors["entry_bg"])
        self.mode_fm_check.pack(side="left", padx=2)

        control_frame = tk.LabelFrame(settings_window, text="Radio Control Port", fg=self.current_colors["fg"], 
                                    bg=self.current_colors["bg"])
        control_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(control_frame, text="Serial Port:", fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(side="left", padx=5)
        com_ports = self.get_com_ports()
        tk.OptionMenu(control_frame, self.serial_port, *com_ports).pack(side="left", padx=5)

        tk.Label(control_frame, text="Baud Rate:", fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(side="left", padx=5)
        baud_rates = ["", "1200", "2400", "4800", "9600", "19200", "38400", "57600", "115200"]
        tk.OptionMenu(control_frame, self.baud_rate, *baud_rates).pack(side="left", padx=5)

        tk.Checkbutton(control_frame, text="RTS", variable=self.rts, fg=self.current_colors["fg"], 
                      bg=self.current_colors["bg"], selectcolor=self.current_colors["entry_bg"], 
                      command=self.update_rts_dtr).pack(side="left", padx=2)
        tk.Checkbutton(control_frame, text="DTR", variable=self.dtr, fg=self.current_colors["fg"], 
                      bg=self.current_colors["bg"], selectcolor=self.current_colors["entry_bg"], 
                      command=self.update_rts_dtr).pack(side="left", padx=2)

        ptt_frame = tk.LabelFrame(settings_window, text="PTT Port", fg=self.current_colors["fg"], 
                                 bg=self.current_colors["bg"])
        ptt_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(ptt_frame, text="PTT Port:", fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(side="left", padx=5)
        tk.OptionMenu(ptt_frame, self.ptt_port, *com_ports).pack(side="left", padx=5)

        tk.Checkbutton(ptt_frame, text="RTS", variable=self.ptt_rts, fg=self.current_colors["fg"], 
                      bg=self.current_colors["bg"], selectcolor=self.current_colors["entry_bg"], 
                      command=self.update_rts_dtr).pack(side="left", padx=2)
        tk.Checkbutton(ptt_frame, text="DTR", variable=self.ptt_dtr, fg=self.current_colors["fg"], 
                      bg=self.current_colors["bg"], selectcolor=self.current_colors["entry_bg"], 
                      command=self.update_rts_dtr).pack(side="left", padx=2)

        self.radio_tx_info = {
            "Icom IC-703": "TX Command: FE FE 68 E0 1C 00 FD (Hex via serial, 9600 baud, RTS/DTR optional)",
            "Icom IC-705": "TX Command: FE FE 94 E0 1C 00 FD (Hex via serial, 9600 baud, RTS/DTR optional)",
            "Generic": "TX Command: User-defined (e.g., enter '26' for some rigs, 2-digit hex required)"
        }
        tk.Label(ptt_frame, text="TX Command Info:", fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(side="left", padx=5)
        self.tx_info_label = tk.Label(ptt_frame, text=self.radio_tx_info.get(self.radio.get(), ""), 
                                     fg=self.current_colors["fg"], bg=self.current_colors["bg"], wraplength=200)
        self.tx_info_label.pack(side="left", padx=5)
        self.radio.trace("w", self.update_tx_info)

        tk.Button(settings_window, text="Save", command=lambda: [self.save_settings(), settings_window.destroy()], 
                  fg=self.current_colors["fg"], bg=self.current_colors["button_bg"]).pack(pady=5)
        tk.Button(settings_window, text="Test PTT", command=self.test_ptt_connection, 
                  fg=self.current_colors["fg"], bg=self.current_colors["button_bg"]).pack(pady=5)

    def update_radio_fields(self, *args):
        radio = self.radio.get()
        default_address = self.get_default_address(radio)
        if default_address:
            self.usb_address.set(default_address)
        else:
            self.usb_address.set("00")
            if self.rx_output:
                self.rx_output.insert(tk.END, "Warning: Generic radio selected. Set a valid 2-digit hex address (e.g., '26').\n")
        if radio == "Generic":
            self.mode_usb.set(False)
            self.mode_usb_digital.set(False)
            self.mode_fm.set(False)
            self.mode_usb_check.config(state="disabled")
            self.mode_usb_digital_check.config(state="disabled")
            self.mode_fm_check.config(state="disabled")
        else:
            self.mode_usb_check.config(state="normal")
            self.mode_usb_digital_check.config(state="normal")
            self.mode_fm_check.config(state="normal")

    def update_rts_dtr(self):
        print(f"RTS: {self.rts.get()}, DTR: {self.dtr.get()}")
        print(f"PTT RTS: {self.ptt_rts.get()}, PTT DTR: {self.ptt_dtr.get()}")

    def update_tx_info(self, *args):
        radio = self.radio.get()
        self.tx_info_label.config(text=self.radio_tx_info.get(radio, ""))

    def communicate_with_radio(self, command):
        if self.serial_port.get() != "NONE" and self.baud_rate.get():
            try:
                radio = self.radio.get()
                address = self.usb_address.get().replace('H', '').replace('h', '')
                if not address:
                    raise ValueError("No address specified for radio communication.")
                if len(address) != 2 or not all(c in '0123456789ABCDEFabcdef' for c in address):
                    raise ValueError(f"Invalid address '{address}'. Must be a 2-character hexadecimal value.")
                address = address.upper()

                if not self.radio_serial or not self.radio_serial.is_open:
                    self.radio_serial = serial.Serial(
                        port=self.serial_port.get(),
                        baudrate=int(self.baud_rate.get()) if self.baud_rate.get().isdigit() else 9600,
                        timeout=0.1,
                        rtscts=self.rts.get(),
                        dsrdtr=self.dtr.get()
                    )
                    print(f"Radio serial connection established on {self.serial_port.get()} at {self.baud_rate.get()} baud")
                    if self.rx_output:
                        self.rx_output.insert(tk.END, f"Radio connection established on {self.serial_port.get()}\n")

                if command == "READ_FREQ":
                    if radio in ["Icom IC-703", "Icom IC-705"]:
                        cmd = bytes.fromhex(f"FE FE {address} E0 03 FD")
                    else:
                        cmd = bytes.fromhex(f"FE FE {address} E0 03 FD")
                    self.radio_serial.write(cmd)
                    time.sleep(0.1)
                    if self.radio_serial.in_waiting > 0:
                        response = self.radio_serial.read(self.radio_serial.in_waiting).hex()
                        print(f"Received frequency: {response}")
                        if self.rx_output:
                            self.rx_output.insert(tk.END, f"Received frequency: {response}\n")
                            self.rx_output.see(tk.END)
                else:
                    print(f"Unknown command: {command}")
                    if self.rx_output:
                        self.rx_output.insert(tk.END, f"Unknown radio command: {command}\n")
            except (serial.SerialException, ValueError) as e:
                print(f"Radio communication error: {str(e)}")
                if self.rx_output:
                    self.rx_output.insert(tk.END, f"Radio communication error: {str(e)}\n")
                if self.radio_serial:
                    self.radio_serial.close()
                    self.radio_serial = None
        else:
            print("No serial port or baud rate selected for radio communication.")
            if self.rx_output:
                self.rx_output.insert(tk.END, "No serial port or baud rate selected for radio communication.\n")

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

# Part 6