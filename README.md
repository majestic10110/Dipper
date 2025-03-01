(Quick disclaimer, this is a program and concept in testing, it is hoped that someone with more technical skill than me could refine and make it easier to install as well as perhaps iron out any bugs! As of DIPPERV4.1CAT Tx control of rig IC-703 implemented and generic slot for your radio details, this version is TXing audio out and dropping TX immediately, receive window needs work, but ready for testing!) 

# Dipper
Dipper HF Datamode


ABOUT
Dipper Mode is a custom, World first in bio-inspired audio-based digital communication mode designed by M0OLI for amateur radio operation over high-frequency (HF) bands via single-sideband (SSB) modulation. Inspired by the American Dipper bird’s (Cinclus cinclus) warbling song, it uses the song which has evolved to cut through literal waterfall and rapid noise and encodes text into melodic sound patterns for transmission and reception, optimized for noise resilience and versatility. This specification outlines its technical characteristics, including signal structure, encoding, error correction, and performance metrics, as implemented in a Python-based system.

​

Signal Characteristics

Frequency Range: 1000–5000 Hz, within the audible spectrum suitable for SSB audio input/output.

Bandwidth: Approximately 4000 Hz, fitting within a standard SSB channel (2.7–3 kHz), though configurable for narrower bands.

Sound Patterns:

Tones: Fixed-frequency sinusoidal waves (e.g., 2000 Hz).

Slides: Linear frequency sweeps (e.g., 2000–4000 Hz).

Trills: Modulated sinusoids (e.g., 3000 Hz with 20 Hz oscillation).

Duration: 0.1–0.2 seconds per sound element, repeated once with a 0.1-second inter-element gap for redundancy.

Sample Rate: 44.1 kHz, standard for audio hardware compatibility.

Vocabulary and Encoding

Symbol Set: 102 unique symbols, divided into:

Characters: 41 symbols (A-Z, 0-9, !, /, -, ., space), each mapped to a distinct sound pattern (e.g., "A": 2000 Hz tone, 0.2 sec).

Words: 61 terms, including:

General Communication: 20 terms (e.g., "the," "and," "you").

Ham Radio Terms: 21 terms (e.g., "CQ," "QTH," "73"), with 3 original ("CQ," "DE," "HTTPS://WWW.") and 18 new.

HTML Tags: 20 terms (e.g., "<html>," "<body>," "<div>").

Encoding Mechanism:

Text is parsed into words (if in WORD_SOUNDS) or characters (if in CHAR_SOUNDS).

Each symbol corresponds to a predefined sound pattern, transmitted as audio bursts.

Forward Error Correction (FEC)

Algorithm: Reed-Solomon RS(31,23) in Galois Field GF(2^8).

Parameters:

Data Symbols (k): 23 per block.

Total Symbols (n): 31 per block (23 data + 8 parity).

Error Correction: Up to 4 symbol errors per 31-symbol block.

Implementation:

Text is converted to symbol indices (0–101), padded to multiples of 23 with space symbols.

Each block is encoded into 31 symbols, transmitted with sound patterns.

Transmission and Reception

Transmission:

Audio bursts generated via sinusoidal wave synthesis (tones, slides, trills).

Repeated twice with 0.1-second gaps, plus 0.2-second inter-word gaps for CQ sequences.

Output via PC sound card to HF transceiver’s mic input.

Reception:

Input audio from transceiver via PC sound card.

Fast Fourier Transform (FFT) analysis detects peak frequencies within 200 Hz tolerance.

Symbols decoded into blocks of 31, corrected by RS(31,23), then mapped back to text.

Performance Metrics

Speed: Approximately 6–13 words per minute (wpm).

Example: "M6WAR DE M0OLI HELLO" (~17 chars) encodes to ~30 symbols with FEC, ~6–7 sec transmission.

Triple CQ call ("CQ CQ CQ DE M0OLI" thrice) ≈ 10–11 sec.

Range: Estimated 2000–5000 km (1200–3100 miles) with 100 W, dipole antenna, 20m band, moderate conditions (SFI ~100).

SNR Requirement: ~-5 to 0 dB, enabled by FEC, comparable to CW, less than FT8’s -24 dB.

Bandwidth Efficiency: ~0.025 W/Hz at 100 W over 4000 Hz, wider than FT8 (50 Hz) or CW (100–200 Hz).

System Requirements

Hardware: PC with sound card, HF SSB transceiver, audio interface (e.g., headphone jack to mic input).

Software: Python 3.x with libraries: pyaudio (audio I/O), numpy (signal generation), scipy (FFT), tkinter (GUI), reedsolo (FEC).

Interface: GUI titled "DIPPER V2.1 by M0OLI," featuring:

Callsign fields (My Callsign, To Callsign).

Transmit message input, Send/CQ buttons.

Received messages display (sent in red bold).

Settings: audio device selection, input volume (0–100%), dark mode toggle.

Operational Characteristics

Modes: Supports free-text messages (e.g., "Hello <html>"), predefined Ham terms (e.g., "CQ QTH"), and character spelling (e.g., "M0OLI").

Noise Resilience: Dual repetition and FEC correct burst errors and fading, suitable for HF noise conditions.

Use Case: Regional to mid-range DX QSOs, bridging casual, technical, and web-related communication with a melodic, bird-like signature.

Limitations

Bandwidth: Wider than FT8 or CW, less efficient per Hz but audible without software decoding.

Speed Trade-off: FEC reduces speed (~6–13 wpm vs. 10–20 wpm without), favoring reliability over throughput.

Symbol Limit: 102 symbols fit within GF(2^8), expandable with larger RS codes if needed.

​

Dipper is FREE and opensource, you may do what you wish with it if you find it useful, under the understanding it always remains FREE to end users. 

​

To download DIPPERV2.2_with_FEC - the python script visit https://mega.nz/file/anhDVCDC#ereDwuJu9ZbZlmevE1xlKizNQ2ZXUNbytP6UI8l_O40

​

 

​

There may be a newer version link on Github https://github.com/majestic10110/Dipper

​

(there is not as yet a consumer friendly installation .exe for this, it is hoped someone will package one up with more knowledge than me. Currently the Python script is offered. If you go to www.python.org and download the latest one and install the libraries for the program, you should be good to go.) If you have not used python before search and use GROK3 or Chat GPT you can paste the code in there and ask it how to make it work and it will tell you. 

​Version 4 Incorporates a Robust mode, the technical specifications of which are listed below:

Robust mode (speed setting "Robust" at ~20-40 WPM) is an advanced transmission mode in the script, designed for reliable communication with error correction and retransmission capabilities. Here are its key technical features:

Convolutional Coding:
Description: Each message is encoded using a simple convolutional code with a constraint length of 2 (state machine with CONV_TABLE).
Purpose: Adds redundancy to detect and correct bit errors during transmission.
Implementation: convolutional_encode doubles the bit length, mapping input bits to output pairs.


Interleaving:
Description: Bits are rearranged in 16-bit blocks using a 4x4 matrix pattern (interleave).
Purpose: Spreads burst errors across the message, improving error correction effectiveness.
Implementation: Shuffles bits to mitigate consecutive errors from noise.

Symbol Mapping:
Description: Encoded bits are grouped into 4-bit chunks, mapped to 16 unique audio symbols (A, B, C, ..., Q) from V4_SYMBOLS.
Purpose: Converts binary data into audio tones for transmission.
Implementation: Each symbol corresponds to a distinct sound pattern (tone, slide, or trill).

Preamble:
Description: A fixed sequence of 7 tones ("1357924") precedes each message.
Purpose: Signals the start of a Robust mode transmission for receiver synchronization.
Implementation: Generated and detected via generate_sound and detect_v4_preamble.

CRC-16 Checksum:
Description: A 16-bit cyclic redundancy check is appended to each message.
Purpose: Ensures data integrity by verifying the received message against corruption.
Implementation: Calculated with crcmod and split into four 4-bit symbols.
Automatic Repeat Request (ARQ):

Description: Uses ACK ("K") and NACK ("N") signals for retransmission control.
Purpose: Guarantees reliable delivery; retransmits only on explicit NACK.
Implementation: transmit_loop waits 0.2s for rx_ack, re-queues if "N", resets otherwise.
Viterbi Decoding:

Description: Received symbols are decoded using the Viterbi algorithm to recover the original bits.
Purpose: Corrects errors introduced during transmission by finding the most likely bit sequence.
Implementation: viterbi_decode processes deinterleaved bits with a trellis path.

Audio Transmission:
Description: Symbols are sent as 50ms audio tones at ~20-40 WPM, with distinct frequencies and patterns.
Purpose: Enables robust audio communication over noisy channels.
Implementation: Uses generate_sound with pyaudio for output.

Summary:
Robust mode employs a sophisticated error-correcting system with convolutional coding, interleaving, and Viterbi decoding, combined with CRC-16 for integrity and ARQ for reliability. It sends data as a single, preamble-led audio burst, making it ideal for challenging conditions, though it’s slower (~20-40 WPM) than Turbo mode due to its redundancy and retransmission features.


