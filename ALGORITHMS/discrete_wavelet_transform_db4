# """
# Radar-Based Respiratory Rate Estimation using Wavelet Transform and FFT

# This script processes radar IQ data to estimate breathing rate by:
# 1. Extracting phase information from complex IQ data
# 2. Removing baseline drift using linear detrending
# 3. Performing 4-level wavelet decomposition using Daubechies db4 wavelet
# 4. Reconstructing the respiratory signal from approximation coefficients
# 5. Estimating breathing rate via FFT peak detection in respiratory band

# Reference Theory:
# - Wavelet decomposition separates signal into approximation (low-freq) and detail (high-freq) components
# - Daubechies wavelets are compactly supported orthogonal wavelets ideal for transient signal analysis
# - Respiratory signals typically fall in 0.1-0.6 Hz (6-36 BPM) range
# """

# import numpy as np
# import h5py
# import matplotlib.pyplot as plt
# import pywt
# from scipy.signal import detrend

# # ======================================================================
# # Data Loading and Preprocessing
# # ======================================================================

# # File path to radar data (HDF5 format)
# file_path = r"C:\Users\GOPAL\guptaradardata\B2.h5"

# # Load data from HDF5 file
# with h5py.File(file_path, "r") as f:
#     frame = f["sessions/session_0/group_0/entry_0/result/frame"]
#     real_part = np.array(frame["real"], dtype=np.float64)
#     imag_part = np.array(frame["imag"], dtype=np.float64)

# # Combine real and imaginary parts into complex IQ data
# # IQ_data shape: (antennas, range bins, sweeps)
# IQ_data = real_part + 1j * imag_part
# IQ_data = IQ_data.transpose(2, 1, 0)  

# # System parameters
# fs = 14.7  # Sampling frequency in Hz

# # ======================================================================
# # Phase Signal Extraction
# # ======================================================================
# """
# Phase extraction steps:
# 1. Compute magnitude of IQ data
# 2. Find range bin with maximum average magnitude (peak reflection point)
# 3. Focus analysis on ±5 bins around peak range
# 4. Compute phase difference between consecutive sweeps using conjugate multiplication
# """

# # Extract magnitude and find peak range bin
# magnitude_data = np.abs(IQ_data)
# mean_magnitude = np.mean(magnitude_data, axis=2)
# peak_range_index = np.argmax(mean_magnitude, axis=1)

# # Select range bins around peak reflection (5 bins on either side)
# range_indices = np.arange(max(0, peak_range_index[0] - 5), 
#                          min(IQ_data.shape[1], peak_range_index[0] + 5) + 1)
# filtered_data = IQ_data[:, range_indices, :]

# # Phase extraction parameters
# tau_iq = 0.5  # Time constant for low-pass filter
# alpha_iq = np.exp(-2 / (tau_iq * fs))  # Filter coefficient

# # Initialize phase signal array
# phase_signal = np.zeros(filtered_data.shape[2])

# # Compute phase difference between consecutive sweeps
# for s in range(1, filtered_data.shape[2]):
#     # Complex conjugate multiplication for phase difference
#     z = np.sum(filtered_data[:, :, s] * np.conj(filtered_data[:, :, s - 1]))
#     phase_signal[s] = np.angle(z)  # Extract phase component

# # ======================================================================
# # Signal Preprocessing
# # ======================================================================
# """
# Baseline drift removal:
# - Linear detrending removes slow-varying baseline wander
# - Essential for accurate respiratory signal extraction
# """
# phase_signal_corrected = detrend(phase_signal, type='linear')

# # ======================================================================
# # Wavelet Decomposition and Reconstruction
# # ======================================================================
# """
# 4-level wavelet decomposition using Daubechies db4 wavelet:
# - cA4: Level 4 approximation coefficients (lowest frequency band)
# - cD4-cD1: Detail coefficients at levels 4-1 (increasing frequency bands)
# - Respiratory signal is reconstructed from approximation coefficients
# """
# wavelet = 'db4'  # Daubechies 4-tap wavelet
# coeffs = pywt.wavedec(phase_signal_corrected, wavelet, level=4)

# # Unpack coefficients (A4, D4, D3, D2, D1)
# cA4, cD4, cD3, cD2, cD1 = coeffs

# # Reconstruct respiratory signal using only approximation coefficients
# resp_signal = pywt.waverec([cA4, None, None, None, None], wavelet)

# # ======================================================================
# # Respiratory Rate Estimation
# # ======================================================================
# """
# FFT-based frequency analysis:
# 1. Compute FFT of reconstructed respiratory signal
# 2. Identify peak frequency in respiratory band (0.04-0.6 Hz)
# 3. Convert to breaths per minute (BPM)
# """
# fft_values = np.fft.fft(resp_signal)
# fft_freqs = np.fft.fftfreq(len(resp_signal), d=1/fs)

# # Focus on positive frequencies only
# positive_freqs = fft_freqs[fft_freqs >= 0]
# positive_magnitude = np.abs(fft_values[fft_freqs >= 0])

# # Respiratory frequency range (0.04-0.6 Hz ≈ 2.4-36 BPM)
# respiratory_mask = (positive_freqs >= 0.04) & (positive_freqs <= 0.6)
# respiratory_freqs = positive_freqs[respiratory_mask]
# respiratory_magnitude = positive_magnitude[respiratory_mask]

# # Find dominant respiratory frequency
# if len(respiratory_magnitude) > 0:
#     dominant_freq = respiratory_freqs[np.argmax(respiratory_magnitude)]
#     breathing_rate_bpm = dominant_freq * 60  # Convert Hz to BPM
#     print(f"Estimated Average Respiratory Rate: {breathing_rate_bpm:.2f} BPM")
# else:
#     print("No dominant frequency detected in the respiratory range.")

# # ======================================================================
# # Visualization
# # ======================================================================
# # ======================================================================
# # Visualization
# # ======================================================================
# # ======================================================================
# # Visualization
# # ======================================================================
# plt.rcParams["font.family"] = "Times New Roman"  # Set global font

# plt.figure(figsize=(12, 10))

# # # Plot 1: Original Phase Signal
# # plt.subplot(5, 1, 1)
# # plt.plot(phase_signal, label="Original Phase Signal")
# # plt.title("Original Phase Signal from IQ Data")
# # plt.xlabel("Sweep Index")
# # plt.ylabel("Phase [radians]")
# # plt.legend()
# # # plt.grid()

# # Plot 2: Detrended Phase Signal
# plt.subplot(5, 1, 2)
# plt.plot(phase_signal_corrected, label="Detrended Phase Signal", color="orange")
# plt.title("Detrended Phase Signal (Baseline Removed)")
# plt.xlabel("Sweep Index")
# plt.ylabel("Phase [radians]")
# plt.legend()
# # plt.grid()

# # Plot 3: Wavelet Approximation Coefficients
# plt.subplot(5, 1, 3)
# plt.plot(cA4, label="Level 4 Approximation Coefficients", color="green")
# plt.title("Wavelet Approximation Coefficients (Level 4, db4)")
# plt.xlabel("Coefficient Index")
# plt.ylabel("Amplitude")
# plt.legend()
# # plt.grid()

# # Plot 4: Reconstructed Respiratory Signal
# plt.subplot(5, 1, 4)
# plt.plot(resp_signal, label="Reconstructed Respiratory Signal", color="blue")
# plt.title("Respiratory Signal Reconstructed from Wavelet Approximation")
# plt.xlabel("Sweep Index")
# plt.ylabel("Amplitude")
# plt.legend()
# # plt.grid()

# # Plot 5: Frequency Spectrum of Respiratory Signal
# plt.subplot(5, 1, 5)
# plt.plot(positive_freqs, positive_magnitude, label="FFT Spectrum", color="purple")
# if len(respiratory_magnitude) > 0:
#     plt.axvline(x=dominant_freq, color="red", linestyle="--", linewidth=1.5)
#     plt.text(dominant_freq + 0.02, max(positive_magnitude) * 0.7,
#              f"{dominant_freq:.2f} Hz\n({breathing_rate_bpm:.1f} BPM)",
#              color="red", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

# plt.title("FFT Spectrum of Reconstructed Respiratory Signal")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Magnitude")
# plt.legend()
# # plt.grid()

# plt.tight_layout()
# plt.show()

import numpy as np
import h5py
import matplotlib.pyplot as plt
import pywt
from scipy.signal import detrend

# ======================================================================
# Data Loading and Preprocessing
# ======================================================================
file_path = r"C:\Users\GOPAL\guptaradardata\A1.h5"
with h5py.File(file_path, "r") as f:
    frame = f["sessions/session_0/group_0/entry_0/result/frame"]
    real_part = np.array(frame["real"], dtype=np.float64)
    imag_part = np.array(frame["imag"], dtype=np.float64)

IQ_data = real_part + 1j * imag_part
IQ_data = IQ_data.transpose(2, 1, 0)  
fs = 14.7  # Sampling frequency

# ======================================================================
# Phase Extraction
# ======================================================================
magnitude_data = np.abs(IQ_data)
mean_magnitude = np.mean(magnitude_data, axis=2)
peak_range_index = np.argmax(mean_magnitude, axis=1)

range_indices = np.arange(max(0, peak_range_index[0] - 5), 
                         min(IQ_data.shape[1], peak_range_index[0] + 5) + 1)
filtered_data = IQ_data[:, range_indices, :]

tau_iq = 0.5
alpha_iq = np.exp(-2 / (tau_iq * fs))

phase_signal = np.zeros(filtered_data.shape[2])
for s in range(1, filtered_data.shape[2]):
    z = np.sum(filtered_data[:, :, s] * np.conj(filtered_data[:, :, s - 1]))
    phase_signal[s] = np.angle(z)

# ======================================================================
# Baseline Drift Removal
# ======================================================================
phase_signal_corrected = detrend(phase_signal, type='linear')

# Moving Average Filter (Window size = 10)
window_size = 10
moving_avg_filtered = np.convolve(phase_signal_corrected, 
                                  np.ones(window_size)/window_size, mode='same')

# ======================================================================
# Wavelet Decomposition and Reconstruction
# ======================================================================
wavelet = 'db4'
coeffs = pywt.wavedec(phase_signal_corrected, wavelet, level=4)
cA4, cD4, cD3, cD2, cD1 = coeffs
resp_signal = pywt.waverec([cA4, None, None, None, None], wavelet)

# ======================================================================
# Respiratory Rate Estimation
# ======================================================================
fft_values = np.fft.fft(resp_signal)
fft_freqs = np.fft.fftfreq(len(resp_signal), d=1/fs)

positive_freqs = fft_freqs[fft_freqs >= 0]
positive_magnitude = np.abs(fft_values[fft_freqs >= 0])

respiratory_mask = (positive_freqs >= 0.04) & (positive_freqs <= 0.6)
respiratory_freqs = positive_freqs[respiratory_mask]
respiratory_magnitude = positive_magnitude[respiratory_mask]

if len(respiratory_magnitude) > 0:
    dominant_freq = respiratory_freqs[np.argmax(respiratory_magnitude)]
    breathing_rate_bpm = dominant_freq * 60
    print(f"Estimated Average Respiratory Rate: {breathing_rate_bpm:.2f} BPM")
else:
    print("No dominant frequency detected in the respiratory range.")

# ======================================================================
# Visualization
# ======================================================================
plt.rcParams["font.family"] = "Times New Roman"

plt.figure(figsize=(12, 12))

# Plot 1: Detrended Phase Signal
plt.subplot(5, 1, 1)
plt.plot(phase_signal_corrected, label="Detrended Phase Signal", color="orange")
plt.title("Detrended Phase Signal (Baseline Removed)")
plt.xlabel("Sweep Index")
plt.ylabel("Phase [radians]")
plt.legend()

# Plot 2: Moving Average Filtered Signal
plt.subplot(5, 1, 2)
plt.plot(moving_avg_filtered, label="Moving Average Filtered", color="brown")
plt.title(f"Moving Average Filtered Signal (Window Size = {window_size})")
plt.xlabel("Sweep Index")
plt.ylabel("Phase [radians]")
plt.legend()

# Plot 3: Wavelet Approximation Coefficients
plt.subplot(5, 1, 3)
plt.plot(cA4, label="Level 4 Approximation Coefficients", color="green")
plt.title("Wavelet Approximation Coefficients (Level 4, db4)")
plt.xlabel("Coefficient Index")
plt.ylabel("Amplitude")
plt.legend()

# Plot 4: Reconstructed Respiratory Signal
plt.subplot(5, 1, 4)
plt.plot(resp_signal, label="Reconstructed Respiratory Signal", color="blue")
plt.title("Respiratory Signal Reconstructed from Wavelet Approximation")
plt.xlabel("Sweep Index")
plt.ylabel("Amplitude")
plt.legend()

# Plot 5: Frequency Spectrum of Respiratory Signal
plt.subplot(5, 1, 5)
plt.plot(positive_freqs, positive_magnitude, label="FFT Spectrum", color="purple")
if len(respiratory_magnitude) > 0:
    plt.axvline(x=dominant_freq, color="red", linestyle="--", linewidth=1.5)
    plt.text(dominant_freq + 0.02, max(positive_magnitude) * 0.7,
             f"{dominant_freq:.2f} Hz\n({breathing_rate_bpm:.1f} BPM)",
             color="red", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.title("FFT Spectrum of Reconstructed Respiratory Signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.legend()

plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.9)
plt.show()
