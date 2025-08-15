import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import find_peaks
import numpy as np
import h5py
import matplotlib.pyplot as plt
from vmdpy import VMD  # Import VMD function

# File path
file_path = r"C:\Users\GOPAL\guptaradardata\A1.h5"

# Load data
with h5py.File(file_path, "r") as f:
    frame = f["sessions/session_0/group_0/entry_0/result/frame"]
    real_part = np.array(frame["real"], dtype=np.float64)  # Extract real part
    imag_part = np.array(frame["imag"], dtype=np.float64)  # Extract imaginary part

# Combine real and imaginary parts into complex IQ data
IQ_data = real_part + 1j * imag_part  # Shape: (1794, 32, 40)

# Transpose data to match MATLAB's order: (antennas x range bins x sweeps)
IQ_data = IQ_data.transpose(2, 1, 0)  # Shape: (40, 32, 1794)

# Parameters
fs = 100  # Sweep rate (Hz)
range_spacing = 0.5e-3  # Range spacing (m)
D = 100  # Downsampling factor
tau_iq = 0.04  # Time constant for low-pass filter (seconds)
f_low = 0.2 # High-pass filter cutoff frequency (Hz)

# Compute magnitude of IQ data
magnitude_data = np.abs(IQ_data)

# Find the range bin with the highest peak magnitude
mean_magnitude = np.mean(magnitude_data, axis=2)
peak_range_index = np.argmax(mean_magnitude, axis=1)

# Select the range indices around the peak
range_start_bin = max(0, peak_range_index[0] - 5)
range_end_bin = min(IQ_data.shape[1], peak_range_index[0] + 5)
range_indices = np.arange(range_start_bin, range_end_bin + 1)

# Downsampling
downsampled_data = IQ_data[:, range_indices[::D], :]  # Shape: (40, downsampled ranges, 1794)

# Low-pass filter parameters
alpha_iq = np.exp(-2 / (tau_iq * fs))

# Initialize filtered data
filtered_data = np.zeros_like(downsampled_data)
filtered_data[:, :, 0] = downsampled_data[:, :, 0]

# Apply temporal low-pass filter
for s in range(1, downsampled_data.shape[2]):
    filtered_data[:, :, s] = alpha_iq * filtered_data[:, :, s - 1] + \
                             (1 - alpha_iq) * downsampled_data[:, :, s]

# High-pass filter coefficient
alpha_phi = np.exp(-2 * f_low / fs)

# Compute phase
phi = np.zeros(filtered_data.shape[2])  # Phase for each sweep
for s in range(1, filtered_data.shape[2]):
    z = np.sum(filtered_data[:, :, s] * np.conj(filtered_data[:, :, s - 1]))
    phi[s] = alpha_phi * phi[s - 1] + np.angle(z)
plt.rcParams["font.family"] = "Times New Roman"  # Set global font

# # Plot original phase signal
# plt.figure(figsize=(12, 6))
# plt.plot(range(len(phi)), phi, linewidth=1.5)
# plt.xticks(np.arange(0, len(phi), step=100))
# plt.xlabel('Frame Index (sweeps)')
# plt.ylabel('Phase (radians)')
# plt.title('Original Phase vs. Frames')
# plt.grid(True)
# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import find_peaks

# Load your demodulated phase signal (phi)
# Assuming `phi` is already loaded from your code
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import find_peaks

# Load your demodulated phase signal (phi)
# Assuming `phi` is already loaded from your code
phi = phi  # Shape: (1794,)
fs = 14.7  # Sampling frequency (Hz)

# 1. Preprocessing: Remove DC offset and normalize
phi = phi - np.mean(phi)
phi = phi / np.max(np.abs(phi))  # Normalize to [-1, 1] range

# 2. Wavelet Decomposition (Daubechies 7)
wavelet = 'db7'
level = 5
coeffs = pywt.wavedec(phi, wavelet, level=level)

# 3. Denoising with adaptive thresholding
threshold = 4* np.max(np.abs(coeffs[-1]))
coeffs_denoised = [coeffs[0]] + [pywt.threshold(c, threshold, 'soft') for c in coeffs[1:]]

# Reconstruction with proper length handling
phi_denoised = pywt.waverec(coeffs_denoised, wavelet)
# # Ensure same length as original
# if len(phi_denoised) > len(phi):
#     phi_denoised = phi_denoised[:len(phi)]
# elif len(phi_denoised) < len(phi):
#     phi_denoised = np.pad(phi_denoised, (0, len(phi)-len(phi_denoised)), 'constant')

# # Create time vector AFTER length adjustment
# t = np.arange(len(phi_denoised)) / fs

# # 4. Cosine Transform with enhanced peak detection
# def enhanced_cosine_transform(signal, fs, f_range=np.linspace(0.1, 0.5, 2000)):
#     T = len(signal) / fs
#     CT = np.zeros_like(f_range, dtype=np.complex128)
#     for i, f in enumerate(f_range):
#         basis = np.exp(2j * np.pi * f * np.arange(len(signal)) / fs)
#         CT[i] = np.sum(signal * basis) / T
#     return f_range, np.abs(CT)

# f_range, CT = enhanced_cosine_transform(phi_denoised, fs)

# # Improved peak finding with prominence detection
# peaks, properties = find_peaks(CT, prominence=0.2*np.max(CT))
# if len(peaks) > 0:
#     main_peak_idx = peaks[np.argmax(properties['prominences'])]
#     f_est = f_range[main_peak_idx]
# else:
#     f_est = f_range[np.argmax(CT)]

# # 5. Validation Plot
# plt.figure(figsize=(14, 8))

# # Original vs Denoised
# plt.subplot(2, 1, 1)
# plt.plot(t, phi[:len(t)], label='Original', alpha=0.5)  # Ensure same length
# plt.plot(t, phi_denoised[:len(t)], label='Denoised', linewidth=2, color='red')
# plt.title('Phase Signal Before/After Denoising')
# plt.xlabel('Time (s)'); plt.ylabel('Phase (rad)')
# plt.legend(); plt.grid(True)

# # CT Spectrum with marked peak
# plt.subplot(2, 1, 2)
# plt.plot(f_range, CT, label='CT Spectrum')
# if len(peaks) > 0:
#     plt.plot(f_est, CT[main_peak_idx], 'ro', 
#              label=f'Dominant: {f_est:.3f} Hz\n({f_est*60:.1f} breaths/min)')
# plt.title('Cosine Transform Spectrum with Peak Detection')
# plt.xlabel('Frequency (Hz)'); plt.ylabel('Magnitude')
# plt.legend(); plt.grid(True)

# plt.tight_layout()
# plt.show()

# print(f"\nFinal Results:")
# print(f"Dominant Frequency: {f_est:.4f} Hz")
# print(f"Estimated Breathing Rate: {f_est*60:.2f} breaths/min")

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

# Assuming phi_denoised is already available from previous processing
signal = phi_denoised
fs = 14.8  # Sampling frequency (Hz)

# Compute DFT
n = len(signal)
yf = fft(signal)
xf = fftfreq(n, 1/fs)[:n//2]  # Get positive frequencies only
magnitude = np.abs(yf[:n//2]) / n  # Normalized magnitude

# Find peaks in the 0.0 to 0.9 Hz range
mask = (xf >= 0.1) & (xf <= 0.9)
xf_filtered = xf[mask]
magnitude_filtered = magnitude[mask]

peaks, properties = find_peaks(magnitude_filtered, height=0.1*np.max(magnitude_filtered))
if len(peaks) > 0:
    main_peak_idx = peaks[np.argmax(properties['peak_heights'])]
    dominant_freq = xf_filtered[main_peak_idx]
    bpm = dominant_freq * 60
else:
    dominant_freq = xf_filtered[np.argmax(magnitude_filtered)]
    bpm = dominant_freq * 60

# Plot DFT
plt.figure(figsize=(12, 6))
plt.plot(xf, magnitude, label='DFT Magnitude')
plt.plot(xf_filtered, magnitude_filtered, 'r', label='0.0-0.9 Hz Range', linewidth=2)

if len(peaks) > 0:
    plt.plot(xf_filtered[peaks], magnitude_filtered[peaks], 'x', label='Detected Peaks')
    plt.plot(dominant_freq, magnitude_filtered[main_peak_idx], 'ro', 
             label=f'Dominant: {dominant_freq:.3f} Hz\n({bpm:.1f} BPM)')

plt.title('DFT Spectrum with Peak Detection')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Normalized Magnitude')
plt.xlim([0, 2])  # Show up to 2 Hz for context
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"\nDFT Analysis Results:")
print(f"Dominant Frequency (0.0-0.9 Hz): {dominant_freq:.4f} Hz")
print(f"Estimated Rate: {bpm:.2f} BPM")
print(f"Number of peaks detected: {len(peaks)}")
