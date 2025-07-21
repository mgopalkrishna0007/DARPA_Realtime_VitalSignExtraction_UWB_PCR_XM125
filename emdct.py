import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
from PyEMD import EEMD

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
fs = 14.7  # Sweep rate (Hz)
range_spacing = 0.5e-3  # Range spacing (m)
D = 100  # Downsampling factor
tau_iq = 0.04  # Time constant for low-pass filter (seconds)
f_low = 0.2  # High-pass filter cutoff frequency (Hz)

# Compute the magnitude of IQ data (sweeps x range bins)
magnitude_data = np.abs(IQ_data)

# Find the range bin with the highest peak magnitude (across all sweeps)
mean_magnitude = np.mean(magnitude_data, axis=2)  # Mean over sweeps
peak_range_index = np.argmax(mean_magnitude, axis=1)  # Index for each antenna

# Select the range indices based on the peak range bin
range_start_bin = max(0, peak_range_index[0] - 5)  # Adjust as needed
range_end_bin = min(IQ_data.shape[1], peak_range_index[0] + 5)
range_indices = np.arange(range_start_bin, range_end_bin + 1)

# Downsampling
downsampled_data = IQ_data[:, range_indices[::D], :]  # Shape: (40, downsampled ranges, 1794)

# Temporal low-pass filter parameters
alpha_iq = np.exp(-2 / (tau_iq * fs))  # Low-pass filter coefficient

# Initialize filtered data
filtered_data = np.zeros_like(downsampled_data)
filtered_data[:, :, 0] = downsampled_data[:, :, 0]

# Apply temporal low-pass filter
for s in range(1, downsampled_data.shape[2]):
    filtered_data[:, :, s] = alpha_iq * filtered_data[:, :, s - 1] + \
                             (1 - alpha_iq) * downsampled_data[:, :, s]

# Phase unwrapping and high-pass filtering parameters
alpha_phi = np.exp(-2 * f_low / fs)  # High-pass filter coefficient

# Initialize phase values
phi = np.zeros(filtered_data.shape[2])  # Phase for each sweep

# Calculate phase for each sweep
for s in range(1, filtered_data.shape[2]):
    z = np.sum(filtered_data[:, :, s] * np.conj(filtered_data[:, :, s - 1]))
    phi[s] = alpha_phi * phi[s - 1] + np.angle(z)

# Clutter suppression: Subtract the mean phase
mean_phi = np.mean(phi)  # Compute the mean phase
phi_clutter_suppressed = phi - mean_phi  # Subtract the mean to remove clutter

# Bandpass filter parameters
lowcut = 0.1  # Low cutoff frequency (Hz)
highcut = 0.5  # High cutoff frequency (Hz)
order = 4  # Filter order

# Design the bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Apply the bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)  # Zero-phase filtering
    return y

# Apply the bandpass filter to the clutter-suppressed phase signal
phi_filtered = bandpass_filter(phi_clutter_suppressed, lowcut, highcut, fs, order)

# Normalize the filtered phase signal
max_abs_phi = np.max(np.abs(phi_filtered))  # Maximum absolute value
phi_normalized = phi_filtered / max_abs_phi  # Normalize to [-1, 1]

# Function to apply EEMD and extract the respiration signal
def extract_respiration_eemd(signal):
    # Initialize EEMD
    eemd = EEMD()
    
    # Decompose the signal into IMFs
    imfs = eemd(signal)
    
    # Respiration is typically in the low-frequency IMFs (e.g., IMF 3-5)
    respiration_signal = np.sum(imfs[2:5], axis=0)  # Sum IMFs 3-5
    
    return respiration_signal, imfs

# Apply EEMD to the normalized phase signal
respiration_signal, imfs = extract_respiration_eemd(phi_normalized)

# Function to apply a moving average filter
def moving_average_filter(signal, window_size):
    window = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(signal, window, mode='same')
    return smoothed_signal

# Apply moving average filter to the reconstructed respiration signal
window_size = 20
respiration_signal_smoothed = moving_average_filter(respiration_signal, window_size)

# Custom peak detection function
def custom_peak_detection(signal):
    dy_dx = np.gradient(signal)
    peaks = []
    for i in range(1, len(dy_dx)):
        if dy_dx[i - 1] > 0 and dy_dx[i] < 0 and signal[i] > 0:
            peaks.append(i)
    return np.array(peaks)

# Time-Domain Analysis (Custom Peak Detection)
def estimate_breath_rate_time_domain(signal, fs):
    peaks = custom_peak_detection(signal)
    total_duration = len(signal) / fs
    breath_rate = (len(peaks) / total_duration) * 60 if total_duration > 0 else 0
    return breath_rate, peaks

# Estimate breath rate using custom peak detection
breath_rate_time_domain, peaks = estimate_breath_rate_time_domain(respiration_signal_smoothed, fs)

# Enhanced Cosine Transform Analysis
def cosine_transform_analysis(signal, fs, f_range=np.linspace(0.1, 1.0, 1000)):
    """
    Enhanced Cosine Transform with peak detection
    """
    T = len(signal) / fs
    CT = np.zeros_like(f_range, dtype=np.complex128)
    
    for i, f in enumerate(f_range):
        basis = np.exp(2j * np.pi * f * np.arange(len(signal)) / fs)
        CT[i] = np.sum(signal * basis) / T
    
    magnitude = np.abs(CT)
    
    # Find peaks in the specified range
    peaks, properties = find_peaks(magnitude, prominence=0.2*np.max(magnitude))
    
    if len(peaks) > 0:
        main_peak_idx = peaks[np.argmax(properties['prominences'])]
        dominant_freq = f_range[main_peak_idx]
        bpm = dominant_freq * 60
    else:
        dominant_freq = f_range[np.argmax(magnitude)]
        bpm = dominant_freq * 60
    
    return f_range, magnitude, peaks, dominant_freq, bpm

# Perform Cosine Transform analysis
f_range, ct_magnitude, ct_peaks, dominant_freq, bpm = cosine_transform_analysis(
    respiration_signal_smoothed, fs, f_range=np.linspace(0.1, 1.0, 1000))

# Visualization
plt.figure(figsize=(15, 10))

# Plot 1: Original vs Filtered Phase
plt.subplot(3, 1, 1)
plt.plot(phi, label='Original Phase', alpha=0.5)
plt.plot(phi_filtered, label='Bandpass Filtered', linewidth=1.5)
plt.title('Phase Signal Processing')
plt.xlabel('Samples')
plt.ylabel('Phase (rad)')
plt.legend()
plt.grid(True)

# Plot 2: Smoothed Respiration Signal with Peaks
plt.subplot(3, 1, 2)
plt.plot(respiration_signal_smoothed, label='Smoothed Respiration', linewidth=1.5)
plt.plot(peaks, respiration_signal_smoothed[peaks], 'ro', label='Detected Peaks')
plt.title(f'Time-Domain Respiration Signal (Rate: {breath_rate_time_domain:.2f} BPM)')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot 3: Cosine Transform Spectrum
plt.subplot(3, 1, 3)
plt.plot(f_range, ct_magnitude, label='Cosine Transform Spectrum')
if len(ct_peaks) > 0:
    plt.plot(f_range[ct_peaks], ct_magnitude[ct_peaks], 'x', markersize=10, label='Detected Peaks')
    plt.plot(dominant_freq, ct_magnitude[np.argmax(ct_magnitude[ct_peaks])], 'ro', 
             label=f'Dominant: {dominant_freq:.3f} Hz\n({bpm:.1f} BPM)')
plt.title('Frequency-Domain Analysis (Cosine Transform)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim([0.1, 1.0])
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print results
print("\n=== Time-Domain Analysis ===")
print(f"Number of Peaks Detected: {len(peaks)}")
print(f"Breathing Rate: {breath_rate_time_domain:.2f} BPM")

print("\n=== Frequency-Domain Analysis (Cosine Transform) ===")
print(f"Dominant Frequency: {dominant_freq:.4f} Hz (0.1-1.0 Hz range)")
print(f"Estimated Breathing Rate: {bpm:.2f} BPM")
print(f"Number of significant peaks: {len(ct_peaks)}")