import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import antropy as an

# Define frequency bands
frequency_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 50)
}

# Function to compute Power Spectral Density (PSD) and related features
def compute_psd_features(data, sfreq):
    freqs, psd = welch(data, sfreq)
    mean_psd = np.mean(psd, axis=1)
    std_psd = np.std(psd, axis=1)

    band_features = {}
    for band, (low_freq, high_freq) in frequency_bands.items():
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        band_psd = psd[:, band_mask]
        band_features[f'mean_PSD_{band}'] = np.mean(band_psd, axis=1)
        band_features[f'STD_PSD_{band}'] = np.std(band_psd, axis=1)

    return mean_psd, std_psd, band_features

# Function to compute amplitude-related features
def compute_amplitude_features(data):
    mean_amp = np.mean(data, axis=1)
    std_amp = np.std(data, axis=1)
    var_amp = np.var(data, axis=1)
    range_amp = np.ptp(data, axis=1)
    skew_amp = skew(data, axis=1)
    kurt_amp = kurtosis(data, axis=1)
    return mean_amp, std_amp, var_amp, range_amp, skew_amp, kurt_amp

# Function to compute entropy features
def compute_entropy_features(data):
    perm_entropy = np.array([an.perm_entropy(d, order=3, normalize=True) for d in data])
    spectral_entropy = np.array([an.spectral_entropy(d, sf=128, method='welch', normalize=True) for d in data])
    svd_entropy = np.array([an.svd_entropy(d, order=3, delay=1, normalize=True) for d in data])
    approx_entropy = np.array([an.app_entropy(d, order=2, metric='chebyshev') for d in data])
    sample_entropy = np.array([an.sample_entropy(d, order=2, metric='chebyshev') for d in data])
    return perm_entropy, spectral_entropy, svd_entropy, approx_entropy, sample_entropy    

# Function to compute fractal dimension features
def compute_fractal_features(data):
    petrosian_fd = np.array([an.petrosian_fd(d) for d in data])
    katz_fd = np.array([an.katz_fd(d) for d in data])
    higuchi_fd = np.array([an.higuchi_fd(d, kmax=10) for d in data])
    return petrosian_fd, katz_fd, higuchi_fd

# Function to compute detrended fluctuation analysis
def compute_dfa(data):
    dfa = np.array([an.detrended_fluctuation(d) for d in data])
    return dfa

# Load numpy arrays
all_data = np.load('/content/drive/MyDrive/EEG Analysis/preprocessed_eeg_data.npy')
labels = np.load('/content/drive/MyDrive/EEG Analysis/eeg_labels.npy')

# Initialize DataFrame to store features
features_list = []

# Parameters
sfreq = 128  # Sampling frequency
window_size = 30 * sfreq  # Window size for 30 seconds
overlap = 0.5  # 50% overlap
num_channels = all_data.shape[1]

for i, data in enumerate(all_data):
    num_segments = int ( data.shape[1] // (window_size * overlap) ) -1
    print("Number of iteration:" , i)
    for seg in range(num_segments):
        start = int (seg * window_size * overlap)
        end = int (start + window_size)
        segment_data = data[:, start:end]
        
        # Compute PSD features
        mean_psd, std_psd, band_features = compute_psd_features(segment_data, sfreq=128)

        # Compute amplitude features
        mean_amp, std_amp, var_amp, range_amp, skew_amp, kurt_amp = compute_amplitude_features(segment_data)

        # Compute entropy features
        perm_entropy, spectral_entropy, svd_entropy, approx_entropy, sample_entropy = compute_entropy_features(segment_data)

        # Compute fractal dimension features
        petrosian_fd, katz_fd, higuchi_fd = compute_fractal_features(segment_data)

        # Compute DFA
        dfa = compute_dfa(segment_data)

        # Append features to the list
        feature_row = {
            "mean_PSD": mean_psd.mean(),
            "STD_PSD": std_psd.mean(),
            "A_mean": mean_amp.mean(),
            "A_STD": std_amp.mean(),
            "A_Var": var_amp.mean(),
            "A_range": range_amp.mean(),
            "A_skew": skew_amp.mean(),
            "A_kurtosis": kurt_amp.mean(),
            "Permutation_E": perm_entropy.mean(),
            "Spectral_E": spectral_entropy.mean(),
            "Sample_E": sample_entropy.mean(),
            "SVD_E": svd_entropy.mean(),
            "Approximate_E": approx_entropy.mean(),
            "Petrosian_FD": petrosian_fd.mean(),
            "Katz_FD": katz_fd.mean(),
            "Higuchi_FD": higuchi_fd.mean(),
            "Detrended_fluctuation_analysis": dfa.mean(),
        }

        # Add band-specific PSD features
        for feature, values in band_features.items():
            feature_row[f"{feature}"] = values.mean()

        feature_row ["Label"] = labels[i]

        # Append to the list
        features_list.append(feature_row)

# Convert list to DataFrame
features_df = pd.DataFrame(features_list)

# Save the features to a CSV file
features_df.to_csv('/content/drive/MyDrive/EEG Analysis/eeg_features.csv', index=False)

print("Feature extraction and saving completed.")
