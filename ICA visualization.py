import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import mne
import os

def high_pass_filter(data, sfreq, l_freq):
    """
    Apply a high-pass filter to the data.

    Parameters:
    data (numpy.ndarray): The EEG data.
    sfreq (float): The sampling frequency of the data.
    l_freq (float): The lower bound of the high-pass filter.

    Returns:
    numpy.ndarray: The filtered data.
    """
    # Design high-pass filter
    nyquist = 0.5 * sfreq
    normal_cutoff = l_freq / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)

    # Apply the filter
    filtered_data = lfilter(b, a, data, axis=1)
    return filtered_data

def ica_comp_viz(file_path):
    """
    Visualize ICA components for EEG data artifact removal.

    Parameters:
    file_path (str): Path to the EEG data file.
    """
    # Load the data
    data = pd.read_csv(file_path, sep="\s+", header=None)

    # Prepare EEG data for ICA
    eeg_data = data.values.T
    sfreq = 128  # Sampling frequency
    l_freq = 1.0  # High-pass filter lower bound

    # Apply high-pass filter
    eeg_data = high_pass_filter(eeg_data, sfreq, l_freq)

    # Create MNE info structure
    ch_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    # Create RawArray
    raw = mne.io.RawArray(eeg_data, info)

    # Set a standard montage (electrode locations)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')

    # Apply ICA
    ica = mne.preprocessing.ICA(n_components=14, max_iter=1000, random_state=97)
    ica.fit(raw)

    # Visualize ICA components
    ica.plot_components(inst=raw)

# Example usage
file_path = "/content/drive/MyDrive/EEG Analysis/STEW Dataset/sub01_lo.txt"
ICA_Viz = ica_comp_viz(file_path)
