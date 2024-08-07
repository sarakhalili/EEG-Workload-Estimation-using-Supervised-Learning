import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import mne
import os


# Function to load and preprocess the EEG data
def load_and_preprocess_data(file_path):
    # Load the data
    data = pd.read_csv(file_path, sep="\s+", header=None)

    # Artifact Removal using ICA
    eeg_data = data.values.T
    info = mne.create_info(ch_names=[str(i) for i in range(14)], sfreq=128, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info)
    ica = mne.preprocessing.ICA(n_components=14, random_state=97)
    ica.fit(raw)
    ica.exclude = [0, 3, 4, 7]  # Assume components 0 and 1 are artifacts

    # Apply ICA to remove the artifacts
    raw_ica = ica.apply(raw)

    # Band-pass Filtering
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=128, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    filtered_data = bandpass_filter(raw_ica.get_data())

    return filtered_data

# Path to the dataset folder
dataset_folder_path = '/content/drive/MyDrive/EEG Analysis/STEW Dataset'


# # Initialize lists to store the data and labels
all_data = []
labels = []
labels_binary=[]

# Read the ratings from the ratings.txt file
ratings_file_path = os.path.join(dataset_folder_path, 'ratings.txt')
subject_ratings = {}

with open(ratings_file_path, 'r') as ratings_file:
    for line in ratings_file:
        parts = line.strip().split(',')
        subject_number = int(parts[0])
        if len(parts) == 3:  # Ensure that there are exactly 3 parts in the line
            rating_rest = int(parts[1])
            rating_test = int(parts[2])
            subject_ratings[subject_number] = (rating_rest, rating_test)

# Process each EEG data file
for file_name in os.listdir(dataset_folder_path):
    if file_name.endswith('.txt') and 'ratings' not in file_name:
        file_path = os.path.join(dataset_folder_path, file_name)


        # Extract subject number from the file name
        subject_number = int(file_name.split('_')[0][3:5])  # Adjust according to your file naming convention

        # Check if ratings are available for the subject
        if subject_number in subject_ratings:
            preprocessed_data = load_and_preprocess_data(file_path)

            all_data.append(preprocessed_data)

            # Append the ratings to the labels lists
            rating_rest, rating_test = subject_ratings[subject_number]
            if file_name.split('_')[1][0:2]=="lo":
                labels.append(rating_rest)
            else:
                labels.append(rating_test)

            # Append the ratings to the binary labels lists
            rating_rest, rating_test = subject_ratings[subject_number]
            if file_name.split('_')[1][0:2]=="lo":
                labels_binary.append(0)
            else:
                labels_binary.append(1)

# Convert to numpy arrays
all_data = np.array(all_data)
print(np.shape(all_data))
labels = np.array(labels)
labels_binary = np.array(labels_binary)

# Save the preprocessed data
np.save('/content/drive/MyDrive/EEG Analysis/preprocessed_eeg_data.npy', all_data)
np.save('/content/drive/MyDrive/EEG Analysis/eeg_labels.npy', labels)
np.save('/content/drive/MyDrive/EEG Analysis/eeg_labels_binary.npy', labels_binary)
