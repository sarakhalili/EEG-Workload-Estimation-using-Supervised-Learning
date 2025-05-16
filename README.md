# EEG Workload Estimation
This project involves estimating workload using EEG data. The analysis is performed using various machine learning models and specific features.

## Introduction
The objective of this project is to analyze workload levels using EEG data from the Simultaneous Task EEG Workload (STEW) dataset. This project aims to develop machine learning models that can accurately classify workload levels based on EEG signals.
This dataset consists of raw EEG data from 48 subjects who participated in a multitasking workload experiment utilizing the SIMKAP multitasking test. The subjects’ brain activity at rest was also recorded before the test and is included as well. The Emotiv EPOC device, with sampling frequency of 128Hz and 14 channels was used to obtain the data, with 2.5 minutes of EEG recording for each case. Subjects were also asked to rate their perceived mental workload after each stage on a rating scale of 1 to 9 and the ratings are provided in a separate file.


## Project Structure

**"ICA Visualization"**:
- **Purpose**: Preprocessing and visualization of ICA components for EEG data artifact removal.
- **Steps**:
  - Applies a high-pass filter to remove low-frequency noise.
  - Loads and transposes EEG data, formats it into an MNE RawArray, sets electrode montage, and applies ICA.
  - Visualizes ICA components to identify and remove artifacts.
- **Usage**: Call `ica_comp_viz` with the EEG data file path.

**"Artifact & Noise Removal"**:
- **Purpose**: Load and preprocess EEG data from text files.
- **Steps**:
  - Removes artifacts using ICA and applies a band-pass filter (0.5-50 Hz).
  - Reads data files, extracts subject numbers and ratings, preprocesses data, and stores processed data and labels.
  - Converts processed data and labels to numpy arrays and saves them.

**"Feature Extraction"**:
- **Purpose**: Load preprocessed EEG data and labels, segment data, and compute features.
- **Steps**:
  - Segments data into 30s windows with 15s overlap.
  - Computes features like PSD, amplitude, entropy, fractal dimension, and detrended fluctuation analysis.
  - Organizes features into a DataFrame and saves it as a CSV file.

**"Feature Importance"**:
- **Purpose**: Perform feature selection and importance analysis using RandomForestClassifier.
- **Steps**:
  - Prepares features (X) and labels (y), converts labels to binary.
  - Trains classifier, sorts and selects top 10 features, and saves them.
  - Visualizes feature importance with a bar plot (highlighting top 10).

**"Model Comparison"**:
- **Purpose**: Compare machine learning models for EEG workload estimation.
- **Steps**:
  - Loads and splits dataset, standardizes features, and converts labels to binary and one-hot encoded.
  - Implements cross-validation and hyperparameter tuning for SVM, KNN, and Gradient Boosting.
  - Builds and trains an ANN model with various hyperparameters.
  - Evaluates model performance, prints classification reports, and visualizes accuracy with bar plots and summary tables (Image4).


## Dependencies

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- antropy
- jupyter




