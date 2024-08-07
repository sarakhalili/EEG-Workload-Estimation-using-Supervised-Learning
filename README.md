# EEG Workload Estimation
This project involves estimating workload using EEG data. The analysis is performed using various machine learning models and specific features.

## Introduction
The objective of this project is to analyze workload levels using EEG data from the Simultaneous Task EEG Workload (STEW) dataset. This project aims to develop machine learning models that can accurately classify workload levels based on EEG signals.
This dataset consists of raw EEG data from 48 subjects who participated in a multitasking workload experiment utilizing the SIMKAP multitasking test. The subjects’ brain activity at rest was also recorded before the test and is included as well. The Emotiv EPOC device, with sampling frequency of 128Hz and 14 channels was used to obtain the data, with 2.5 minutes of EEG recording for each case. Subjects were also asked to rate their perceived mental workload after each stage on a rating scale of 1 to 9 and the ratings are provided in a separate file.

## Data Preprocessing
The preprocessing steps involved cleaning the EEG data to ensure its suitability for analysis. The steps were:

1. **Artifact Removal**: Techniques like Independent Component Analysis (ICA) were used to eliminate eye blinks and other artifacts from the EEG signals. ICA separates the recorded signals into statistically independent components, allowing us to identify and remove artifacts such as eye blinks and muscle movements. For the first three subjects, components 000, 003, 004, and 007 were discarded based on ICA results.

![ICA Visualization](images/ICA%20Viz.png)

2. **Filtering**: Band-pass filters were applied to remove noise and retain relevant frequency bands (e.g., 0.5-50 Hz). Filtering enhances the clarity of the neural signals by eliminating unwanted frequencies, improving the signal-to-noise ratio for more accurate analysis.

## Feature Extraction
Feature extraction is crucial for translating raw EEG data into meaningful inputs for machine learning models. The following features were extracted:

### Power Spectral Density (PSD) and Related Features
- **mean_PSD**: Average power of the EEG signal across different frequency bands.
- **STD_PSD**: Variability of the power spectral density across frequency bands.
- **mean_PSD_delta, mean_PSD_theta, mean_PSD_alpha, mean_PSD_beta, mean_PSD_gamma**: Mean PSD within delta, theta, alpha, beta, and gamma frequency bands.
- **STD_PSD_delta, STD_PSD_theta, STD_PSD_alpha, STD_PSD_beta, STD_PSD_gamma**: Standard deviation of PSD within respective frequency bands.

### Amplitude-Related Features
- **A_mean**: Average amplitude of the EEG signal over time.
- **A_STD**: Variability in the amplitude of the EEG signal.
- **A_Var**: Dispersion of amplitude values around the mean.
- **A_range**: Difference between the maximum and minimum amplitude values.
- **A_skew**: Asymmetry of the amplitude distribution.
- **A_kurtosis**: "Tailedness" of the amplitude distribution.

### Entropy and Fractal Dimension Features
- **Permutation_E**: Complexity of the EEG signal.
- **Spectral_E**: Disorder of the power spectrum.
- **SVD_E**: Complexity and redundancy of the signal.
- **Approximate_E**: Regularity and complexity of the EEG signal.
- **Sample_E**: Complexity and irregularity of the signal.
- **Petrosian_FD**: Complexity of the waveform by measuring fractal characteristics.
- **Katz_FD**: Spatial complexity of the waveform.
- **Higuchi_FD**: Geometric complexity of the EEG signal.
- **Detrended Fluctuation Analysis (DFA)**: Long-range temporal correlations within the EEG signal.

## Feature Importance
To select the 10 best features, we can use a feature importance algorithm such as the Random Forest classifier's feature importance attribute. Here's how to integrate it into the existing code:
1.	Extract the features from the preprocessed data.
2.	Train a Random Forest classifier on the features.
3.	Determine the importance of each feature.
4.	Select the 10 most important features.

![Feature importance](images/Feature%20Importance.png)

## Model Development
Five machine learning models were selected for this project: Support Vector Machine (SVM), Random Forest, K-Nearest Neighbors (KNN), Gradient Boosting, and a Convolutional Neural Network (CNN). These models were chosen due to their effectiveness in classification tasks.
1.	**Support Vector Machine (SVM):**
o	Kernel: Radial Basis Function (RBF) kernel was used for its ability to handle non-linear relationships in the data.
o	Parameters: Hyperparameters such as C (regularization parameter) and gamma were optimized using grid search and cross-validation.
2.	**K-Nearest Neighbors (KNN):**
o	Neighbors: The number of neighbors (k) was optimized using grid search.
o	Distance Metric: Euclidean distance was used for calculating the distance between points.
3.	**Gradient Boosting:**
o	Learning Rate: The learning rate was optimized to balance between model complexity and performance.
o	Number of Estimators: The number of boosting stages was set to 100.
4.	**Artificial Neural Network (ANN):**
o	Architecture: A simple ANN architecture with fully connected layers was used.
o	Parameters: The number of filters, kernel size, and number of neurons in fully connected layers were optimized.

![Model comparison_table](images/Model%20Comparison_table.png)

![Model comparison](images/Model%20Comparison.png)

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




