# EEG Workload Estimation
This project involves estimating workload using EEG data. The analysis is performed using various machine learning models and specific features.

## Introduction
The objective of this project is to analyze workload levels using EEG data from the Simultaneous Task EEG Workload (STEW) dataset. This project aims to develop machine learning models that can accurately classify workload levels based on EEG signals.
This dataset consists of raw EEG data from 48 subjects who participated in a multitasking workload experiment utilizing the SIMKAP multitasking test. The subjects’ brain activity at rest was also recorded before the test and is included as well. The Emotiv EPOC device, with sampling frequency of 128Hz and 14 channels was used to obtain the data, with 2.5 minutes of EEG recording for each case. Subjects were also asked to rate their perceived mental workload after each stage on a rating scale of 1 to 9 and the ratings are provided in a separate file.

## Data Preprocessing
The preprocessing steps involved cleaning the EEG data to ensure it is suitable for analysis. The following steps were taken:

1.	**Artifact Removal:** Using techniques such as Independent Component Analysis (ICA) to remove eye blinks and other artifacts from the EEG signals.
Independent Component Analysis (ICA) removes artifacts from EEG data by separating the recorded signals into statistically independent components, each representing a different source of brain activity or noise. Artifacts such as eye blinks, muscle movements or electrical noise typically exhibit distinctive and recognizable patterns. For instance, eye blink artifacts often show strong activity in the frontal electrodes, while muscle artifacts can appear as high-frequency noise localized to specific areas. By plotting these components, we can visually inspect and distinguish between true neural signals and artifacts. The ICA results for the first 3 subjects are depicted in Image1. According to these results, the 000,003,004, and 007 components were dropped for further analyses.

2.	**Filtering:** Applying band-pass filters to remove noise and retain relevant frequency bands (e.g., 0.5-50 Hz).
Filtering EEG data is essential to remove noise and artifacts that can obscure the true neural signals. EEG recordings are susceptible to various types of interference, including power line noise, muscle activity, eye movements, and other external electrical sources. These unwanted signals can contaminate the data, making it difficult to accurately analyze brain activity. By applying filters, such as high-pass, low-pass, and band-pass filters, we can selectively remove these unwanted frequencies and enhance the clarity of the neural signals. This preprocessing step improves the signal-to-noise ratio, facilitating more accurate interpretation and analysis of the EEG data for research or clinical purposes.

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


## Project Structure
**"ICA Visualization.py"** performs preprocessing and visualization of Independent Component Analysis (ICA) components for EEG data artifact removal. It begins by defining a function high_pass_filter to apply a high-pass filter to the EEG data, removing low-frequency noise. The main function, ica_comp_viz, loads EEG data from a specified file path, transposes it, and applies the high-pass filter. The EEG data is then structured into an MNE RawArray format, with channel names and sampling frequency information. A standard electrode montage is set, and ICA is applied to decompose the EEG data into independent components. Finally, the ICA components are visualized to help identify and remove artifacts from the EEG recordings. The example usage demonstrates how to call the ica_comp_viz function with a file path to an EEG data file.

**"Artifact & noise removal.py"** loads and preprocesses EEG data from text files, applying artifact removal and band-pass filtering to prepare the data for analysis. The load_and_preprocess_data function loads EEG data from a specified file path, removes artifacts using Independent Component Analysis (ICA), and applies a band-pass filter to retain frequencies between 0.5 and 50 Hz. The main script reads EEG data files from a dataset folder, extracting subject numbers and corresponding ratings from a separate ratings.txt file. For each EEG data file, the script preprocesses the data using the defined function, then stores the processed data and associated ratings in lists. The ratings are used to generate both continuous and binary labels based on the file naming convention. Finally, the processed EEG data and labels are converted to numpy arrays and saved to disk for future use.

**"Feature Extraction.py"** involves loading preprocessed EEG data and labels, and then iterating through the data to segment it based on window size = 30s and overlap = 15s. For each segment, the code computes all types of features, including Power Spectral Density (PSD) features, amplitude-related features, entropy features, fractal dimension features, and detrended fluctuation analysis. These features are calculated using functions from libraries like NumPy, SciPy, and Antropy. The extracted features are then organized into a DataFrame, which is saved as a CSV file for further analysis and model training.

**"Feature Importance.py"** performs feature selection and importance analysis using a RandomForestClassifier on EEG workload estimation data. It begins by preparing the feature set (X) and labels (y) from the DataFrame features_df. Labels are converted to binary classes where values less than 5 are set to 0, and values 5 and above are set to 1. The RandomForestClassifier is trained on this data to determine the importance of each feature. The resulting feature importances are sorted, and the top 10 most important features are selected and saved to a new CSV file. Finally, a bar plot is generated to visualize the importance of all features, with a highlight on the top 10 features. Also, the importance for each feature is shown in Image2.

**"Model Comparison.py"** performs a comprehensive analysis to compare different machine learning models for EEG workload estimation using selected features from a dataset. Initially, it loads the dataset and splits it into training and testing sets. The features are standardized, and labels are converted to binary classes and one-hot encoded for use with an Artificial Neural Network (ANN). The code implements cross-validation and hyperparameter tuning using GridSearchCV for three classifiers: Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Gradient Boosting. For each model, the best parameters are selected based on cross-validation scores. Additionally, an ANN model is built and trained using different hyperparameters. The performance of each model is evaluated and stored in a dictionary. Finally, classification reports for each model are printed, and a bar plot (Image3) is created to visualize and compare the accuracy of the models, identifying the best-performing model. The results are summarized in the Image4 table.


## Dependencies

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- antropy
- jupyter




