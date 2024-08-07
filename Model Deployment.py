import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load the dataset with selected features
data = pd.read_csv('/content/drive/MyDrive/EEG Analysis/eeg_top_features.csv')

# Separate features and labels
X = data.drop(columns=['Label'])
y = data['Label']

# Convert labels to binary classes
y_binary = np.where(y < 5, 0, 1)

# One-hot encode the binary labels for ANN
y_binary_categorical = to_categorical(y_binary)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y_binary_categorical, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_cat_scaled = scaler.fit_transform(X_train_cat)
X_test_cat_scaled = scaler.transform(X_test_cat)

# Dictionary to store the results
results = {}

# SVM with GridSearchCV and cross-validation
svm_params = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}
svm_model = GridSearchCV(SVC(), svm_params, refit=True, verbose=0)
svm_model.fit(X_train_scaled, y_train)
svm_best_model = svm_model.best_estimator_
svm_scores = cross_val_score(svm_best_model, X_train_scaled, y_train, cv=5)
svm_y_pred = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_y_pred)
results['SVM'] = svm_scores.mean()

# KNN with GridSearchCV and cross-validation
knn_params = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}
knn_model = GridSearchCV(KNeighborsClassifier(), knn_params, refit=True, verbose=0)
knn_model.fit(X_train_scaled, y_train)
knn_best_model = knn_model.best_estimator_
knn_scores = cross_val_score(knn_best_model, X_train_scaled, y_train, cv=5)
knn_y_pred = knn_model.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_y_pred)
results['KNN'] = knn_scores.mean()

# Gradient Boosting with GridSearchCV and cross-validation
gb_params = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
gb_model = GridSearchCV(GradientBoostingClassifier(), gb_params, refit=True, verbose=0)
gb_model.fit(X_train_scaled, y_train)
gb_best_model = gb_model.best_estimator_
gb_scores = cross_val_score(gb_best_model, X_train_scaled, y_train, cv=5)
gb_y_pred = gb_model.predict(X_test_scaled)
gb_accuracy = accuracy_score(y_test, gb_y_pred)
results['Gradient Boosting'] = gb_scores.mean()

# Define function to create ANN model
def create_ann_model(optimizer='adam'):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train_cat_scaled.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(y_binary_categorical.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Grid Search for ANN
best_ann_accuracy = 0
best_ann_params = {}

# Hyperparameters to search
batch_sizes = [10, 20]
epochs_list = [50, 100]
optimizers = ['adam', 'rmsprop']

for batch_size in batch_sizes:
    for epochs in epochs_list:
        for optimizer in optimizers:
            ann_model = create_ann_model(optimizer)
            ann_model.fit(X_train_cat_scaled, y_train_cat, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_test_cat_scaled, y_test_cat))
            ann_y_pred = ann_model.predict(X_test_cat_scaled)
            ann_y_pred_classes = np.argmax(ann_y_pred, axis=1)
            ann_accuracy = accuracy_score(y_test, ann_y_pred_classes)
            if ann_accuracy > best_ann_accuracy:
                best_ann_accuracy = ann_accuracy
                best_ann_params = {'batch_size': batch_size, 'epochs': epochs, 'optimizer': optimizer}

results['ANN'] = best_ann_accuracy

# Print classification reports
print("SVM Classification Report:\n", classification_report(y_test, svm_y_pred))
print("KNN Classification Report:\n", classification_report(y_test, knn_y_pred))
print("Gradient Boosting Classification Report:\n", classification_report(y_test, gb_y_pred))
print("ANN Classification Report:\n", classification_report(y_test, ann_y_pred_classes))

# Plot the results
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.ylim(0, 1)
plt.show()

# Print the best model
best_model = max(results, key=results.get)
print(f"The best model is {best_model} with an accuracy of {results[best_model]:.2f}")
