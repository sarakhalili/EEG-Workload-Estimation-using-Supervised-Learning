import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Feature selection
X = features_df.drop(columns=['Label'])
y = features_df['Label']

# Convert labels to binary classes
y_binary = np.where(y < 5, 0, 1)


# Using RandomForestClassifier for feature importance
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_binary)

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Select top 10 features
top_features = importance_df.head(10)['Feature'].values
X_top = X[top_features]

# Save the selected features to a CSV file
X_top['Label'] = y.values
X_top.to_csv('/content/drive/MyDrive/EEG Analysis/eeg_top_features.csv', index=False)

print("Feature extraction, selection, and saving completed.")

# Plot all feature importances
plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='lightgray', label='Other features')
plt.barh(importance_df.head(10)['Feature'], importance_df.head(10)['Importance'], color='skyblue', label='Top 10 features')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.legend()
plt.gca().invert_yaxis()
plt.show()