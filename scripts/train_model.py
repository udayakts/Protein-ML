import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier  # Replacing SVM with XGBoost
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.combine import SMOTETomek  # Better oversampling technique

# Ensure output directory exists
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Redirect logs to a file
log_file = open(os.path.join(output_dir, "training_log.txt"), "w")
sys.stdout = log_file  

# Load the extracted features dataset
df = pd.read_csv("data/swissprot_features.csv")

# Separate features and labels
X = df.drop(columns=["Label"])  
y = df["Label"]  

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# **Fix Imbalance: Apply SMOTETomek Instead of SMOTE**
smote = SMOTETomek(sampling_strategy=0.2, random_state=42)  # Less aggressive oversampling
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Class distribution after SMOTETomek: {np.bincount(y_train_balanced)}")

# **Apply Feature Scaling**
scaler = StandardScaler()
X_train_balanced = scaler.fit_transform(X_train_balanced)
X_test = scaler.transform(X_test)

# **Train an Improved Random Forest Classifier**
rf_model = RandomForestClassifier(n_estimators=100,  
                                  max_depth=15,  
                                  min_samples_leaf=10,  
                                  class_weight="balanced",  
                                  n_jobs=-1,  
                                  random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# **Train XGBoost Instead of SVM**
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train_balanced, y_train_balanced)

# Make Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate XGBoost
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))

# Close log file
sys.stdout.close()
