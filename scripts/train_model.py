import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance  # Optimized XGBoost
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import BorderlineSMOTE  # Improved SMOTE

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

# **Fix Imbalance: Apply Borderline SMOTE (Reduced Ratio)**
smote = BorderlineSMOTE(sampling_strategy=0.15, kind="borderline-1", random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Class distribution after Borderline SMOTE: {np.bincount(y_train_balanced)}")

# **Apply Feature Scaling**
scaler = StandardScaler()
X_train_balanced = scaler.fit_transform(X_train_balanced)
X_test = scaler.transform(X_test)

# **Train an Optimized Random Forest Classifier**
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

# **Train XGBoost with Focal Loss and Class Weighting**
scale_pos_weight = len(y_train_balanced) / sum(y_train_balanced == 1)
xgb_model = XGBClassifier(use_label_encoder=False, 
                          eval_metric="logloss", 
                          scale_pos_weight=scale_pos_weight, 
                          max_depth=10, 
                          learning_rate=0.05, 
                          objective="binary:logistic", 
                          gamma=2)  # Focal Loss effect
xgb_model.fit(X_train_balanced, y_train_balanced)

# Make Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate XGBoost
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))

# **Feature Importance Analysis**
plt.figure(figsize=(10,6))
plot_importance(xgb_model, max_num_features=10)
plt.savefig(os.path.join(output_dir, "feature_importance_xgb.png"))
plt.close()

# Close log file
sys.stdout.close()
