import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import BorderlineSMOTE  # More controlled oversampling
import xgboost as xgb

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

# **Fix Imbalance: Use Borderline-SMOTE (safer than ADASYN)**
smote = BorderlineSMOTE(sampling_strategy=0.3, random_state=42, kind="borderline-1")
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Class distribution after Borderline-SMOTE: {np.bincount(y_train_balanced)}")

# **Train XGBoost Classifier with Class Weighting**
xgb_model = xgb.XGBClassifier(n_estimators=200,
                              max_depth=10,
                              scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # Balancing classes
                              use_label_encoder=False,
                              eval_metric="logloss",
                              random_state=42)

xgb_model.fit(X_train_balanced, y_train_balanced)

# **Make predictions with threshold tuning**
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]  # Get probability scores
threshold = 0.6  # Increase threshold to reduce false positives
y_pred_xgb = (y_pred_proba > threshold).astype(int)

# Evaluate the XGBoost model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))

# **Confusion Matrix - XGBoost**
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, cmap="coolwarm", fmt="d",
            xticklabels=["Non-Enzyme", "Enzyme"], yticklabels=["Non-Enzyme", "Enzyme"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.savefig(os.path.join(output_dir, "confusion_matrix_xgb.png"))
plt.close()  

# **Feature Importance**
feature_importance = xgb_model.get_booster().get_score(importance_type="weight")
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
important_features = [feat for feat, _ in sorted_features[:10]]  # Top 10 features

# **Plot Feature Importance**
plt.figure(figsize=(10,6))
sns.barplot(x=[feature_importance[f] for f in important_features], y=important_features)
plt.xlabel("F score")
plt.ylabel("Features")
plt.title("Feature Importance - XGBoost")
plt.savefig(os.path.join(output_dir, "feature_importance_xgb.png"))
plt.close()

# **Reduce dataset to important features**
X_train_selected = X_train_balanced[important_features]
X_test_selected = X_test[important_features]

# **Retrain XGBoost with selected features**
xgb_model.fit(X_train_selected, y_train_balanced)
y_pred_selected = (xgb_model.predict_proba(X_test_selected)[:, 1] > threshold).astype(int)

# Evaluate with selected features
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print(f"XGBoost (Selected Features) Accuracy: {accuracy_selected:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_selected))

# Close log file
sys.stdout.close()
