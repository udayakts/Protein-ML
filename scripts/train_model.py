import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN  

# Ensure output directory exists
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Redirect logs to a file
log_file = open(os.path.join(output_dir, "training_log.txt"), "w")
sys.stdout = log_file  

# Load dataset
df = pd.read_csv("data/swissprot_features.csv")

# Separate features and labels
X = df.drop(columns=["Label"])  
y = df["Label"]  

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# **Apply ADASYN for Improved Oversampling**
adasyn = ADASYN(sampling_strategy=0.4, random_state=42)  # Increased balance
X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train, y_train)

print(f"Class distribution after ADASYN: {np.bincount(y_train_balanced)}")

# **Train an Initial XGBoost Classifier**
xgb_model = XGBClassifier(n_estimators=300, 
                          learning_rate=0.05, 
                          max_depth=10,  
                          scale_pos_weight=8,  # Handle class imbalance
                          random_state=42,
                          n_jobs=-1)
xgb_model.fit(X_train_balanced, y_train_balanced)

# **Feature Importance Analysis**
feature_importance = xgb_model.feature_importances_
important_features = X.columns[feature_importance > 0.005]  # Keep top features

# **Reduce dataset to important features**
X_train_selected = X_train_balanced[important_features]
X_test_selected = X_test[important_features]

# **Retrain XGBoost on Selected Features**
xgb_model.fit(X_train_selected, y_train_balanced)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test_selected)

# Evaluate the XGBoost model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))

# Confusion Matrix - XGBoost
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, cmap="coolwarm", fmt="d",
            xticklabels=["Non-Enzyme", "Enzyme"], yticklabels=["Non-Enzyme", "Enzyme"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.savefig(os.path.join(output_dir, "confusion_matrix_xgb.png"))
plt.close()  

# Save Feature Importance Plot
plt.figure(figsize=(8,6))
plt.barh(important_features, feature_importance[feature_importance > 0.005])
plt.xlabel("F score")
plt.ylabel("Features")
plt.title("Feature Importance - XGBoost")
plt.savefig(os.path.join(output_dir, "feature_importance_xgb.png"))
plt.close()

# Close log file
sys.stdout.close()
