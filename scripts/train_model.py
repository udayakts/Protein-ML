import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier

# Ensure output directory exists
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
df = pd.read_csv("data/swissprot_features.csv")
X = df.drop(columns=["Label"])
y = df["Label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Hybrid Sampling: SMOTE + NearMiss to balance dataset
sampler = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)

# Convert labels back to integer format (No one-hot encoding)
y_train_balanced = y_train_balanced.astype(int)

# Train XGBoost (Using Standard Log Loss Instead of Focal Loss)
xgb_model = XGBClassifier(n_estimators=200, 
                          learning_rate=0.05, 
                          max_depth=7, 
                          min_child_weight=10,  
                          scale_pos_weight=20,  # Adjusted for class imbalance
                          subsample=0.8, 
                          colsample_bytree=0.8,  
                          objective="binary:logistic",  # FIX: Use standard binary log loss
                          random_state=42)

xgb_model.fit(X_train_balanced, y_train_balanced)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))

# Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, cmap="coolwarm", fmt="d",
            xticklabels=["Non-Enzyme", "Enzyme"], yticklabels=["Non-Enzyme", "Enzyme"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.savefig(os.path.join(output_dir, "confusion_matrix_xgb.png"))
plt.close()

# Feature Importance Analysis using SHAP
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_train_balanced)

# Plot SHAP summary
shap.summary_plot(shap_values, X_train_balanced, show=False)
plt.savefig(os.path.join(output_dir, "shap_feature_importance.png"))
plt.close()

# Save important features for reference
feature_importance = xgb_model.get_booster().get_score(importance_type='weight')
important_features = sorted(feature_importance, key=feature_importance.get, reverse=True)[:10]

plt.figure(figsize=(10,6))
plt.barh(important_features, [feature_importance[f] for f in important_features])
plt.xlabel("F score")
plt.ylabel("Features")
plt.title("Feature Importance - XGBoost")
plt.savefig(os.path.join(output_dir, "feature_importance_xgb.png"))
plt.close()

# Save log
with open(os.path.join(output_dir, "training_log.txt"), "w") as log_file:
    log_file.write(f"XGBoost Accuracy: {accuracy_xgb:.2f}\n")
    log_file.write("\nClassification Report:\n")
    log_file.write(classification_report(y_test, y_pred_xgb))
