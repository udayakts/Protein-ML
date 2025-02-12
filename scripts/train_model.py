import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Ensure output directory exists
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Load the extracted features dataset
df = pd.read_csv("data/swissprot_features.csv")

# Separate features and labels
X = df.drop(columns=["Label"])
y = df["Label"]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# **Fix Imbalance: Apply SMOTE with a lower ratio**
smote = SMOTE(sampling_strategy=0.1, random_state=42)  # Prevents over-representation of the minority class
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# **Train an Optimized XGBoost Classifier**
xgb_model = XGBClassifier(n_estimators=100, 
                          learning_rate=0.1, 
                          max_depth=10, 
                          reg_lambda=10, 
                          reg_alpha=5,
                          scale_pos_weight=20,  # Adjusts for class imbalance
                          random_state=42)

xgb_model.fit(X_train_balanced, y_train_balanced)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

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

# **Feature Importance using XGBoost**
feature_importance = xgb_model.get_booster().get_score(importance_type='weight')
important_features = sorted(feature_importance, key=feature_importance.get, reverse=True)[:10]  # Top 10 features

plt.figure(figsize=(10,6))
plt.barh(important_features[::-1], [feature_importance[f] for f in important_features[::-1]], color=sns.color_palette("tab10"))
plt.xlabel("F score")
plt.ylabel("Features")
plt.title("Feature Importance - XGBoost")
plt.savefig(os.path.join(output_dir, "feature_importance_xgb.png"))
plt.close()

# **SHAP Feature Importance Analysis**
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)

plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig(os.path.join(output_dir, "shap_feature_importance.png"))
plt.close()

print("Model training and evaluation completed. Check the outputs directory for results.")
