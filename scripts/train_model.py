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
from sklearn.feature_selection import SelectFromModel

# Ensure output directory exists
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
df = pd.read_csv("data/swissprot_features.csv")
X = df.drop(columns=["Label"])
y = df["Label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Hybrid Sampling: SMOTE + Tomek Links
sampler = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)
y_train_balanced = y_train_balanced.astype(int)

# **Feature Selection Using XGBoost**
feature_selector = XGBClassifier(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42)
feature_selector.fit(X_train_balanced, y_train_balanced)

# Retain Only Important Features
selector = SelectFromModel(feature_selector, prefit=True, max_features=15)
X_train_selected = selector.transform(X_train_balanced)
X_test_selected = selector.transform(X_test)

# **Optimized XGBoost Model**
xgb_model = XGBClassifier(
    n_estimators=200,  
    learning_rate=0.03, 
    max_depth=5,  # Reduced depth for speed
    min_child_weight=8,  
    scale_pos_weight=15,  
    subsample=0.85,  
    colsample_bytree=0.85,  
    objective="binary:logistic",  
    n_jobs=-1,  # Uses all CPU cores
    random_state=42
)

# Train Model
xgb_model.fit(X_train_selected, y_train_balanced)

# **Predictions & Evaluation**
y_pred_xgb = xgb_model.predict(X_test_selected)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))

# **Confusion Matrix**
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, cmap="coolwarm", fmt="d",
            xticklabels=["Non-Enzyme", "Enzyme"], yticklabels=["Non-Enzyme", "Enzyme"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.savefig(os.path.join(output_dir, "confusion_matrix_xgb.png"))
plt.close()

# **SHAP Feature Importance**
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_train_selected)

shap.summary_plot(shap_values, X_train_selected, show=False)
plt.savefig(os.path.join(output_dir, "shap_feature_importance.png"))
plt.close()

# **XGBoost Feature Importance**
feature_importance = xgb_model.get_booster().get_score(importance_type='weight')
important_features = sorted(feature_importance, key=feature_importance.get, reverse=True)[:10]

plt.figure(figsize=(10,6))
plt.barh(important_features, [feature_importance[f] for f in important_features])
plt.xlabel("F score")
plt.ylabel("Features")
plt.title("Feature Importance - XGBoost")
plt.savefig(os.path.join(output_dir, "feature_importance_xgb.png"))
plt.close()

# **Save log**
with open(os.path.join(output_dir, "training_log.txt"), "w") as log_file:
    log_file.write(f"XGBoost Accuracy: {accuracy_xgb:.2f}\n")
    log_file.write("\nClassification Report:\n")
    log_file.write(classification_report(y_test, y_pred_xgb))
