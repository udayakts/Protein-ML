import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import ADASYN  # Use ADASYN instead of SMOTE
from xgboost import XGBClassifier

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

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# **Fix Imbalance with ADASYN**
adasyn = ADASYN(sampling_strategy=0.3, random_state=42)  # Generates harder examples
X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train, y_train)

print(f"Class distribution after ADASYN: {np.bincount(y_train_balanced)}")

# **Train an Improved XGBoost Classifier**
xgb_model = XGBClassifier(n_estimators=200, 
                          max_depth=10, 
                          learning_rate=0.1, 
                          scale_pos_weight=len(y_train_balanced[y_train_balanced == 0]) / len(y_train_balanced[y_train_balanced == 1]),  # Adjusts for imbalance
                          use_label_encoder=False, 
                          eval_metric='logloss', 
                          random_state=42)
xgb_model.fit(X_train_balanced, y_train_balanced)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate XGBoost Model
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

# Feature Importance Plot
plt.figure(figsize=(8,6))
feature_importance = xgb_model.get_booster().get_score(importance_type='weight')
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
features, scores = zip(*sorted_features[:10])  # Top 10 important features
plt.barh(features, scores)
plt.xlabel("F score")
plt.ylabel("Features")
plt.title("Feature Importance - XGBoost")
for i, v in enumerate(scores):
    plt.text(v + 5, i, str(v), color='black')
plt.savefig(os.path.join(output_dir, "feature_importance_xgb.png"))
plt.close()

# Close log file
sys.stdout.close()
