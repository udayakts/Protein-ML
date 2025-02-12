import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm  # Import colormap utilities
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  

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

# **Fix Imbalance: Apply Moderate SMOTE**
smote = SMOTE(sampling_strategy=0.3, random_state=42)  # Oversample to 30%
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Class distribution after SMOTE: {np.bincount(y_train_balanced)}")

# **Train an Optimized XGBoost Model**
xgb_model = XGBClassifier(n_estimators=200, 
                          max_depth=10, 
                          learning_rate=0.1,
                          scale_pos_weight=len(y_train_balanced[y_train_balanced == 0]) / len(y_train_balanced[y_train_balanced == 1]),
                          use_label_encoder=False,
                          eval_metric="logloss",
                          n_jobs=-1,  
                          random_state=42)
xgb_model.fit(X_train_balanced, y_train_balanced)

# Make Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the XGBoost model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))

# Confusion Matrix - XGBoost
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, cmap="coolwarm", fmt="d",
            xticklabels=["Non-Enzyme", "Enzyme"], yticklabels=["Non-Enzyme", "Enzyme"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.savefig(os.path.join(output_dir, "confusion_matrix_xgb.png"))
plt.close()  

# **Feature Importance Analysis**
feature_importance = xgb_model.get_booster().get_score(importance_type="weight")
important_features = sorted(feature_importance, key=feature_importance.get, reverse=True)[:10]

# Reduce dataset to important features
X_train_selected = X_train_balanced[important_features]
X_test_selected = X_test[important_features]

# Plot Feature Importance
plt.figure(figsize=(10, 6))
colors = cm.get_cmap('tab10', len(important_features)).colors  # Corrected color mapping
plt.barh(important_features[::-1], [feature_importance[f] for f in important_features[::-1]], color=colors)
for i, v in enumerate([feature_importance[f] for f in important_features[::-1]]):
    plt.text(v + 50, i, f"{v:.1f}", va="center")  # Annotate bars
plt.xlabel("F score")
plt.ylabel("Features")
plt.title("Feature Importance - XGBoost")
plt.savefig(os.path.join(output_dir, "feature_importance_xgb.png"))
plt.close()

# Close log file
sys.stdout.close()
