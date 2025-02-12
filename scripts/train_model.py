import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import ADASYN

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

# **Apply ADASYN to Handle Imbalance**
adasyn = ADASYN(sampling_strategy=0.5, random_state=42)  # Adjust sampling ratio
X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train, y_train)

print(f"Class distribution after ADASYN: {np.bincount(y_train_balanced)}")

# **Feature Selection: Keep Most Important Features**
important_features = ["Molecular_Weight", "Hydrophobicity", "W", "Y", "F", "C", "E", "D", "G", "V"]
X_train_selected = X_train_balanced[important_features]
X_test_selected = X_test[important_features]

# **Train Optimized XGBoost Model**
xgb_model = xgb.XGBClassifier(
    n_estimators=300,  # Increased for better learning
    max_depth=10,
    learning_rate=0.05,  # Reduced to prevent overfitting
    scale_pos_weight=5,  # Balancing false positives
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train_selected, y_train_balanced)

# Make Predictions
y_pred_xgb = xgb_model.predict(X_test_selected)

# Evaluate the Model
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

# **Plot Feature Importance**
importance = xgb_model.get_booster().get_score(importance_type="weight")
importance_df = pd.DataFrame(importance.items(), columns=["Feature", "F score"]).sort_values(by="F score", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x="F score", y="Feature", data=importance_df)
for i, v in enumerate(importance_df["F score"]):
    plt.text(v + 100, i, f"{v:.1f}", color="black")
plt.xlabel("F score")
plt.ylabel("Feature")
plt.title("Feature Importance - XGBoost")
plt.savefig(os.path.join(output_dir, "feature_importance_xgb.png"))
plt.close()

# Close log file
sys.stdout.close()
