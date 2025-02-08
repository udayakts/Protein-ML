import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
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
smote = SMOTE(sampling_strategy=0.3, random_state=42)  # Only oversample to 30% instead of 60%
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Class distribution after SMOTE: {np.bincount(y_train_balanced)}")

# **Train an Optimized Random Forest Classifier**
rf_model = RandomForestClassifier(n_estimators=100,  # Reduced to speed up training
                                  max_depth=15, 
                                  min_samples_leaf=5,  
                                  class_weight="balanced",  
                                  n_jobs=-1,  # Uses all CPU cores
                                  random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion Matrix - Random Forest
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, cmap="coolwarm", fmt="d",
            xticklabels=["Non-Enzyme", "Enzyme"], yticklabels=["Non-Enzyme", "Enzyme"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.savefig(os.path.join(output_dir, "confusion_matrix_rf.png"))
plt.close()  

# **Train a Faster SVM Model (LinearSVC Instead of RBF Kernel)**
svm_model = LinearSVC(class_weight="balanced", random_state=42, max_iter=5000)  # Faster than Kernel SVM
svm_model.fit(X_train_balanced, y_train_balanced)

# Make Predictions
y_pred_svm = svm_model.predict(X_test)

# Evaluate the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

# Confusion Matrix - SVM
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, cmap="coolwarm", fmt="d",
            xticklabels=["Non-Enzyme", "Enzyme"], yticklabels=["Non-Enzyme", "Enzyme"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM")
plt.savefig(os.path.join(output_dir, "confusion_matrix_svm.png"))
plt.close()  

# Close log file
sys.stdout.close()
