import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier  # Faster alternative to SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the extracted features dataset
df = pd.read_csv("data/swissprot_features.csv")

# Separate features and labels
X = df.drop(columns=["Label"])  # Features
y = df["Label"]  # Labels (1 = enzyme, 0 = non-enzyme)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion Matrix Visualization (Fix: Prevent script from pausing)
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, cmap="coolwarm", fmt="d",
            xticklabels=["Non-Enzyme", "Enzyme"], yticklabels=["Non-Enzyme", "Enzyme"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")

# Show or Save Plot
plt.show(block=False)  # Fix: Continue script execution
plt.savefig("outputs/confusion_matrix_rf.png")  # Save figure as an image
plt.close()  # Close figure

# Reduce dataset size for SVM to make it faster (1000 samples)
X_train_svm, _, y_train_svm, _ = train_test_split(X_train, y_train, train_size=1000, random_state=42)

# Use SGDClassifier instead of SVC for faster training
svm_model = SGDClassifier(loss="hinge", random_state=42, max_iter=1000, tol=1e-3, verbose=1)  # Much faster SVM
svm_model.fit(X_train_svm, y_train_svm)

# Make Predictions
y_pred_svm = svm_model.predict(X_test)

# Evaluate the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
