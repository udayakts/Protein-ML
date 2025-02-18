import os
import numpy as np
import pandas as pd
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
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

# Convert labels back to integer format
y_train_balanced = y_train_balanced.astype(int)

# Hyperparameter tuning function
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 10, 50, step=10),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
    }
    
    model = XGBClassifier(**params, tree_method="hist", objective="binary:logistic", random_state=42)
    score = cross_val_score(model, X_train_balanced, y_train_balanced, cv=3, scoring="f1").mean()
    return score

# Run Optuna Optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)  # Runs 20 different experiments

# Get best parameters
best_params = study.best_params
print("Best parameters found:", best_params)

# Train final model with best params
xgb_model = XGBClassifier(**best_params, tree_method="hist", objective="binary:logistic", random_state=42)
xgb_model.fit(X_train_balanced, y_train_balanced)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))

# Save best parameters to log file
with open(os.path.join(output_dir, "best_hyperparameters.txt"), "w") as log_file:
    log_file.write(str(best_params))

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

print("Training complete. Best model saved.")
