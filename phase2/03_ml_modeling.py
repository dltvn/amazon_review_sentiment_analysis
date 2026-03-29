import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score

# ---- load preprocessed data ----
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")

X_train = joblib.load(os.path.join(data_dir, "X_train_tfidf.pkl"))
y_train = joblib.load(os.path.join(data_dir, "y_train.pkl"))

print(f"Loaded training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Class distribution in training set:")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  {label}: {count}")

# ---- Model 1: Logistic Regression ----
# Logistic Regression is chosen because:
# 1. It works well with high-dimensional sparse TF-IDF features
# 2. It provides probability estimates for each class
# 3. It is fast to train and interpretable
# 4. L2 regularization prevents overfitting on sparse text data

print("\n" + "-" * 80)
print("Model 1: Logistic Regression")

# Hyperparameter tuning using GridSearchCV with 5-fold cross-validation
# - C: Inverse regularization strength (smaller = stronger regularization)
# - solver: Algorithm for optimization (lbfgs works well for multiclass)
# - max_iter: Increased to ensure convergence with sparse data
lr_param_grid = {"C": [0.1, 1.0, 10.0], "solver": ["lbfgs"], "max_iter": [500]}

lr = LogisticRegression(random_state=42)
lr_grid = GridSearchCV(
    lr, lr_param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
)

print("Running GridSearchCV for Logistic Regression...")
lr_grid.fit(X_train, y_train)

print(f"\nBest parameters: {lr_grid.best_params_}")
print(f"Best cross-validation accuracy: {lr_grid.best_score_:.4f}")

# Cross-validation scores for best model
lr_best = lr_grid.best_estimator_
lr_cv_scores = cross_val_score(lr_best, X_train, y_train, cv=5, scoring="accuracy")
print(f"\nCross-validation scores (5-fold): {lr_cv_scores}")
print(f"Mean CV accuracy: {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std() * 2:.4f})")

# Training accuracy
lr_train_pred = lr_best.predict(X_train)
lr_train_acc = accuracy_score(y_train, lr_train_pred)
print(f"Training accuracy: {lr_train_acc:.4f}")

# ---- Model 2: Support Vector Machine (SVM) ----
# SVM is chosen because:
# 1. It excels at high-dimensional classification problems like text
# 2. Linear kernel is effective for TF-IDF representations (linearly separable in high dimensions)
# 3. Maximum margin principle provides good generalization
# 4. Robust to overfitting, especially with proper regularization

print("\n" + "-" * 80)
print("Model 2: Support Vector Machine (SVM)")

# Hyperparameter tuning using GridSearchCV
# - C: Regularization parameter (trade-off between margin width and misclassification)
# - kernel: Linear kernel is standard for text classification (efficient, effective)
svm_param_grid = {"C": [0.1, 1.0, 10.0], "kernel": ["linear"]}

svm = SVC(random_state=42)
svm_grid = GridSearchCV(
    svm, svm_param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
)

print("Running GridSearchCV for SVM...")
svm_grid.fit(X_train, y_train)

print(f"\nBest parameters: {svm_grid.best_params_}")
print(f"Best cross-validation accuracy: {svm_grid.best_score_:.4f}")

# Cross-validation scores for best model
svm_best = svm_grid.best_estimator_
svm_cv_scores = cross_val_score(svm_best, X_train, y_train, cv=5, scoring="accuracy")
print(f"\nCross-validation scores (5-fold): {svm_cv_scores}")
print(
    f"Mean CV accuracy: {svm_cv_scores.mean():.4f} (+/- {svm_cv_scores.std() * 2:.4f})"
)

# Training accuracy
svm_train_pred = svm_best.predict(X_train)
svm_train_acc = accuracy_score(y_train, svm_train_pred)
print(f"Training accuracy: {svm_train_acc:.4f}")

# ---- Training results summary ----
print("\n" + "-" * 80)
print("Training Results Summary")
print(f"\n{'Model':<25} {'Best CV Accuracy':<20} {'Training Accuracy':<20}")
print("-" * 65)
print(f"{'Logistic Regression':<25} {lr_grid.best_score_:<20.4f} {lr_train_acc:<20.4f}")
print(f"{'SVM (Linear)':<25} {svm_grid.best_score_:<20.4f} {svm_train_acc:<20.4f}")

# ---- save models ----
print("\n" + "-" * 80)
print("Saving Models")

joblib.dump(lr_best, os.path.join(data_dir, "logistic_regression.pkl"))
joblib.dump(svm_best, os.path.join(data_dir, "svm.pkl"))

# Save training results for report
training_results = {
    "logistic_regression": {
        "best_params": lr_grid.best_params_,
        "best_cv_score": lr_grid.best_score_,
        "cv_scores": lr_cv_scores.tolist(),
        "cv_mean": lr_cv_scores.mean(),
        "cv_std": lr_cv_scores.std(),
        "train_accuracy": lr_train_acc,
    },
    "svm": {
        "best_params": svm_grid.best_params_,
        "best_cv_score": svm_grid.best_score_,
        "cv_scores": svm_cv_scores.tolist(),
        "cv_mean": svm_cv_scores.mean(),
        "cv_std": svm_cv_scores.std(),
        "train_accuracy": svm_train_acc,
    },
}
joblib.dump(training_results, os.path.join(data_dir, "training_results.pkl"))

print(f"Saved logistic_regression.pkl")
print(f"Saved svm.pkl")
print(f"Saved training_results.pkl")

print("\n" + "-" * 80)
print("Model training complete.")
