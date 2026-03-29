import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

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

# ---- Model 3: LightGBM (Gradient Boosting) ----
# LightGBM is chosen because:
# 1. It is a highly efficient gradient boosting framework optimized for speed
# 2. It handles sparse data (TF-IDF) efficiently with histogram-based algorithms
# 3. It captures non-linear relationships and feature interactions that linear models miss
# 4. Built-in regularization (num_leaves, min_child_samples) prevents overfitting
# 5. Often achieves state-of-the-art results on tabular/structured data

print("\n" + "-" * 80)
print("Model 3: LightGBM (Gradient Boosting)")

# Hyperparameter tuning using GridSearchCV
# - n_estimators: Number of boosting rounds (trees)
# - learning_rate: Step size shrinkage to prevent overfitting
# - num_leaves: Max number of leaves per tree (controls complexity)
# - max_depth: Maximum tree depth (-1 means no limit)
lgbm_param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "num_leaves": [31, 50],
    "max_depth": [-1, 10],
}

lgbm = LGBMClassifier(random_state=42, verbose=-1, force_col_wise=True)
lgbm_grid = GridSearchCV(
    lgbm, lgbm_param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
)

print("Running GridSearchCV for LightGBM...")
lgbm_grid.fit(X_train, y_train)

print(f"\nBest parameters: {lgbm_grid.best_params_}")
print(f"Best cross-validation accuracy: {lgbm_grid.best_score_:.4f}")

# Cross-validation scores for best model
lgbm_best = lgbm_grid.best_estimator_
lgbm_cv_scores = cross_val_score(lgbm_best, X_train, y_train, cv=5, scoring="accuracy")
print(f"\nCross-validation scores (5-fold): {lgbm_cv_scores}")
print(
    f"Mean CV accuracy: {lgbm_cv_scores.mean():.4f} (+/- {lgbm_cv_scores.std() * 2:.4f})"
)

# Training accuracy
lgbm_train_pred = lgbm_best.predict(X_train)
lgbm_train_acc = accuracy_score(y_train, lgbm_train_pred)
print(f"Training accuracy: {lgbm_train_acc:.4f}")

# Feature importance analysis (top 20 features)
print("\nTop 20 most important features (by gain):")
feature_importance = lgbm_best.feature_importances_
tfidf_vectorizer = joblib.load(os.path.join(data_dir, "tfidf_vectorizer.pkl"))
feature_names = tfidf_vectorizer.get_feature_names_out()
importance_df = sorted(
    zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True
)[:20]
for i, (feat, imp) in enumerate(importance_df, 1):
    print(f"  {i:2d}. {feat:<25} {imp:.4f}")

# ---- Model 4: MLP (Multi-Layer Perceptron) ----
# MLP is chosen because:
# 1. Neural networks can learn complex non-linear decision boundaries
# 2. Multiple hidden layers enable hierarchical feature learning
# 3. Works well with normalized/scaled features (TF-IDF is already normalized)
# 4. Can capture semantic relationships between words through learned embeddings
# 5. Provides a bridge between traditional ML and deep learning approaches

print("\n" + "-" * 80)
print("Model 4: MLP (Multi-Layer Perceptron)")

# Hyperparameter tuning using GridSearchCV
# - hidden_layer_sizes: Architecture of hidden layers (neurons per layer)
# - activation: Non-linear activation function
# - alpha: L2 regularization strength to prevent overfitting
# - learning_rate_init: Initial learning rate for weight updates
# - max_iter: Maximum epochs for training
mlp_param_grid = {
    "hidden_layer_sizes": [(100,), (100, 50), (128, 64)],
    "activation": ["relu"],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate_init": [0.001],
    "max_iter": [500],
}

mlp = MLPClassifier(random_state=42, early_stopping=False)
mlp_grid = GridSearchCV(
    mlp, mlp_param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
)

print("Running GridSearchCV for MLP...")
mlp_grid.fit(X_train, y_train)

print(f"\nBest parameters: {mlp_grid.best_params_}")
print(f"Best cross-validation accuracy: {mlp_grid.best_score_:.4f}")

# Cross-validation scores for best model
mlp_best = mlp_grid.best_estimator_
mlp_cv_scores = cross_val_score(mlp_best, X_train, y_train, cv=5, scoring="accuracy")
print(f"\nCross-validation scores (5-fold): {mlp_cv_scores}")
print(
    f"Mean CV accuracy: {mlp_cv_scores.mean():.4f} (+/- {mlp_cv_scores.std() * 2:.4f})"
)

# Training accuracy
mlp_train_pred = mlp_best.predict(X_train)
mlp_train_acc = accuracy_score(y_train, mlp_train_pred)
print(f"Training accuracy: {mlp_train_acc:.4f}")

# MLP architecture analysis
print(f"\nMLP Architecture:")
print(f"  Input layer   : {X_train.shape[1]} features")
for i, layer_size in enumerate(mlp_best.hidden_layer_sizes):
    if isinstance(mlp_best.hidden_layer_sizes, tuple):
        print(f"  Hidden layer {i + 1}: {layer_size} neurons")
    else:
        print(f"  Hidden layer 1: {mlp_best.hidden_layer_sizes} neurons")
        break
print(f"  Output layer  : {len(np.unique(y_train))} classes")
print(f"  Activation    : {mlp_best.activation}")
print(f"  Training epochs (actual): {mlp_best.n_iter_}")

# ---- Training results summary ----
print("\n" + "-" * 80)
print("Training Results Summary")
print(f"\n{'Model':<25} {'Best CV Accuracy':<20} {'Training Accuracy':<20}")
print("-" * 65)
print(f"{'Logistic Regression':<25} {lr_grid.best_score_:<20.4f} {lr_train_acc:<20.4f}")
print(f"{'SVM (Linear)':<25} {svm_grid.best_score_:<20.4f} {svm_train_acc:<20.4f}")
print(f"{'LightGBM':<25} {lgbm_grid.best_score_:<20.4f} {lgbm_train_acc:<20.4f}")
print(f"{'MLP':<25} {mlp_grid.best_score_:<20.4f} {mlp_train_acc:<20.4f}")

# Analyze overfitting by comparing training vs CV accuracy
print("\n" + "-" * 80)
print("Overfitting Analysis (Training Accuracy - CV Accuracy)")
print(f"\n{'Model':<25} {'Difference':<15} {'Assessment':<30}")
print("-" * 70)
for name, train_acc, cv_acc in [
    ("Logistic Regression", lr_train_acc, lr_grid.best_score_),
    ("SVM (Linear)", svm_train_acc, svm_grid.best_score_),
    ("LightGBM", lgbm_train_acc, lgbm_grid.best_score_),
    ("MLP", mlp_train_acc, mlp_grid.best_score_),
]:
    diff = train_acc - cv_acc
    if diff < 0.02:
        assessment = "Minimal overfitting"
    elif diff < 0.05:
        assessment = "Slight overfitting"
    elif diff < 0.10:
        assessment = "Moderate overfitting"
    else:
        assessment = "Significant overfitting"
    print(f"{name:<25} {diff:<15.4f} {assessment:<30}")

# ---- save models ----
print("\n" + "-" * 80)
print("Saving Models")

joblib.dump(lr_best, os.path.join(data_dir, "logistic_regression.pkl"))
joblib.dump(svm_best, os.path.join(data_dir, "svm.pkl"))
joblib.dump(lgbm_best, os.path.join(data_dir, "lightgbm.pkl"))
joblib.dump(mlp_best, os.path.join(data_dir, "mlp.pkl"))

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
    "lightgbm": {
        "best_params": lgbm_grid.best_params_,
        "best_cv_score": lgbm_grid.best_score_,
        "cv_scores": lgbm_cv_scores.tolist(),
        "cv_mean": lgbm_cv_scores.mean(),
        "cv_std": lgbm_cv_scores.std(),
        "train_accuracy": lgbm_train_acc,
    },
    "mlp": {
        "best_params": mlp_grid.best_params_,
        "best_cv_score": mlp_grid.best_score_,
        "cv_scores": mlp_cv_scores.tolist(),
        "cv_mean": mlp_cv_scores.mean(),
        "cv_std": mlp_cv_scores.std(),
        "train_accuracy": mlp_train_acc,
        "architecture": mlp_best.hidden_layer_sizes,
        "n_iter": mlp_best.n_iter_,
    },
}
joblib.dump(training_results, os.path.join(data_dir, "training_results.pkl"))

print(f"Saved logistic_regression.pkl")
print(f"Saved svm.pkl")
print(f"Saved lightgbm.pkl")
print(f"Saved mlp.pkl")
print(f"Saved training_results.pkl")

print("\n" + "-" * 80)
print("Model training complete.")
