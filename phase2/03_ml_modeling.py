import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

# ---- load preprocessed sample ----
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")

sample_df = pd.read_csv(os.path.join(data_dir, "sample_2000_phase2.csv"))
sample_df["ml_text"] = sample_df["ml_text"].fillna("")

print(f"Loaded {len(sample_df)} reviews from sample_2000_phase2.csv")
print("Class distribution in sample:")
print(sample_df["sentiment"].value_counts().to_string())

# ---- train/test split (70/30 stratified) ----
# Stratified split ensures both train and test sets maintain the same class distribution,
# which is critical for fair evaluation on imbalanced data.
print("\n" + "-" * 80)
print("Train/Test Split")

train_df, test_df = train_test_split(
    sample_df, test_size=0.30, stratify=sample_df["sentiment"], random_state=42
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Total samples : {len(sample_df)}")
print(f"Training set  : {len(train_df)} ({len(train_df) / len(sample_df) * 100:.1f}%)")
print(f"Test set      : {len(test_df)} ({len(test_df) / len(sample_df) * 100:.1f}%)")

print("\nTraining set distribution:")
print(train_df["sentiment"].value_counts().to_string())
print("\nTest set distribution:")
print(test_df["sentiment"].value_counts().to_string())

X_train = train_df["ml_text"]
X_test = test_df["ml_text"]
y_train = train_df["sentiment"]
y_test = test_df["sentiment"]

# ---- TF-IDF vectorization ----
# TF-IDF (Term Frequency-Inverse Document Frequency) was chosen because:
# 1. It captures term importance by downweighting common words (like "the", "is")
# 2. It works well with linear classifiers (Logistic Regression, SVM)
# 3. It produces sparse representations, which are memory-efficient
# 4. Unlike raw counts, TF-IDF considers document frequency, making rare but
#    meaningful terms (like "excellent", "terrible") stand out

print("\n" + "-" * 80)
print("TF-IDF Vectorization")

# Parameters chosen:
# - max_features=5000: Limits vocabulary to top 5000 terms to prevent overfitting
# - ngram_range=(1,2): Includes unigrams and bigrams to capture phrases like "not good"
# - min_df=2: Ignores terms appearing in fewer than 2 documents (typos, very rare words)
# - max_df=0.95: Ignores terms appearing in >95% of documents (too common to be useful)
# - sublinear_tf=True: Applies log scaling to term frequency, reducing impact of very frequent terms
tfidf = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True
)

# Fit on training data only to prevent data leakage
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"Vocabulary size: {len(tfidf.vocabulary_)}")
print(f"Training matrix shape: {X_train_tfidf.shape}")
print(f"Test matrix shape    : {X_test_tfidf.shape}")
print("\nTop 20 features by index:")
vocab_items = sorted(tfidf.vocabulary_.items(), key=lambda x: x[1])[:20]
for word, idx in vocab_items:
    print(f"  {idx}: {word}")

print(
    f"\nLoaded training data: {X_train.shape[0]} samples, {X_train_tfidf.shape[1]} features"
)
print("Class distribution in training set:")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  {label}: {count}")

# ---- Model 1: Logistic Regression ----
print("\n" + "-" * 80)
print("Model 1: Logistic Regression")

lr_param_grid = {"C": [0.1, 1.0, 10.0], "solver": ["lbfgs"], "max_iter": [500]}

lr = LogisticRegression(random_state=42)
lr_grid = GridSearchCV(
    lr, lr_param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
)

print("Running GridSearchCV for Logistic Regression...")
lr_grid.fit(X_train_tfidf, y_train)

print(f"\nBest parameters: {lr_grid.best_params_}")
print(f"Best cross-validation accuracy: {lr_grid.best_score_:.4f}")

lr_best = lr_grid.best_estimator_
lr_cv_scores = cross_val_score(
    lr_best, X_train_tfidf, y_train, cv=5, scoring="accuracy"
)
print(f"\nCross-validation scores (5-fold): {lr_cv_scores}")
print(f"Mean CV accuracy: {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std() * 2:.4f})")

lr_train_pred = lr_best.predict(X_train_tfidf)
lr_train_acc = accuracy_score(y_train, lr_train_pred)
print(f"Training accuracy: {lr_train_acc:.4f}")

# ---- Model 2: Support Vector Machine (SVM) ----
print("\n" + "-" * 80)
print("Model 2: Support Vector Machine (SVM)")

svm_param_grid = {"C": [0.1, 1.0, 10.0], "kernel": ["linear"]}

svm = SVC(random_state=42)
svm_grid = GridSearchCV(
    svm, svm_param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
)

print("Running GridSearchCV for SVM...")
svm_grid.fit(X_train_tfidf, y_train)

print(f"\nBest parameters: {svm_grid.best_params_}")
print(f"Best cross-validation accuracy: {svm_grid.best_score_:.4f}")

svm_best = svm_grid.best_estimator_
svm_cv_scores = cross_val_score(
    svm_best, X_train_tfidf, y_train, cv=5, scoring="accuracy"
)
print(f"\nCross-validation scores (5-fold): {svm_cv_scores}")
print(
    f"Mean CV accuracy: {svm_cv_scores.mean():.4f} (+/- {svm_cv_scores.std() * 2:.4f})"
)

svm_train_pred = svm_best.predict(X_train_tfidf)
svm_train_acc = accuracy_score(y_train, svm_train_pred)
print(f"Training accuracy: {svm_train_acc:.4f}")

# ---- Model 3: LightGBM (Gradient Boosting) ----
print("\n" + "-" * 80)
print("Model 3: LightGBM (Gradient Boosting)")

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
lgbm_grid.fit(X_train_tfidf, y_train)

print(f"\nBest parameters: {lgbm_grid.best_params_}")
print(f"Best cross-validation accuracy: {lgbm_grid.best_score_:.4f}")

lgbm_best = lgbm_grid.best_estimator_
lgbm_cv_scores = cross_val_score(
    lgbm_best, X_train_tfidf, y_train, cv=5, scoring="accuracy"
)
print(f"\nCross-validation scores (5-fold): {lgbm_cv_scores}")
print(
    f"Mean CV accuracy: {lgbm_cv_scores.mean():.4f} (+/- {lgbm_cv_scores.std() * 2:.4f})"
)

lgbm_train_pred = lgbm_best.predict(X_train_tfidf)
lgbm_train_acc = accuracy_score(y_train, lgbm_train_pred)
print(f"Training accuracy: {lgbm_train_acc:.4f}")

print("\nTop 20 most important features (by gain):")
feature_importance = lgbm_best.feature_importances_
feature_names = tfidf.get_feature_names_out()
importance_df = sorted(
    zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True
)[:20]
for i, (feat, imp) in enumerate(importance_df, 1):
    print(f"  {i:2d}. {feat:<25} {imp:.4f}")

# ---- Model 4: MLP (Multi-Layer Perceptron) ----
print("\n" + "-" * 80)
print("Model 4: MLP (Multi-Layer Perceptron)")

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
mlp_grid.fit(X_train_tfidf, y_train)

print(f"\nBest parameters: {mlp_grid.best_params_}")
print(f"Best cross-validation accuracy: {mlp_grid.best_score_:.4f}")

mlp_best = mlp_grid.best_estimator_
mlp_cv_scores = cross_val_score(
    mlp_best, X_train_tfidf, y_train, cv=5, scoring="accuracy"
)
print(f"\nCross-validation scores (5-fold): {mlp_cv_scores}")
print(
    f"Mean CV accuracy: {mlp_cv_scores.mean():.4f} (+/- {mlp_cv_scores.std() * 2:.4f})"
)

mlp_train_pred = mlp_best.predict(X_train_tfidf)
mlp_train_acc = accuracy_score(y_train, mlp_train_pred)
print(f"Training accuracy: {mlp_train_acc:.4f}")

print("\nMLP Architecture:")
print(f"  Input layer   : {X_train_tfidf.shape[1]} features")
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

# ---- save outputs ----
print("\n" + "-" * 80)
print("Saving Outputs")

train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)

joblib.dump(tfidf, os.path.join(data_dir, "tfidf_vectorizer.pkl"))
joblib.dump(X_train_tfidf, os.path.join(data_dir, "X_train_tfidf.pkl"))
joblib.dump(X_test_tfidf, os.path.join(data_dir, "X_test_tfidf.pkl"))
joblib.dump(y_train.values, os.path.join(data_dir, "y_train.pkl"))
joblib.dump(y_test.values, os.path.join(data_dir, "y_test.pkl"))

joblib.dump(lr_best, os.path.join(data_dir, "logistic_regression.pkl"))
joblib.dump(svm_best, os.path.join(data_dir, "svm.pkl"))
joblib.dump(lgbm_best, os.path.join(data_dir, "lightgbm.pkl"))
joblib.dump(mlp_best, os.path.join(data_dir, "mlp.pkl"))

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

print(f"Saved train.csv        -> {len(train_df)} rows")
print(f"Saved test.csv         -> {len(test_df)} rows")
print("Saved tfidf_vectorizer.pkl")
print("Saved X_train_tfidf.pkl, X_test_tfidf.pkl")
print("Saved y_train.pkl, y_test.pkl")
print("Saved logistic_regression.pkl")
print("Saved svm.pkl")
print("Saved lightgbm.pkl")
print("Saved mlp.pkl")
print("Saved training_results.pkl")

print("\n" + "-" * 80)
print("Model training complete.")
