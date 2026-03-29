import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import emoji

# ---- load test data and models ----
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
plots_dir = os.path.join(script_dir, "plots")

X_test_tfidf = joblib.load(os.path.join(data_dir, "X_test_tfidf.pkl"))
y_test = joblib.load(os.path.join(data_dir, "y_test.pkl"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

lr_model = joblib.load(os.path.join(data_dir, "logistic_regression.pkl"))
svm_model = joblib.load(os.path.join(data_dir, "svm.pkl"))
training_results = joblib.load(os.path.join(data_dir, "training_results.pkl"))

print(f"Loaded test data: {len(y_test)} samples")
print(f"Test set distribution:")
unique, counts = np.unique(y_test, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  {label}: {count}")

label_order = ["positive", "neutral", "negative"]

# ---- ML model predictions ----
print("\n" + "-" * 80)
print("ML Model Testing")

lr_pred = lr_model.predict(X_test_tfidf)
svm_pred = svm_model.predict(X_test_tfidf)

# ---- lexicon model predictions on test data ----
# To compare "apples to apples", we run VADER and TextBlob on the same test set
# that the ML models are evaluated on.
print("\n" + "-" * 80)
print("Lexicon Models on Test Data")

VADER_POS_THRESH = 0.05
VADER_NEG_THRESH = -0.05
TEXTBLOB_POS_THRESH = 0.05
TEXTBLOB_NEG_THRESH = -0.05


def clean_for_vader(text):
    """Minimal cleaning for VADER - preserves casing, punctuation, emojis."""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_for_textblob(text):
    """Full cleaning for TextBlob - demojize, lowercase, strip special chars."""
    text = emoji.demojize(str(text), delimiters=(" ", " "))
    text = re.sub(r":([a-z_]+):", r"\1", text)
    text = text.replace("_", " ")
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Prepare text for lexicon models
test_df["vader_text"] = test_df["text"].apply(clean_for_vader)
test_df["textblob_text"] = test_df["text"].apply(clean_for_textblob)

# VADER predictions
analyzer = SentimentIntensityAnalyzer()
test_df["vader_compound"] = test_df["vader_text"].apply(
    lambda x: analyzer.polarity_scores(x)["compound"]
)


def vader_classify(compound):
    if compound >= VADER_POS_THRESH:
        return "positive"
    elif compound <= VADER_NEG_THRESH:
        return "negative"
    else:
        return "neutral"


test_df["vader_pred"] = test_df["vader_compound"].apply(vader_classify)

# TextBlob predictions
test_df["textblob_polarity"] = test_df["textblob_text"].apply(
    lambda x: TextBlob(x).sentiment.polarity
)


def textblob_classify(polarity):
    if polarity >= TEXTBLOB_POS_THRESH:
        return "positive"
    elif polarity <= TEXTBLOB_NEG_THRESH:
        return "negative"
    else:
        return "neutral"


test_df["textblob_pred"] = test_df["textblob_polarity"].apply(textblob_classify)

vader_pred = test_df["vader_pred"].values
textblob_pred = test_df["textblob_pred"].values

print(f"VADER predictions    : {len(vader_pred)}")
print(f"TextBlob predictions : {len(textblob_pred)}")

# ---- metrics calculation ----
print("\n" + "-" * 80)
print("Test Results")


def calculate_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{name}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (weighted)")
    print(f"  Recall    : {rec:.4f}  (weighted)")
    print(f"  F1 Score  : {f1:.4f}  (weighted)")
    print(f"\n  Per-class report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}


lr_metrics = calculate_metrics("Logistic Regression", y_test, lr_pred)
svm_metrics = calculate_metrics("SVM", y_test, svm_pred)
vader_metrics = calculate_metrics("VADER", y_test, vader_pred)
textblob_metrics = calculate_metrics("TextBlob", y_test, textblob_pred)

# ---- comparison table (all 4 models) ----
print("\n" + "-" * 80)
print("Comparison Table - All Models on Same Test Data (600 reviews)")

comparison = pd.DataFrame([lr_metrics, svm_metrics, vader_metrics, textblob_metrics])
comparison = comparison.set_index("Model")
print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))

# ---- confusion matrices ----
print("\n" + "-" * 80)
print("Generating Confusion Matrices")


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_order,
        yticklabels=label_order,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=120)
    plt.close()
    print(f"Saved: {filename}")


plot_confusion_matrix(
    y_test,
    lr_pred,
    "Logistic Regression Confusion Matrix",
    "05_lr_confusion_matrix.png",
)
plot_confusion_matrix(
    y_test, svm_pred, "SVM Confusion Matrix", "06_svm_confusion_matrix.png"
)
plot_confusion_matrix(
    y_test,
    vader_pred,
    "VADER Confusion Matrix (Test Set)",
    "07_vader_confusion_matrix.png",
)
plot_confusion_matrix(
    y_test,
    textblob_pred,
    "TextBlob Confusion Matrix (Test Set)",
    "08_textblob_confusion_matrix.png",
)

# ---- side-by-side comparison bar chart ----
print("\n" + "-" * 80)
print("Generating Comparison Charts")

metrics_df = pd.DataFrame(
    {
        "Logistic Regression": [
            lr_metrics["Accuracy"],
            lr_metrics["Precision"],
            lr_metrics["Recall"],
            lr_metrics["F1"],
        ],
        "SVM": [
            svm_metrics["Accuracy"],
            svm_metrics["Precision"],
            svm_metrics["Recall"],
            svm_metrics["F1"],
        ],
        "VADER": [
            vader_metrics["Accuracy"],
            vader_metrics["Precision"],
            vader_metrics["Recall"],
            vader_metrics["F1"],
        ],
        "TextBlob": [
            textblob_metrics["Accuracy"],
            textblob_metrics["Precision"],
            textblob_metrics["Recall"],
            textblob_metrics["F1"],
        ],
    },
    index=["Accuracy", "Precision", "Recall", "F1"],
)

fig, ax = plt.subplots(figsize=(12, 6))
metrics_df.plot(kind="bar", ax=ax, edgecolor="white", width=0.8)
ax.set_title("ML vs Lexicon Models - Performance Comparison (Same Test Data)")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "09_model_comparison.png"), dpi=120)
plt.close()
print("Saved: 09_model_comparison.png")

# ML vs Lexicon grouped comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ML models
ml_df = metrics_df[["Logistic Regression", "SVM"]]
ml_df.plot(kind="bar", ax=axes[0], edgecolor="white", color=["steelblue", "coral"])
axes[0].set_title("Machine Learning Models")
axes[0].set_ylabel("Score")
axes[0].set_ylim(0, 1)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
axes[0].legend(loc="lower right")

# Lexicon models
lex_df = metrics_df[["VADER", "TextBlob"]]
lex_df.plot(kind="bar", ax=axes[1], edgecolor="white", color=["forestgreen", "purple"])
axes[1].set_title("Lexicon Models")
axes[1].set_ylabel("Score")
axes[1].set_ylim(0, 1)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "10_ml_vs_lexicon.png"), dpi=120)
plt.close()
print("Saved: 10_ml_vs_lexicon.png")

# ---- save results ----
print("\n" + "-" * 80)
print("Saving Results")

# Save comparison table as CSV
comparison.to_csv(os.path.join(data_dir, "model_comparison.csv"))
print("Saved: model_comparison.csv")

# Save predictions for analysis
test_df["lr_pred"] = lr_pred
test_df["svm_pred"] = svm_pred
test_df.to_csv(os.path.join(data_dir, "test_predictions.csv"), index=False)
print("Saved: test_predictions.csv")

print("\n" + "-" * 80)
print("Evaluation complete.")
