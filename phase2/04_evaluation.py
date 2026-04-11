import os
import warnings
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
    precision_recall_fscore_support,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

# ---- load test data and models ----
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

evaluation_plot_files = [
    "05_lr_confusion_matrix.png",
    "06_svm_confusion_matrix.png",
    "07_lgbm_confusion_matrix.png",
    "07_vader_confusion_matrix.png",
    "08_mlp_confusion_matrix.png",
    "08_textblob_confusion_matrix.png",
    "09_lr_confusion_matrix.png",
    "09_model_comparison.png",
    "09_vader_confusion_matrix.png",
    "10_ml_vs_lexicon.png",
    "10_svm_confusion_matrix.png",
    "10_textblob_confusion_matrix.png",
    "11_lgbm_confusion_matrix.png",
    "11_model_comparison.png",
    "12_ml_vs_lexicon.png",
    "12_mlp_confusion_matrix.png",
    "13_ml_models_comparison.png",
    "13_vader_confusion_matrix.png",
    "14_per_class_f1.png",
    "14_textblob_confusion_matrix.png",
    "15_model_comparison.png",
    "16_ml_vs_lexicon.png",
    "17_ml_models_comparison.png",
    "18_per_class_f1.png",
]

for filename in evaluation_plot_files:
    file_path = os.path.join(plots_dir, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

X_test_tfidf = joblib.load(os.path.join(data_dir, "X_test_tfidf.pkl"))
y_test = joblib.load(os.path.join(data_dir, "y_test.pkl"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

lr_model = joblib.load(os.path.join(data_dir, "logistic_regression.pkl"))
svm_model = joblib.load(os.path.join(data_dir, "svm.pkl"))
lgbm_model = joblib.load(os.path.join(data_dir, "lightgbm.pkl"))
mlp_model = joblib.load(os.path.join(data_dir, "mlp.pkl"))
training_results = joblib.load(os.path.join(data_dir, "training_results.pkl"))

print(f"Loaded test data: {len(y_test)} samples")
print("Test set distribution:")
unique, counts = np.unique(y_test, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  {label}: {count}")

label_order = ["positive", "neutral", "negative"]

# ---- ML model predictions ----
print("\n" + "-" * 80)
print("ML Model Testing")

lr_pred = lr_model.predict(X_test_tfidf)
svm_pred = svm_model.predict(X_test_tfidf)
lgbm_pred = lgbm_model.predict(X_test_tfidf)
mlp_pred = mlp_model.predict(X_test_tfidf)

# ---- lexicon model predictions on test data ----
# To compare apples to apples, run VADER and TextBlob on the same test set.
print("\n" + "-" * 80)
print("Lexicon Models on Test Data")

VADER_POS_THRESH = 0.05
VADER_NEG_THRESH = -0.05
TEXTBLOB_POS_THRESH = 0.05
TEXTBLOB_NEG_THRESH = -0.05

test_df["vader_text"] = test_df["vader_text"].fillna("")
test_df["combined_text"] = test_df["combined_text"].fillna("")

analyzer = SentimentIntensityAnalyzer()
vader_scores = test_df["vader_text"].apply(lambda x: analyzer.polarity_scores(str(x)))
test_df["vader_pos"] = vader_scores.apply(lambda s: s["pos"])
test_df["vader_neg"] = vader_scores.apply(lambda s: s["neg"])
test_df["vader_neu"] = vader_scores.apply(lambda s: s["neu"])
test_df["vader_compound"] = vader_scores.apply(lambda s: s["compound"])


def predict_vader(compound):
    if compound >= VADER_POS_THRESH:
        return "positive"
    elif compound <= VADER_NEG_THRESH:
        return "negative"
    else:
        return "neutral"


test_df["vader_pred"] = test_df["vader_compound"].apply(predict_vader)
test_df["textblob_polarity"] = test_df["combined_text"].apply(
    lambda x: TextBlob(str(x)).sentiment.polarity
)
test_df["textblob_pred"] = test_df["textblob_polarity"].apply(
    lambda p: (
        "positive"
        if p >= TEXTBLOB_POS_THRESH
        else ("negative" if p <= TEXTBLOB_NEG_THRESH else "neutral")
    )
)

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
    print("\n  Per-class report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}


lr_metrics = calculate_metrics("Logistic Regression", y_test, lr_pred)
svm_metrics = calculate_metrics("SVM", y_test, svm_pred)
lgbm_metrics = calculate_metrics("LightGBM", y_test, lgbm_pred)
mlp_metrics = calculate_metrics("MLP", y_test, mlp_pred)
vader_metrics = calculate_metrics("VADER", y_test, vader_pred)
textblob_metrics = calculate_metrics("TextBlob", y_test, textblob_pred)

# ---- comparison table (all 6 models) ----
print("\n" + "-" * 80)
print(f"Comparison Table - All Models on Same Test Data ({len(y_test)} reviews)")

comparison = pd.DataFrame(
    [
        lr_metrics,
        svm_metrics,
        lgbm_metrics,
        mlp_metrics,
        vader_metrics,
        textblob_metrics,
    ]
)
comparison = comparison.set_index("Model")
print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))

# ---- confusion matrices ----
print("\n" + "-" * 80)
print("Generating Confusion Matrices")

confusion_matrix_rows = []


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    cm_df = pd.DataFrame(cm, index=label_order, columns=label_order)
    print(f"\n{title}")
    print(cm_df.to_string())

    for actual_label in label_order:
        for predicted_label in label_order:
            confusion_matrix_rows.append(
                {
                    "model": title.replace(" Confusion Matrix", "").replace(
                        " (Test Set)", ""
                    ),
                    "actual": actual_label,
                    "predicted": predicted_label,
                    "count": cm_df.loc[actual_label, predicted_label],
                }
            )

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
    fig.savefig(os.path.join(plots_dir, filename), dpi=120)
    plt.close(fig)
    print(f"Saved: {filename}")


plot_confusion_matrix(
    y_test,
    lr_pred,
    "Logistic Regression Confusion Matrix",
    "09_lr_confusion_matrix.png",
)
plot_confusion_matrix(
    y_test, svm_pred, "SVM Confusion Matrix", "10_svm_confusion_matrix.png"
)
plot_confusion_matrix(
    y_test, lgbm_pred, "LightGBM Confusion Matrix", "11_lgbm_confusion_matrix.png"
)
plot_confusion_matrix(
    y_test, mlp_pred, "MLP Confusion Matrix", "12_mlp_confusion_matrix.png"
)
plot_confusion_matrix(
    y_test,
    vader_pred,
    "VADER Confusion Matrix (Test Set)",
    "13_vader_confusion_matrix.png",
)
plot_confusion_matrix(
    y_test,
    textblob_pred,
    "TextBlob Confusion Matrix (Test Set)",
    "14_textblob_confusion_matrix.png",
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
        "LightGBM": [
            lgbm_metrics["Accuracy"],
            lgbm_metrics["Precision"],
            lgbm_metrics["Recall"],
            lgbm_metrics["F1"],
        ],
        "MLP": [
            mlp_metrics["Accuracy"],
            mlp_metrics["Precision"],
            mlp_metrics["Recall"],
            mlp_metrics["F1"],
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

fig, ax = plt.subplots(figsize=(14, 6))
metrics_df.plot(kind="bar", ax=ax, edgecolor="white", width=0.8)
ax.set_title("All Models - Performance Comparison (Same Test Data)")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(plots_dir, "15_model_comparison.png"), dpi=120)
plt.close(fig)
print("Saved: 15_model_comparison.png")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ml_df = metrics_df[["Logistic Regression", "SVM", "LightGBM", "MLP"]]
ml_df.plot(
    kind="bar",
    ax=axes[0],
    edgecolor="white",
    color=["steelblue", "coral", "forestgreen", "purple"],
)
axes[0].set_title("Machine Learning Models")
axes[0].set_ylabel("Score")
axes[0].set_ylim(0, 1)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
axes[0].legend(loc="lower right", fontsize=8)

lex_df = metrics_df[["VADER", "TextBlob"]]
lex_df.plot(kind="bar", ax=axes[1], edgecolor="white", color=["teal", "orange"])
axes[1].set_title("Lexicon Models")
axes[1].set_ylabel("Score")
axes[1].set_ylim(0, 1)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].legend(loc="lower right")

plt.tight_layout()
fig.savefig(os.path.join(plots_dir, "16_ml_vs_lexicon.png"), dpi=120)
plt.close(fig)
print("Saved: 16_ml_vs_lexicon.png")

fig, ax = plt.subplots(figsize=(12, 6))
ml_colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
ml_df.plot(kind="bar", ax=ax, edgecolor="white", color=ml_colors, width=0.8)
ax.set_title("Machine Learning Models - Detailed Performance Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(loc="lower right")

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=7, rotation=90, padding=3)

plt.tight_layout()
fig.savefig(os.path.join(plots_dir, "17_ml_models_comparison.png"), dpi=120)
plt.close(fig)
print("Saved: 17_ml_models_comparison.png")

# ---- Per-class performance analysis ----
print("\n" + "-" * 80)
print("Per-Class Performance Analysis")


def get_per_class_metrics(y_true, y_pred, model_name):
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=label_order, zero_division=0
    )
    return {
        "Model": model_name,
        "Positive F1": f1[0],
        "Neutral F1": f1[1],
        "Negative F1": f1[2],
    }


per_class_data = [
    get_per_class_metrics(y_test, lr_pred, "Logistic Regression"),
    get_per_class_metrics(y_test, svm_pred, "SVM"),
    get_per_class_metrics(y_test, lgbm_pred, "LightGBM"),
    get_per_class_metrics(y_test, mlp_pred, "MLP"),
    get_per_class_metrics(y_test, vader_pred, "VADER"),
    get_per_class_metrics(y_test, textblob_pred, "TextBlob"),
]

per_class_df = pd.DataFrame(per_class_data).set_index("Model")
print("\nF1 Score by Class:")
print(per_class_df.to_string(float_format=lambda x: f"{x:.4f}"))

fig, ax = plt.subplots(figsize=(12, 6))
per_class_df.plot(kind="bar", ax=ax, edgecolor="white", width=0.8)
ax.set_title("F1 Score by Sentiment Class - All Models")
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(os.path.join(plots_dir, "18_per_class_f1.png"), dpi=120)
plt.close(fig)
print("Saved: 18_per_class_f1.png")

# ---- save results ----
print("\n" + "-" * 80)
print("Saving Results")

comparison.to_csv(os.path.join(data_dir, "model_comparison.csv"))
print("Saved: model_comparison.csv")

per_class_df.to_csv(os.path.join(data_dir, "per_class_metrics.csv"))
print("Saved: per_class_metrics.csv")

confusion_matrix_df = pd.DataFrame(confusion_matrix_rows)
confusion_matrix_df.to_csv(
    os.path.join(data_dir, "confusion_matrices.csv"), index=False
)
print("Saved: confusion_matrices.csv")

test_df["lr_pred"] = lr_pred
test_df["svm_pred"] = svm_pred
test_df["lgbm_pred"] = lgbm_pred
test_df["mlp_pred"] = mlp_pred
test_df.to_csv(os.path.join(data_dir, "test_predictions.csv"), index=False)
print("Saved: test_predictions.csv")

# ---- Model ranking and insights ----
print("\n" + "-" * 80)
print("Model Ranking and Insights")

ranking = comparison.sort_values("F1", ascending=False)
print("\nModels ranked by F1 Score:")
for i, (model, row) in enumerate(ranking.iterrows(), 1):
    print(f"  {i}. {model:<25} F1: {row['F1']:.4f}")

best_model = ranking.index[0]
best_f1 = ranking.iloc[0]["F1"]
print(f"\nBest performing model: {best_model} (F1: {best_f1:.4f})")

ml_models = ["Logistic Regression", "SVM", "LightGBM", "MLP"]
lexicon_models = ["VADER", "TextBlob"]

ml_avg_f1 = comparison.loc[ml_models, "F1"].mean()
lex_avg_f1 = comparison.loc[lexicon_models, "F1"].mean()

print(f"\nAverage CV accuracy by model from training:")
for model_name, result in training_results.items():
    print(f"  {model_name}: {result['best_cv_score']:.4f}")

print(f"\nML models average F1: {ml_avg_f1:.4f}")
print(f"Lexicon models average F1: {lex_avg_f1:.4f}")
print(
    f"ML improvement over lexicon: {((ml_avg_f1 - lex_avg_f1) / lex_avg_f1 * 100):.1f}%"
)

print("\n" + "-" * 80)
print("Evaluation complete.")
