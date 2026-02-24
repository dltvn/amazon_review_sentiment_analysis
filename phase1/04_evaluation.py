import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---- load predictions ----
script_dir = os.path.dirname(os.path.abspath(__file__))
pred_path = os.path.join(script_dir, "data", "predictions.csv")
df = pd.read_csv(pred_path)

y_true = df["sentiment"]
y_vader = df["vader_pred"]
y_tb = df["textblob_pred"]

label_order = ["positive", "neutral", "negative"]


# ---- helper: print metrics ----
def print_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n{name}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (weighted)")
    print(f"  Recall    : {rec:.4f}  (weighted)")
    print(f"  F1 Score  : {f1:.4f}  (weighted)")
    print(
        f"\n  Per-class report:\n{classification_report(y_true, y_pred, zero_division=0)}"
    )
    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}


print("-" * 80)
print("Model Evaluation")

vader_metrics = print_metrics("VADER", y_true, y_vader)
tb_metrics = print_metrics("TextBlob", y_true, y_tb)

# ---- comparison table ----
print("-" * 80)
print("Comparison Table")
comparison = pd.DataFrame([vader_metrics, tb_metrics]).set_index("Model")
comparison = comparison.map(lambda x: f"{x:.4f}")
print(comparison.to_string())

# ---- plots ----
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)


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
    plt.show()
    plt.close()
    print(f"Saved: {filename}")


plot_confusion_matrix(
    y_true, y_vader, "VADER Confusion Matrix", "09_vader_confusion_matrix.png"
)
plot_confusion_matrix(
    y_true, y_tb, "TextBlob Confusion Matrix", "10_textblob_confusion_matrix.png"
)

# ---- side-by-side metric bar chart ----
metrics_df = pd.DataFrame(
    {
        "VADER": [
            vader_metrics["Accuracy"],
            vader_metrics["Precision"],
            vader_metrics["Recall"],
            vader_metrics["F1"],
        ],
        "TextBlob": [
            tb_metrics["Accuracy"],
            tb_metrics["Precision"],
            tb_metrics["Recall"],
            tb_metrics["F1"],
        ],
    },
    index=["Accuracy", "Precision", "Recall", "F1"],
)
fig, ax = plt.subplots(figsize=(8, 5))
metrics_df.plot(kind="bar", ax=ax, edgecolor="white")
ax.set_title("VADER vs TextBlob - Performance Metrics")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "11_model_comparison.png"), dpi=120)
plt.show()
plt.close()
print("Saved: 11_model_comparison.png")

print("\n" + "-" * 80)
print("Evaluation complete.")
