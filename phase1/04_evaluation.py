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

# ============================================================
# MIXED REVIEW ANALYSIS
# ============================================================
print("\n" + "-" * 80)
print("MIXED REVIEW ANALYSIS")

# ---- Method 1: Borderline zone (score near decision boundary) ----
# Captures reviews where the model is least confident.
# These are often ambiguous reviews where sentiment signals are weak.
BORDER_LOW = -0.2
BORDER_HIGH = 0.2

vader_borderline = df[
    (df["vader_compound"] >= BORDER_LOW) & (df["vader_compound"] <= BORDER_HIGH)
].copy()

tb_borderline = df[
    (df["textblob_polarity"] >= BORDER_LOW) & (df["textblob_polarity"] <= BORDER_HIGH)
].copy()

print(f"\n[Method 1] Borderline zone ({BORDER_LOW} to {BORDER_HIGH}):")
print(f"  VADER    borderline reviews : {len(vader_borderline)}")
print(f"  TextBlob borderline reviews : {len(tb_borderline)}")

# Misclassified within borderline — predicted neutral but true label != neutral
vader_border_misc = vader_borderline[
    (vader_borderline["vader_pred"] == "neutral")
    & (vader_borderline["sentiment"] != "neutral")
]
print(
    f"\n  VADER predicted neutral but true label is pos/neg: {len(vader_border_misc)}"
)
if len(vader_border_misc) > 0:
    print(
        vader_border_misc[["sentiment", "vader_pred", "vader_compound", "vader_text"]]
        .head(3)
        .to_string()
    )

# ---- Method 2: Truly mixed reviews (strong pos AND strong neg signals) ----
# A truly mixed review has both high positive and high negative sub-scores
# simultaneously. This is more rigorous than just looking at compound alone.
MIXED_POS_THRESH = 0.15
MIXED_NEG_THRESH = 0.15

truly_mixed = df[
    (df["vader_pos"] > MIXED_POS_THRESH) & (df["vader_neg"] > MIXED_NEG_THRESH)
].copy()

print(f"\n[Method 2] Truly mixed reviews")
print(f"  (vader_pos > {MIXED_POS_THRESH} AND vader_neg > {MIXED_NEG_THRESH})")
print(f"  Count: {len(truly_mixed)}")

if len(truly_mixed) > 0:
    print(f"\n  Sentiment distribution in truly mixed reviews:")
    print(truly_mixed["sentiment"].value_counts().to_string())
    print(f"\n  Sample of truly mixed reviews:")
    cols = [
        "sentiment",
        "vader_pred",
        "vader_pos",
        "vader_neg",
        "vader_compound",
        "vader_text",
    ]
    print(truly_mixed[cols].head(5).to_string())

# ---- Scatter plot: VADER compound score by true sentiment ----
# Visualizes how well the compound score separates the three classes
# and highlights where the model is confused.
color_map = {"positive": "steelblue", "neutral": "gold", "negative": "tomato"}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# VADER scatter
for label, grp in df.groupby("sentiment"):
    axes[0].scatter(
        grp.index,
        grp["vader_compound"],
        label=label,
        alpha=0.4,
        s=10,
        color=color_map[label],
    )
axes[0].axhline(
    y=0.05, color="green", linestyle="--", linewidth=1, label="pos threshold"
)
axes[0].axhline(
    y=-0.05, color="red", linestyle="--", linewidth=1, label="neg threshold"
)
axes[0].set_title("VADER Compound Score by True Sentiment")
axes[0].set_xlabel("Review Index")
axes[0].set_ylabel("Compound Score")
axes[0].legend(markerscale=2, fontsize=8)

# TextBlob scatter
for label, grp in df.groupby("sentiment"):
    axes[1].scatter(
        grp.index,
        grp["textblob_polarity"],
        label=label,
        alpha=0.4,
        s=10,
        color=color_map[label],
    )
axes[1].axhline(
    y=0.05, color="green", linestyle="--", linewidth=1, label="pos threshold"
)
axes[1].axhline(
    y=-0.05, color="red", linestyle="--", linewidth=1, label="neg threshold"
)
axes[1].set_title("TextBlob Polarity Score by True Sentiment")
axes[1].set_xlabel("Review Index")
axes[1].set_ylabel("Polarity Score")
axes[1].legend(markerscale=2, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "12_polarity_scatter.png"), dpi=120)
plt.show()
plt.close()
print("\nSaved: 12_polarity_scatter.png")

# ---- Distribution plot: compound score histogram by true sentiment ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for label in label_order:
    axes[0].hist(
        df[df["sentiment"] == label]["vader_compound"],
        bins=40,
        alpha=0.5,
        label=label,
        color=color_map[label],
    )
axes[0].axvline(x=0.05, color="green", linestyle="--", linewidth=1)
axes[0].axvline(x=-0.05, color="red", linestyle="--", linewidth=1)
axes[0].set_title("VADER Compound Score Distribution by Sentiment")
axes[0].set_xlabel("Compound Score")
axes[0].set_ylabel("Count")
axes[0].legend()

for label in label_order:
    axes[1].hist(
        df[df["sentiment"] == label]["textblob_polarity"],
        bins=40,
        alpha=0.5,
        label=label,
        color=color_map[label],
    )
axes[1].axvline(x=0.05, color="green", linestyle="--", linewidth=1)
axes[1].axvline(x=-0.05, color="red", linestyle="--", linewidth=1)
axes[1].set_title("TextBlob Polarity Distribution by Sentiment")
axes[1].set_xlabel("Polarity Score")
axes[1].set_ylabel("Count")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "13_polarity_distribution.png"), dpi=120)
plt.show()
plt.close()
print("Saved: 13_polarity_distribution.png")

print("\n" + "-" * 80)
print("Evaluation complete.")
