import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ---- load phase 1 preprocessed data ----
script_dir = os.path.dirname(os.path.abspath(__file__))
phase1_data_path = os.path.join(script_dir, "..", "phase1", "data", "preprocessed.csv")
df = pd.read_csv(phase1_data_path)

print(f"Loaded {len(df)} reviews from Phase 1 preprocessed data")
print(f"Columns: {df.columns.tolist()}")

# ---- create stratified 2000-review sample ----
# Stratified sampling preserves the sentiment class distribution from the full dataset,
# ensuring the subset is representative for fair model training and evaluation.
TARGET = 2000
fracs = df["sentiment"].value_counts(normalize=True)
parts = []
allocated = 0
labels = fracs.index.tolist()

for i, label in enumerate(labels):
    if i < len(labels) - 1:
        n = round(fracs[label] * TARGET)
    else:
        n = TARGET - allocated
    group = df[df["sentiment"] == label]
    parts.append(group.sample(n=min(n, len(group)), random_state=42))
    allocated += min(n, len(group))

sample = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nStratified {len(sample)}-review sample created")

# ---- data exploration on subset ----
print("\n" + "-" * 80)
print("Data Exploration - 2000 Review Subset")

# 1. Basic info
print("\n" + "-" * 80)
print("Basic Information")
print(f"Total reviews: {len(sample)}")
print(f"Columns: {sample.columns.tolist()}")
print(f"\nData types:")
print(sample.dtypes.to_string())
print(f"\nMissing values:")
print(sample.isnull().sum().to_string())

# 2. Sentiment distribution
print("\n" + "-" * 80)
print("Sentiment Distribution")
sentiment_counts = sample["sentiment"].value_counts()
sentiment_pcts = sample["sentiment"].value_counts(normalize=True) * 100
print(f"\nCounts:")
print(sentiment_counts.to_string())
print(f"\nPercentages:")
for label in sentiment_counts.index:
    print(f"  {label}: {sentiment_pcts[label]:.2f}%")

# 3. Rating distribution
print("\n" + "-" * 80)
print("Rating Distribution")
rating_counts = sample["overall"].value_counts().sort_index()
print(rating_counts.to_string())

# 4. Review length statistics
print("\n" + "-" * 80)
print("Review Length Statistics (words)")
sample["word_count"] = sample["reviewText"].apply(lambda x: len(str(x).split()))
print(f"Mean   : {sample['word_count'].mean():.2f}")
print(f"Median : {sample['word_count'].median():.2f}")
print(f"Std Dev: {sample['word_count'].std():.2f}")
print(f"Min    : {sample['word_count'].min()}")
print(f"Max    : {sample['word_count'].max()}")

# Length by sentiment
print("\nMean word count by sentiment:")
for label in ["positive", "neutral", "negative"]:
    mean_len = sample[sample["sentiment"] == label]["word_count"].mean()
    print(f"  {label}: {mean_len:.2f}")

# 5. Top words analysis (basic frequency)
print("\n" + "-" * 80)
print("Top 20 Words (from clean_text)")
all_words = " ".join(sample["clean_text"].dropna().tolist()).split()
word_freq = Counter(all_words)
print("Word frequencies:")
for word, count in word_freq.most_common(20):
    print(f"  {word}: {count}")

# ---- plots ----
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Plot 1: Sentiment distribution
fig, ax = plt.subplots(figsize=(8, 5))
colors = {"positive": "steelblue", "neutral": "gold", "negative": "tomato"}
sentiment_order = ["positive", "neutral", "negative"]
bars = ax.bar(
    sentiment_order,
    [sentiment_counts[s] for s in sentiment_order],
    color=[colors[s] for s in sentiment_order],
    edgecolor="black",
)
ax.set_title("Sentiment Distribution (2000 Review Subset)")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Count")
for bar, label in zip(bars, sentiment_order):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{int(height)}\n({sentiment_pcts[label]:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=10,
    )
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "01_sentiment_distribution.png"), dpi=120)
plt.close()
print("\nSaved: 01_sentiment_distribution.png")

# Plot 2: Rating distribution
fig, ax = plt.subplots(figsize=(8, 5))
rating_order = [1.0, 2.0, 3.0, 4.0, 5.0]
ax.bar(
    [str(int(r)) for r in rating_order],
    [rating_counts.get(r, 0) for r in rating_order],
    color="steelblue",
    edgecolor="black",
)
ax.set_title("Rating Distribution (2000 Review Subset)")
ax.set_xlabel("Rating")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "02_rating_distribution.png"), dpi=120)
plt.close()
print("Saved: 02_rating_distribution.png")

# Plot 3: Review length distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(sample["word_count"], bins=50, color="steelblue", edgecolor="black", alpha=0.7)
ax.axvline(
    x=sample["word_count"].mean(),
    color="red",
    linestyle="--",
    label=f"Mean: {sample['word_count'].mean():.1f}",
)
ax.axvline(
    x=sample["word_count"].median(),
    color="green",
    linestyle="--",
    label=f"Median: {sample['word_count'].median():.1f}",
)
ax.set_title("Review Length Distribution (2000 Review Subset)")
ax.set_xlabel("Word Count")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "03_review_length_distribution.png"), dpi=120)
plt.close()
print("Saved: 03_review_length_distribution.png")

# Plot 4: Review length by sentiment (boxplot)
fig, ax = plt.subplots(figsize=(8, 5))
box_data = [sample[sample["sentiment"] == s]["word_count"] for s in sentiment_order]
bp = ax.boxplot(box_data, labels=sentiment_order, patch_artist=True)
for patch, color in zip(bp["boxes"], [colors[s] for s in sentiment_order]):
    patch.set_facecolor(color)
ax.set_title("Review Length by Sentiment (2000 Review Subset)")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Word Count")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "04_length_by_sentiment.png"), dpi=120)
plt.close()
print("Saved: 04_length_by_sentiment.png")

# ---- save sample ----
sample_path = os.path.join(script_dir, "data", "sample_2000.csv")
sample.to_csv(sample_path, index=False)
print(f"\nSaved 2000-review sample -> {sample_path}")

print("\n" + "-" * 80)
print("Data exploration complete.")
