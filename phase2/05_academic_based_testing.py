import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from cornac.eval_methods import CrossValidation
from cornac.models import SVD
from cornac.metrics import RMSE, MAE
import cornac

# ---- paths ----
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)
data_path = os.path.join(script_dir, "..", "phase1", "data", "Software_5.json")

# ---- load raw dataset ----
records = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

df = pd.DataFrame(records)
df = df[["reviewerID", "asin", "overall", "reviewText", "summary"]].copy()
df["reviewText"] = df["reviewText"].fillna("")
df["summary"] = df["summary"].fillna("")

print(f"Loaded {len(df)} reviews")
print(f"Unique users: {df['reviewerID'].nunique()}")
print(f"Unique products: {df['asin'].nunique()}")
print(f"Rating distribution:\n{df['overall'].value_counts().sort_index()}")

# ---- text preprocessing (duplicated per-script convention) ----
def clean_text(text):
    text = emoji.demojize(str(text), delimiters=(" ", " "))
    text = re.sub(r":([a-z_]+):", r"\1", text)
    text = text.replace("_", " ")
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_for_vader(text):
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---- prepare text columns ----
df["combined_text"] = df.apply(
    lambda r: clean_text(r["summary"]) + " " + clean_text(r["reviewText"]), axis=1
)
df["vader_text"] = df.apply(
    lambda r: clean_for_vader(r["summary"]) + " " + clean_for_vader(r["reviewText"]), axis=1
)

# ---- VADER sentiment scoring ----
print("\n" + "=" * 80)
print("Scoring all reviews with VADER...")

analyzer = SentimentIntensityAnalyzer()
vader_scores = df["vader_text"].apply(lambda x: analyzer.polarity_scores(str(x)))
df["vader_compound"] = vader_scores.apply(lambda s: s["compound"])

print(f"VADER compound stats:\n{df['vader_compound'].describe()}")

# ---- ML (SVM) sentiment scoring ----
print("\nScoring all reviews with SVM model...")

tfidf = joblib.load(os.path.join(data_dir, "tfidf_vectorizer.pkl"))
svm_model = joblib.load(os.path.join(data_dir, "svm.pkl"))

X_tfidf = tfidf.transform(df["combined_text"])
df["ml_pred"] = svm_model.predict(X_tfidf)

# Map 3-class labels to numeric: negative=1, neutral=3, positive=5
ml_map = {"negative": 1.0, "neutral": 3.0, "positive": 5.0}
df["ml_score"] = df["ml_pred"].map(ml_map)

print(f"ML prediction distribution:\n{df['ml_pred'].value_counts()}")

# ---- rating enhancement ----
print("\n" + "=" * 80)
print("Computing enhanced ratings...")

ALPHA = 0.7   # weight for original rating in linear blend
BETA = 0.5    # scaling factor for additive adjustment

# Scale VADER compound [-1, 1] to rating scale [1, 5]
df["vader_scaled"] = 1.0 + (df["vader_compound"] + 1.0) * 2.0  # maps [-1,1] -> [1,5]

# Linear blend: alpha * overall + (1 - alpha) * sentiment_scaled
df["rating_vader_blend"] = ALPHA * df["overall"] + (1 - ALPHA) * df["vader_scaled"]
df["rating_ml_blend"] = ALPHA * df["overall"] + (1 - ALPHA) * df["ml_score"]

# Additive: overall + beta * sentiment_signal, clamped to [1, 5]
df["rating_vader_additive"] = (df["overall"] + BETA * df["vader_compound"]).clip(1.0, 5.0)
df["rating_ml_additive"] = (df["overall"] + BETA * (df["ml_score"] - 3.0) / 2.0).clip(1.0, 5.0)
# Note: (ml_score - 3.0) / 2.0 maps {1,3,5} -> {-1,0,1} for additive adjustment

# ---- spot-check ----
print("\nSpot-check (10 random reviews):")
check_cols = ["overall", "vader_compound", "ml_pred", "ml_score",
              "rating_vader_blend", "rating_vader_additive",
              "rating_ml_blend", "rating_ml_additive"]
print(df[check_cols].sample(10, random_state=42).to_string())

# Verify no NaN
for col in ["rating_vader_blend", "rating_vader_additive", "rating_ml_blend", "rating_ml_additive"]:
    assert df[col].notna().all(), f"NaN found in {col}"
print("\nAll enhanced rating columns populated (no NaN).")

# ---- SVD cross-validation for all 5 rating variants ----
print("\n" + "=" * 80)
print("Running SVD cross-validation for all 5 rating variants...")

rating_variants = {
    "Baseline (Raw Ratings)":       "overall",
    "VADER + Linear Blend":         "rating_vader_blend",
    "VADER + Additive":             "rating_vader_additive",
    "ML (SVM) + Linear Blend":      "rating_ml_blend",
    "ML (SVM) + Additive":          "rating_ml_additive",
}

results = []

for name, col in rating_variants.items():
    print(f"\n--- {name} ---")
    tuples = list(
        zip(df["reviewerID"].astype(str), df["asin"].astype(str), df[col].astype(float))
    )
    cv_method = CrossValidation(tuples, n_folds=5, seed=42, verbose=False)
    svd = SVD(k=100, max_iter=20, seed=42, verbose=False)
    exp = cornac.Experiment(
        eval_method=cv_method, models=[svd], metrics=[RMSE(), MAE()], verbose=False
    )
    exp.run()
    r = exp.result[0]
    row = {
        "Model": name,
        "RMSE_mean": float(r.metric_mean["RMSE"]),
        "RMSE_std": float(r.metric_std["RMSE"]),
        "MAE_mean": float(r.metric_mean["MAE"]),
        "MAE_std": float(r.metric_std["MAE"]),
    }
    results.append(row)
    print(f"  RMSE: {row['RMSE_mean']:.4f} (+/- {row['RMSE_std']:.4f})")
    print(f"  MAE:  {row['MAE_mean']:.4f} (+/- {row['MAE_std']:.4f})")

# ---- comparison table ----
comparison = pd.DataFrame(results)
comparison = comparison.sort_values("RMSE_mean")

print("\n" + "=" * 80)
print("RECOMMENDER SYSTEM COMPARISON")
print("=" * 80)
print(comparison.to_string(index=False))

baseline_rmse = comparison.loc[comparison["Model"] == "Baseline (Raw Ratings)", "RMSE_mean"].values[0]
print(f"\nBaseline RMSE: {baseline_rmse:.4f}")
print("\nImprovement over baseline:")
for _, row in comparison.iterrows():
    if row["Model"] != "Baseline (Raw Ratings)":
        diff = baseline_rmse - row["RMSE_mean"]
        pct = (diff / baseline_rmse) * 100
        direction = "better" if diff > 0 else "worse"
        print(f"  {row['Model']}: {abs(diff):.4f} ({abs(pct):.2f}% {direction})")

comparison.to_csv(os.path.join(data_dir, "recommender_comparison.csv"), index=False)
print(f"\nSaved: recommender_comparison.csv")

# ---- plot 19: RMSE comparison ----
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(comparison))
bars = ax.bar(x, comparison["RMSE_mean"], yerr=comparison["RMSE_std"],
              capsize=5, color=["#95a5a6", "#3498db", "#2ecc71", "#e74c3c", "#9b59b6"])
ax.set_xticks(x)
ax.set_xticklabels(comparison["Model"], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("RMSE")
ax.set_title("Recommender System RMSE Comparison (5-Fold CV)")
ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
plt.tight_layout()
fig.savefig(os.path.join(plots_dir, "19_recommender_rmse_comparison.png"), dpi=120)
plt.close(fig)
print("Saved: 19_recommender_rmse_comparison.png")

# ---- plot 20: MAE comparison ----
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(x, comparison["MAE_mean"], yerr=comparison["MAE_std"],
              capsize=5, color=["#95a5a6", "#3498db", "#2ecc71", "#e74c3c", "#9b59b6"])
ax.set_xticks(x)
ax.set_xticklabels(comparison["Model"], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("MAE")
ax.set_title("Recommender System MAE Comparison (5-Fold CV)")
ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
plt.tight_layout()
fig.savefig(os.path.join(plots_dir, "20_recommender_mae_comparison.png"), dpi=120)
plt.close(fig)
print("Saved: 20_recommender_mae_comparison.png")

# ---- plot 21: formula comparison ----
fig, ax = plt.subplots(figsize=(8, 5))
signals = ["VADER", "ML (SVM)"]
blend_rmse = [
    comparison.loc[comparison["Model"] == "VADER + Linear Blend", "RMSE_mean"].values[0],
    comparison.loc[comparison["Model"] == "ML (SVM) + Linear Blend", "RMSE_mean"].values[0],
]
additive_rmse = [
    comparison.loc[comparison["Model"] == "VADER + Additive", "RMSE_mean"].values[0],
    comparison.loc[comparison["Model"] == "ML (SVM) + Additive", "RMSE_mean"].values[0],
]
x_pos = np.arange(len(signals))
width = 0.35
b1 = ax.bar(x_pos - width/2, blend_rmse, width, label="Linear Blend", color="#3498db")
b2 = ax.bar(x_pos + width/2, additive_rmse, width, label="Additive", color="#e74c3c")
ax.set_xticks(x_pos)
ax.set_xticklabels(signals)
ax.set_ylabel("RMSE")
ax.set_title("Rating Enhancement Formula Comparison")
ax.legend()
ax.bar_label(b1, fmt="%.4f", fontsize=8, padding=3)
ax.bar_label(b2, fmt="%.4f", fontsize=8, padding=3)
plt.tight_layout()
fig.savefig(os.path.join(plots_dir, "21_formula_comparison.png"), dpi=120)
plt.close(fig)
print("Saved: 21_formula_comparison.png")

# ---- plot 22: sentiment signal comparison ----
fig, ax = plt.subplots(figsize=(8, 5))
formulas = ["Linear Blend", "Additive"]
vader_rmse = [
    comparison.loc[comparison["Model"] == "VADER + Linear Blend", "RMSE_mean"].values[0],
    comparison.loc[comparison["Model"] == "VADER + Additive", "RMSE_mean"].values[0],
]
ml_rmse = [
    comparison.loc[comparison["Model"] == "ML (SVM) + Linear Blend", "RMSE_mean"].values[0],
    comparison.loc[comparison["Model"] == "ML (SVM) + Additive", "RMSE_mean"].values[0],
]
x_pos = np.arange(len(formulas))
width = 0.35
b1 = ax.bar(x_pos - width/2, vader_rmse, width, label="VADER", color="#2ecc71")
b2 = ax.bar(x_pos + width/2, ml_rmse, width, label="ML (SVM)", color="#9b59b6")
ax.set_xticks(x_pos)
ax.set_xticklabels(formulas)
ax.set_ylabel("RMSE")
ax.set_title("Sentiment Signal Comparison")
ax.legend()
ax.bar_label(b1, fmt="%.4f", fontsize=8, padding=3)
ax.bar_label(b2, fmt="%.4f", fontsize=8, padding=3)
plt.tight_layout()
fig.savefig(os.path.join(plots_dir, "22_signal_comparison.png"), dpi=120)
plt.close(fig)
print("Saved: 22_signal_comparison.png")

# ---- plot 23: rating distributions ----
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
rating_cols = [
    ("rating_vader_blend", "VADER + Linear Blend"),
    ("rating_vader_additive", "VADER + Additive"),
    ("rating_ml_blend", "ML (SVM) + Linear Blend"),
    ("rating_ml_additive", "ML (SVM) + Additive"),
]
for ax, (col, title) in zip(axes.flat, rating_cols):
    ax.hist(df["overall"], bins=50, alpha=0.5, label="Raw Rating", color="#95a5a6")
    ax.hist(df[col], bins=50, alpha=0.5, label="Enhanced", color="#3498db")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
plt.suptitle("Rating Distributions: Raw vs. Enhanced", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(plots_dir, "23_rating_distributions.png"), dpi=120)
plt.close(fig)
print("Saved: 23_rating_distributions.png")
