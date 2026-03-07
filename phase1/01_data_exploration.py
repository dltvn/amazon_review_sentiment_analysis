import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- load dataset ----
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data", "Software_5.json")

records = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

df = pd.DataFrame(records)

# ---- basic info ----
print("-" * 80)
print("Dataset Overview")
print(f"Total reviews : {len(df)}")
print(f"Columns       : {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ---- rating distribution ----
print("\n" + "-" * 80)
print("Rating Distribution")
rating_counts = df["overall"].value_counts().sort_index()
print(rating_counts.to_string())
print(f"\nAverage rating : {df['overall'].mean():.4f}")
print(f"Median rating  : {df['overall'].median():.1f}")
print(f"Std dev rating : {df['overall'].std():.4f}")

# ---- reviews per product ----
print("\n" + "-" * 80)
print("Reviews per Product (asin)")
reviews_per_product = df.groupby("asin").size()
print(f"Unique products           : {reviews_per_product.nunique()}")
print(f"Avg reviews / product     : {reviews_per_product.mean():.2f}")
print(f"Median reviews / product  : {reviews_per_product.median():.1f}")
print(f"Max reviews / product     : {reviews_per_product.max()}")
print(f"Min reviews / product     : {reviews_per_product.min()}")

# ---- reviews per user ----
print("\n" + "-" * 80)
print("Reviews per User (reviewerID)")
reviews_per_user = df.groupby("reviewerID").size()
print(f"Unique reviewers         : {len(reviews_per_user)}")
print(f"Avg reviews / user       : {reviews_per_user.mean():.2f}")
print(f"Median reviews / user    : {reviews_per_user.median():.1f}")
print(f"Max reviews / user       : {reviews_per_user.max()}")
print(f"Users with > 1 review    : {(reviews_per_user > 1).sum()}")

# ---- review length analysis ----
print("\n" + "-" * 80)
print("Review Length Analysis (reviewText, in words)")
df["review_len"] = df["reviewText"].fillna("").apply(lambda x: len(str(x).split()))
print(f"Average length  : {df['review_len'].mean():.2f} words")
print(f"Median length   : {df['review_len'].median():.1f} words")
print(f"Max length      : {df['review_len'].max()} words")
print(f"Min length      : {df['review_len'].min()} words")
print(f"Std dev         : {df['review_len'].std():.2f} words")

# IQR-based outlier thresholds
q1 = df["review_len"].quantile(0.25)
q3 = df["review_len"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = df[(df["review_len"] < lower_bound) | (df["review_len"] > upper_bound)]
print(f"\nIQR outlier bounds: [{lower_bound:.0f}, {upper_bound:.0f}] words")
print(f"Outlier reviews   : {len(outliers)} ({100 * len(outliers) / len(df):.1f}%)")

# ---- summary stats ----
print("\n" + "-" * 80)
print("Summary Analysis (reviewText length by rating)")
summary = df.groupby("overall")["review_len"].describe().round(2)
print(summary.to_string())

# ---- duplicate check ----
print("\n" + "-" * 80)
print("Duplicate Check")
# exclude dict/list columns that are not hashable
hashable_cols = [
    c
    for c in df.columns
    if df[c].apply(lambda x: not isinstance(x, (dict, list))).all()
]
exact_dupes = df[hashable_cols].duplicated().sum()
text_dupes = df["reviewText"].dropna().duplicated().sum()
print(f"Exact duplicate rows (hashable cols) : {exact_dupes}")
print(f"Duplicate reviewText                 : {text_dupes}")

# ---- verified purchase breakdown ----
print("\n" + "-" * 80)
print("Verified Purchase Breakdown")
if "verified" in df.columns:
    print(df["verified"].value_counts().to_string())

# ---- vote field presence ----
print("\n" + "-" * 80)
print("'vote' (helpful votes) field presence")
vote_present = df["vote"].notna().sum() if "vote" in df.columns else 0
print(
    f"Reviews with helpful vote count : {vote_present} ({100 * vote_present / len(df):.1f}%)"
)

# ============================================================
# Plots
# ============================================================
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# 1. Rating distribution bar chart
fig, ax = plt.subplots(figsize=(7, 4))
rating_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Distribution of Ratings")
ax.set_xlabel("Rating")
ax.set_ylabel("Number of Reviews")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "01_rating_distribution.png"), dpi=120)
plt.show()
plt.close()
print("\nSaved: 01_rating_distribution.png")

# 2. Reviews per product (top 20 products)
fig, ax = plt.subplots(figsize=(10, 4))
reviews_per_product.sort_values(ascending=False).head(20).plot(
    kind="bar", ax=ax, color="teal", edgecolor="white"
)
ax.set_title("Top 20 Products by Review Count")
ax.set_xlabel("Product ASIN")
ax.set_ylabel("Number of Reviews")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "02_reviews_per_product_top20.png"), dpi=120)
plt.show()
plt.close()
print("Saved: 02_reviews_per_product_top20.png")

# 3. Distribution of reviews per product (histogram)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(reviews_per_product.values, bins=50, color="steelblue", edgecolor="white")
ax.set_title("Distribution of Reviews per Product")
ax.set_xlabel("Number of Reviews")
ax.set_ylabel("Number of Products")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "03_reviews_per_product_hist.png"), dpi=120)
plt.show()
plt.close()
print("Saved: 03_reviews_per_product_hist.png")

# 4. Distribution of reviews per user (histogram)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(reviews_per_user.values, bins=50, color="darkorange", edgecolor="white")
ax.set_title("Distribution of Reviews per User")
ax.set_xlabel("Number of Reviews")
ax.set_ylabel("Number of Users")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "04_reviews_per_user_hist.png"), dpi=120)
plt.show()
plt.close()
print("Saved: 04_reviews_per_user_hist.png")

# 5. Review length distribution (box plot by rating)
fig, ax = plt.subplots(figsize=(8, 5))
df.boxplot(
    column="review_len",
    by="overall",
    ax=ax,
    flierprops=dict(marker=".", markersize=2, alpha=0.3),
)
ax.set_title("Review Length by Rating")
ax.set_xlabel("Rating")
ax.set_ylabel("Review Length (words)")
plt.suptitle("")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "05_review_length_by_rating.png"), dpi=120)
plt.show()
plt.close()
print("Saved: 05_review_length_by_rating.png")

# 6. Review length histogram
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(
    df["review_len"].clip(upper=500), bins=60, color="mediumpurple", edgecolor="white"
)
ax.set_title("Review Length Distribution (clipped at 500 words)")
ax.set_xlabel("Number of Words")
ax.set_ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "06_review_length_hist.png"), dpi=120)
plt.show()
plt.close()
print("Saved: 06_review_length_hist.png")

# 7. Verified purchase by rating
if "verified" in df.columns:
    verified_rating = df.groupby(["overall", "verified"]).size().unstack(fill_value=0)
    verified_rating.plot(kind="bar", figsize=(8, 4), edgecolor="white")
    plt.title("Verified vs Unverified Purchases by Rating")
    plt.xlabel("Rating")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "07_verified_by_rating.png"), dpi=120)
    plt.show()
    plt.close()
    print("Saved: 07_verified_by_rating.png")

# 8. Avg review length per rating
avg_len_by_rating = df.groupby("overall")["review_len"].mean()
fig, ax = plt.subplots(figsize=(7, 4))
avg_len_by_rating.plot(kind="bar", ax=ax, color="coral", edgecolor="white")
ax.set_title("Average Review Length by Rating")
ax.set_xlabel("Rating")
ax.set_ylabel("Average Words")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "08_avg_length_by_rating.png"), dpi=120)
plt.show()
plt.close()
print("Saved: 08_avg_length_by_rating.png")

print("\n" + "-" * 80)
print("Data exploration complete.")
