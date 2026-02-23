import os
import re
import json
import random
import pandas as pd

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
print(f"Loaded {len(df)} reviews")

# ---- step 1: column selection ----
# reviewText: the primary long-form opinion text written by the reviewer.
#   This is the richest source of sentiment signal — reviewers express likes,
#   dislikes, recommendations, and emotions in detail here.
# summary: a short headline written by the reviewer that captures the
#   overall sentiment concisely. Combining it with reviewText gives the
#   lexicon models both explicit sentiment keywords (summary) and context
#   (reviewText), improving coverage for short or missing review bodies.
# overall: the numeric star rating — used exclusively for labeling, not as
#   a model input feature.
df = df[["overall", "reviewText", "summary"]].copy()
print(f"Columns selected: overall, reviewText, summary")

# ---- step 2: drop rows with missing text ----
# A review with no reviewText and no summary provides no signal.
before = len(df)
df = df.dropna(subset=["reviewText"])
print(f"Dropped {before - len(df)} rows with missing reviewText ({len(df)} remain)")

# ---- step 3: drop duplicate reviewText ----
# Exact duplicate review bodies skew model evaluation by repeating identical
# signal. Keeping only the first occurrence removes fabricated/copy-paste reviews.
before = len(df)
df = df.drop_duplicates(subset=["reviewText"])
print(f"Dropped {before - len(df)} duplicate reviewText rows ({len(df)} remain)")


# ---- step 4: sentiment labeling ----
# Ratings 4–5 → Positive, 3 → Neutral, 1–2 → Negative (per project spec)
def label_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"


df["sentiment"] = df["overall"].apply(label_sentiment)
print("\nLabel distribution after deduplication:")
print(df["sentiment"].value_counts().to_string())

# ---- step 5: outlier removal ----
# Compute word count on the raw reviewText
df["review_len"] = df["reviewText"].apply(lambda x: len(str(x).split()))

# Remove empty reviews (0 words) — no signal
before = len(df)
df = df[df["review_len"] > 0]
print(f"\nDropped {before - len(df)} empty reviews (0 words)")

# IQR-based upper outlier removal:
# Reviews exceeding Q3 + 1.5*IQR (> 503 words, as found in exploration) are
# disproportionately long and can dominate lexicon score averaging. Removing
# them keeps the sample representative of typical user behaviour.
q1 = df["review_len"].quantile(0.25)
q3 = df["review_len"].quantile(0.75)
upper_bound = q3 + 1.5 * (q3 - q1)
before = len(df)
df = df[df["review_len"] <= upper_bound]
print(
    f"Dropped {before - len(df)} reviews above upper IQR bound ({upper_bound:.0f} words)"
)
print(f"Reviews remaining: {len(df)}")
print("\nLabel distribution after outlier removal:")
print(df["sentiment"].value_counts().to_string())

# ---- step 6: text cleaning ----
# The combined_text field merges summary + reviewText so both headline
# sentiment and body detail are available to the lexicon models.
df["summary"] = df["summary"].fillna("")


def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove HTML entities / tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode()
    # Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


# VADER is designed to work on social / review text and handles punctuation
# (e.g. !!!, ALL CAPS) as emphasis boosters, so we preserve the original
# casing and punctuation for the VADER input column.
# TextBlob also benefits from natural sentence structure, but we still clean
# URLs, HTML, and normalize whitespace for both models.
df["clean_text"] = df["reviewText"].apply(clean_text)

# Combine summary + reviewText for model input (after cleaning each)
df["clean_summary"] = df["summary"].apply(clean_text)
df["combined_text"] = df.apply(
    lambda r: (r["clean_summary"] + " " + r["clean_text"]).strip(), axis=1
)


# Also keep a version with original casing for VADER (only URL/HTML/whitespace cleaned)
def clean_for_vader(text):
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["vader_text"] = df.apply(
    lambda r: (
        clean_for_vader(str(r["summary"])) + " " + clean_for_vader(str(r["reviewText"]))
    ).strip(),
    axis=1,
)

print("\nSample cleaned review:")
print(df["combined_text"].iloc[0][:200])

# ---- step 7: random sample of 1000 reviews ----
# Stratified sampling preserves the class distribution of the cleaned dataset
# so evaluation metrics fairly reflect real-world performance.
TARGET = 1000
fracs = df["sentiment"].value_counts(normalize=True)
parts = []
allocated = 0
labels = fracs.index.tolist()
for i, label in enumerate(labels):
    if i < len(labels) - 1:
        n = round(fracs[label] * TARGET)
    else:
        # last group gets the remainder to guarantee exactly TARGET total
        n = TARGET - allocated
    group = df[df["sentiment"] == label]
    parts.append(group.sample(n=min(n, len(group)), random_state=42))
    allocated += min(n, len(group))
sample_1000 = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nStratified 1000-review sample:")
print(sample_1000["sentiment"].value_counts().to_string())

# ---- save outputs ----
out_path = os.path.join(script_dir, "data", "preprocessed.csv")
sample_path = os.path.join(script_dir, "data", "sample_1000.csv")

df.to_csv(out_path, index=False)
sample_1000.to_csv(sample_path, index=False)

print(f"\nSaved full preprocessed data  -> {out_path}")
print(f"Saved 1000-review sample      -> {sample_path}")
