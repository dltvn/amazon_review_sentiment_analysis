import os
import re
import json
import random
import pandas as pd
import emoji

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
# ---- emoji audit (raw data) ----
raw_emojis = df["reviewText"].dropna().apply(lambda x: emoji.emoji_count(str(x))).sum()
print(f"Total emojis in raw reviewText: {raw_emojis}")

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
# Exact duplicate review bodies skew model evaluation by repeating identical signal. Keeping only the first occurrence removes fabricated/copy-paste reviews.
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
# IQR-based upper outlier removal: Reviews exceeding Q3 + 1.5*IQR (> 503 words, as found in exploration) are disproportionately long and can dominate lexicon score averaging. Removing them keeps the sample representative of typical user behaviour.
df["review_len"] = df["reviewText"].apply(lambda x: len(str(x).split()))

before = len(df)
df = df[df["review_len"] > 0]
print(f"\nDropped {before - len(df)} empty reviews (0 words)")

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
df["summary"] = df["summary"].fillna("")


def clean_text(text):
    """
    Full cleaning pipeline for TextBlob (combined_text).
    - Demojizes emojis to plain words (e.g. 😊 -> 'smiling face') so TextBlob
      can extract sentiment signal from them rather than silently losing them.
    - Lowercases, removes URLs, HTML, non-ASCII, collapses whitespace.
    """
    # Convert emojis to text descriptions before stripping non-ASCII
    text = emoji.demojize(str(text), delimiters=(" ", " "))
    # Remove colon-style demoji artifacts just in case (e.g. :smiling_face:)
    text = re.sub(r":([a-z_]+):", r"\1", text)
    # Replace underscores from demoji token names with spaces
    text = text.replace("_", " ")
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove remaining non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_for_vader(text):
    """
    Minimal cleaning pipeline for VADER (vader_text).
    - Preserves original casing and punctuation (VADER uses ALL CAPS and !!!
      as intensity boosters).
    - Preserves raw UTF-8 emojis — VADER has a native emoji valence lexicon
      so keeping 😡 intact gives a stronger signal than demojizing to words.
    - Only strips URLs, HTML tags, and collapses whitespace.
    NOTE: The ASCII-encoding strip used in other functions is intentionally
    OMITTED here so emojis pass through unchanged.
    """
    text = str(text)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace — no lowercasing, no ASCII strip
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_text"] = df["reviewText"].apply(clean_text)
df["clean_summary"] = df["summary"].apply(clean_text)
df["combined_text"] = df.apply(
    lambda r: (r["clean_summary"] + " " + r["clean_text"]).strip(), axis=1
)

df["vader_text"] = df.apply(
    lambda r: (
        clean_for_vader(str(r["summary"])) + " " + clean_for_vader(str(r["reviewText"]))
    ).strip(),
    axis=1,
)

print("\nSample cleaned review (combined_text):")
print(df["combined_text"].iloc[0][:200])
print("\nSample cleaned review (vader_text):")
print(df["vader_text"].iloc[0][:200])

# ---- step 7: random stratified sample of 1000 reviews ----
# Stratified sampling preserves the class distribution of the cleaned dataset so evaluation metrics fairly reflect real-world performance.
TARGET = 1000
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
sample_1000 = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nStratified 1000-review sample:")
print(sample_1000["sentiment"].value_counts().to_string())

# ============================================================
# VALIDATION BLOCK
# ============================================================
print("\n" + "=" * 80)
print("VALIDATION BLOCK")

# 1. Sample 5 rows side by side
print("\n[1] Sample 5 rows (reviewText | vader_text | combined_text | sentiment):")
sample_display = df[["reviewText", "vader_text", "combined_text", "sentiment"]].head(5)
for i, row in sample_display.iterrows():
    print(f"\n  Row {i}")
    print(f"  reviewText    : {str(row['reviewText'])[:80]}")
    print(f"  vader_text    : {str(row['vader_text'])[:80]}")
    print(f"  combined_text : {str(row['combined_text'])[:80]}")
    print(f"  sentiment     : {row['sentiment']}")

# 2. Sentiment column integrity
print("\n[2] Sentiment column null check:")
print(f"  Nulls in sentiment: {df['sentiment'].isnull().sum()}")
print(f"  Unique values     : {df['sentiment'].unique().tolist()}")

# 3. Cross-tab: overall rating vs sentiment label (catches float edge cases)
print("\n[3] Cross-tab overall vs sentiment (should be clean 1-to-1 mapping):")
print(pd.crosstab(df["overall"], df["sentiment"]).to_string())

# 4. Casing check — vader_text should preserve original casing
caps_in_vader = (
    df["vader_text"].apply(lambda x: bool(re.search(r"[A-Z]", str(x)))).sum()
)
caps_in_combined = (
    df["combined_text"].apply(lambda x: bool(re.search(r"[A-Z]", str(x)))).sum()
)
print(f"\n[4] Casing check:")
print(f"  Rows with uppercase chars in vader_text    : {caps_in_vader} (should be > 0)")
print(
    f"  Rows with uppercase chars in combined_text : {caps_in_combined} (should be 0)"
)

# 5. Leftover HTML / URL check
html_in_vader = df["vader_text"].str.contains(r"<[^>]+>|http", regex=True).sum()
html_in_combined = df["combined_text"].str.contains(r"<[^>]+>|http", regex=True).sum()
print(f"\n[5] Leftover HTML/URL check:")
print(f"  vader_text    : {html_in_vader} rows still contain HTML or URLs")
print(f"  combined_text : {html_in_combined} rows still contain HTML or URLs")

# 6. Length stats post-emoji fix
print(f"\n[6] Length stats (words) post-emoji fix:")
df["vader_len"] = df["vader_text"].apply(lambda x: len(str(x).split()))
df["combined_len"] = df["combined_text"].apply(lambda x: len(str(x).split()))
for col in ["vader_len", "combined_len"]:
    print(
        f"  {col}: min={df[col].min()}, max={df[col].max()}, avg={df[col].mean():.2f}"
    )

print("=" * 80)

# ---- save outputs ----
out_path = os.path.join(script_dir, "data", "preprocessed.csv")
sample_path = os.path.join(script_dir, "data", "sample_1000.csv")

df.to_csv(out_path, index=False)
sample_1000.to_csv(sample_path, index=False)

print(f"\nSaved full preprocessed data  -> {out_path}")
print(f"Saved 1000-review sample      -> {sample_path}")
