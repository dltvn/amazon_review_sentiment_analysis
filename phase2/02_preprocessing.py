import os
import re
import pandas as pd
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

# ---- load 2000-review sample ----
script_dir = os.path.dirname(os.path.abspath(__file__))
sample_path = os.path.join(script_dir, "data", "sample_2000.csv")
df = pd.read_csv(sample_path)

print(f"Loaded {len(df)} reviews from sample_2000.csv")
print(f"Sentiment distribution:")
print(df["sentiment"].value_counts().to_string())

# ---- text cleaning for ML models ----
# The following preprocessing steps prepare text for TF-IDF vectorization:
# 1. Demojize emojis: Convert emoji characters to text descriptions so sentiment
#    signal from emojis is captured as words (e.g., "thumbs_up" instead of losing the emoji)
# 2. Lowercase: Reduces vocabulary size by treating "Good" and "good" as the same token,
#    which is essential for sparse models to generalize better
# 3. Remove URLs: URLs provide no sentiment signal and add noise to the vocabulary
# 4. Remove HTML tags: Leftover HTML markup from web scraping is irrelevant to sentiment
# 5. Remove non-ASCII: Removes special characters that don't contribute to meaning
# 6. Collapse whitespace: Ensures clean tokenization


def clean_text_ml(text):
    """
    Text cleaning pipeline optimized for ML/TF-IDF models.
    """
    # Convert emojis to text descriptions
    text = emoji.demojize(str(text), delimiters=(" ", " "))
    # Remove colon-style demoji artifacts (e.g., :smiling_face:)
    text = re.sub(r":([a-z_]+):", r"\1", text)
    # Replace underscores from demoji token names with spaces
    text = text.replace("_", " ")
    # Lowercase - reduces vocabulary, improves generalization
    text = text.lower()
    # Remove URLs - no sentiment signal
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove HTML tags - noise from web scraping
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("\n" + "-" * 80)
print("Preprocessing")

# Apply cleaning to reviewText
df["clean_text"] = df["reviewText"].apply(clean_text_ml)

# Combine summary and reviewText for richer signal
# Summary often contains key sentiment words (e.g., "Great product!", "Waste of money")
df["summary"] = df["summary"].fillna("")
df["clean_summary"] = df["summary"].apply(clean_text_ml)
df["text_combined"] = df.apply(
    lambda r: (r["clean_summary"] + " " + r["clean_text"]).strip(), axis=1
)

print(f"Applied text cleaning to {len(df)} reviews")
print(f"\nSample cleaned text:")
print(f"  Original : {df['reviewText'].iloc[0][:100]}...")
print(f"  Cleaned  : {df['clean_text'].iloc[0][:100]}...")

# ---- train/test split (70/30 stratified) ----
# Stratified split ensures both train and test sets maintain the same class distribution,
# which is critical for fair evaluation on imbalanced data.
print("\n" + "-" * 80)
print("Train/Test Split")

X = df["text_combined"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

print(f"Total samples : {len(df)}")
print(f"Training set  : {len(X_train)} ({len(X_train) / len(df) * 100:.1f}%)")
print(f"Test set      : {len(X_test)} ({len(X_test) / len(df) * 100:.1f}%)")

print(f"\nTraining set distribution:")
print(y_train.value_counts().to_string())
print(f"\nTest set distribution:")
print(y_test.value_counts().to_string())

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
print(f"\nTop 20 features by index:")
vocab_items = sorted(tfidf.vocabulary_.items(), key=lambda x: x[1])[:20]
for word, idx in vocab_items:
    print(f"  {idx}: {word}")

# ---- save outputs ----
print("\n" + "-" * 80)
print("Saving Outputs")

data_dir = os.path.join(script_dir, "data")

# Save train/test splits with indices for reproducibility
train_df = pd.DataFrame({"text": X_train.values, "sentiment": y_train.values})
test_df = pd.DataFrame({"text": X_test.values, "sentiment": y_test.values})

train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)

# Save TF-IDF vectorizer and matrices
joblib.dump(tfidf, os.path.join(data_dir, "tfidf_vectorizer.pkl"))
joblib.dump(X_train_tfidf, os.path.join(data_dir, "X_train_tfidf.pkl"))
joblib.dump(X_test_tfidf, os.path.join(data_dir, "X_test_tfidf.pkl"))
joblib.dump(y_train.values, os.path.join(data_dir, "y_train.pkl"))
joblib.dump(y_test.values, os.path.join(data_dir, "y_test.pkl"))

print(f"Saved train.csv        -> {len(train_df)} rows")
print(f"Saved test.csv         -> {len(test_df)} rows")
print(f"Saved tfidf_vectorizer.pkl")
print(f"Saved X_train_tfidf.pkl, X_test_tfidf.pkl")
print(f"Saved y_train.pkl, y_test.pkl")

print("\n" + "-" * 80)
print("Preprocessing complete.")
