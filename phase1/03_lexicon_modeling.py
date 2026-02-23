import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# ---- load sample ----
script_dir = os.path.dirname(os.path.abspath(__file__))
sample_path = os.path.join(script_dir, "data", "sample_1000.csv")
df = pd.read_csv(sample_path)
print(f"Loaded {len(df)} reviews from sample")
print(f"Label distribution:\n{df['sentiment'].value_counts().to_string()}\n")

# ============================================================
# VADER
# ============================================================
# VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based
# lexicon tuned for social media and short review text. It uses a dictionary
# of words with pre-assigned valence scores and applies heuristic rules for
# punctuation, capitalization, degree modifiers, and negation. It natively
# produces a compound score in [-1, 1], making it straightforward to map to
# three classes.
#
# Pre-processing choice for VADER:
#   We feed `vader_text` (original casing, punctuation preserved, only
#   URLs/HTML stripped). VADER explicitly leverages CAPS and !!! as sentiment
#   boosters, so lowercasing would discard useful signal.
#
# Thresholds follow the guidance in the VADER paper (Hutto & Gilbert, 2014):
#   compound >= 0.05  -> positive
#   compound <= -0.05 -> negative
#   otherwise         -> neutral

vader = SentimentIntensityAnalyzer()


def predict_vader(text):
    scores = vader.polarity_scores(str(text))
    c = scores["compound"]
    if c >= 0.05:
        return "positive"
    elif c <= -0.05:
        return "negative"
    else:
        return "neutral"


df["vader_pred"] = df["vader_text"].apply(predict_vader)
print("VADER prediction distribution:")
print(df["vader_pred"].value_counts().to_string())

# ============================================================
# TextBlob
# ============================================================
# TextBlob uses the Pattern library's sentiment lexicon, which assigns
# polarity [-1, 1] and subjectivity [0, 1] to words and averages across
# tokens. It works best on clean, lowercased prose — exactly what
# `combined_text` provides.
#
# Pre-processing choice for TextBlob:
#   We feed `combined_text` (lowercased, URLs/HTML removed, whitespace
#   normalised). TextBlob does not use capitalisation or punctuation as
#   boosters, so clean text is preferred to reduce noise.
#
# Thresholds mirror VADER for consistency and fairness in comparison:
#   polarity >= 0.05  -> positive
#   polarity <= -0.05 -> negative
#   otherwise         -> neutral


def predict_textblob(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity >= 0.05:
        return "positive"
    elif polarity <= -0.05:
        return "negative"
    else:
        return "neutral"


df["textblob_pred"] = df["combined_text"].apply(predict_textblob)
print("\nTextBlob prediction distribution:")
print(df["textblob_pred"].value_counts().to_string())

# ---- save predictions ----
out_path = os.path.join(script_dir, "data", "predictions.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved predictions -> {out_path}")
