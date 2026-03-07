import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# ============================================================
# RULE BASE CONSTANTS
# Thresholds follow Hutto & Gilbert (2014) VADER paper.
# Extracted as named constants for transparency and easy tuning.
# ============================================================
VADER_POS_THRESH = 0.05  # compound >= this  -> positive
VADER_NEG_THRESH = -0.05  # compound <= this  -> negative
# between VADER_NEG_THRESH and VADER_POS_THRESH   -> neutral

TEXTBLOB_POS_THRESH = 0.05  # polarity >= this  -> positive
TEXTBLOB_NEG_THRESH = -0.05  # polarity <= this  -> negative
# between TEXTBLOB_NEG_THRESH and TEXTBLOB_POS_THRESH -> neutral

# ---- load sample ----
script_dir = os.path.dirname(os.path.abspath(__file__))
sample_path = os.path.join(script_dir, "data", "sample_1000.csv")
df = pd.read_csv(sample_path)
# fill NaNs with empty strings to prevent CSV parsing errors on empty reviews
df["vader_text"] = df["vader_text"].fillna("")
df["combined_text"] = df["combined_text"].fillna("")

print(f"Loaded {len(df)} reviews from sample")
print(f"Label distribution:\n{df['sentiment'].value_counts().to_string()}\n")

# ---- print rule base to terminal  ----
print("------------------------------------------------")
print("SENTIMENT ANALYZER RULE BASE")
print("------------------------------------------------")
print("VADER Thresholds (compound score):")
print(f"  Positive : compound >= {VADER_POS_THRESH}")
print(f"  Neutral  : {VADER_NEG_THRESH} < compound < {VADER_POS_THRESH}")
print(f"  Negative : compound <= {VADER_NEG_THRESH}")
print()
print("TextBlob Thresholds (polarity score):")
print(f"  Positive : polarity >= {TEXTBLOB_POS_THRESH}")
print(f"  Neutral  : {TEXTBLOB_NEG_THRESH} < polarity < {TEXTBLOB_POS_THRESH}")
print(f"  Negative : polarity <= {TEXTBLOB_NEG_THRESH}")
print("=" * 60 + "\n")

# ============================================================
# VADER
# ============================================================
# VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based
# lexicon tuned for social media and short review text. It uses a dictionary
# of words with pre-assigned valence scores and applies heuristic rules for
# punctuation, capitalization, degree modifiers, and negation. Raw UTF-8
# emojis are preserved in vader_text so VADER can apply its native emoji
# valence lexicon directly.
#
# All four sub-scores are saved:
#   vader_pos / vader_neg / vader_neu: proportion of text that is
#     positive / negative / neutral
#   vader_compound: normalized aggregate score in [-1, 1]
# Saving the sub-scores allows identification of truly mixed reviews
# (high pos AND high neg simultaneously) in the evaluation script.

# Pre-processing choice for VADER:
#   We feed `vader_text` (original casing, punctuation preserved, only
#   URLs/HTML stripped). VADER explicitly leverages CAPS and !!! as sentiment
#   boosters, so lowercasing would discard useful signal.

vader = SentimentIntensityAnalyzer()


def score_vader(text):
    return vader.polarity_scores(str(text))


def predict_vader(compound):
    if compound >= VADER_POS_THRESH:
        return "positive"
    elif compound <= VADER_NEG_THRESH:
        return "negative"
    else:
        return "neutral"


vader_scores = df["vader_text"].apply(score_vader)
df["vader_pos"] = vader_scores.apply(lambda s: s["pos"])
df["vader_neg"] = vader_scores.apply(lambda s: s["neg"])
df["vader_neu"] = vader_scores.apply(lambda s: s["neu"])
df["vader_compound"] = vader_scores.apply(lambda s: s["compound"])
df["vader_pred"] = df["vader_compound"].apply(predict_vader)

print("VADER prediction distribution:")
print(df["vader_pred"].value_counts().to_string())

# ============================================================
# TextBlob
# ============================================================
# TextBlob uses the Pattern library's sentiment lexicon, which assigns
# polarity [-1, 1] and subjectivity [0, 1] to words and averages across
# tokens. Emojis are demojized to plain words in combined_text so TextBlob
# can extract sentiment signal from them rather than silently ignoring them.
# Thresholds mirror VADER for a fair apples-to-apples comparison.

# Pre-processing choice for TextBlob:
#   We feed `combined_text` (lowercased, URLs/HTML removed, whitespace
#   normalised). TextBlob does not use capitalisation or punctuation as
#   boosters, so clean text is preferred to reduce noise.


def predict_textblob(text):
    blob = TextBlob(str(text))
    return blob.sentiment.polarity


df["textblob_polarity"] = df["combined_text"].apply(predict_textblob)
df["textblob_pred"] = df["textblob_polarity"].apply(
    lambda p: (
        "positive"
        if p >= TEXTBLOB_POS_THRESH
        else ("negative" if p <= TEXTBLOB_NEG_THRESH else "neutral")
    )
)

print("\nTextBlob prediction distribution:")
print(df["textblob_pred"].value_counts().to_string())

# ---- save predictions ----
out_path = os.path.join(script_dir, "data", "predictions.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved predictions -> {out_path}")
