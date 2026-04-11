# Amazon Software Reviews — Sentiment Analysis Report

**Dataset:** Amazon Software 5-core (Software_5.json)  
**Source:** Jianmo Ni, Jiacheng Li, and Julian McAuley, "Justifying recommendations using distantly-labeled reviews and fined-grained aspects," EMNLP 2019. Available at https://nijianmo.github.io/amazon/index.html

---

## Phase 1

### 1. Dataset Exploration (`phase1/01_data_exploration.py`)

#### Dataset Overview

| Field | Value |
|---|---|
| Total reviews | 12,805 |
| Columns | overall, verified, reviewTime, reviewerID, asin, style, reviewerName, reviewText, summary, unixReviewTime, vote, image |

**Missing values:**

| Column | Missing |
|---|---|
| style | 5,644 (44.1%) |
| image | 12,734 (99.4%) |
| vote | 8,903 (69.5%) |
| reviewerName | 9 |
| reviewText | 1 |
| summary | 6 |
| All other columns | 0 |

The `image` field is nearly entirely absent and is not useful for sentiment analysis. The `vote` field (helpful vote count) is missing for 69.5% of reviews.

#### Rating Distribution

| Rating | Count | % |
|---|---|---|
| 1 | 1,500 | 11.7% |
| 2 | 719 | 5.6% |
| 3 | 1,598 | 12.5% |
| 4 | 3,016 | 23.6% |
| 5 | 5,972 | 46.6% |

- Mean rating: **3.88** — the dataset skews strongly positive.
- Median: **4.0**, Std dev: **1.36**
- 5-star reviews make up nearly half (46.6%) of all reviews, reflecting a well-known positivity bias in voluntary review datasets.

*See `phase1/plots/01_rating_distribution.png`.*

#### Distribution of Reviews per Product

| Metric | Value |
|---|---|
| Unique products (ASINs) | 62 |
| Average reviews / product | 15.97 |
| Median reviews / product | 12.0 |
| Max reviews / product | 452 |
| Min reviews / product | 1 |

The distribution is right-skewed: most products have fewer than 20 reviews, but one product accounts for 452. This is consistent with the 5-core constraint (at least 5 reviews per product/user).

*See `phase1/plots/02_reviews_per_product_top20.png` and `03_reviews_per_product_hist.png`.*

#### Distribution of Reviews per User

| Metric | Value |
|---|---|
| Unique reviewers | 29 |
| Average reviews / user | 7.01 |
| Median reviews / user | 6.0 |
| Max reviews / user | 55 |
| Users with > 1 review | 1,825 |

The vast majority of users contributed multiple reviews. The 5-core constraint guarantees every reviewer has at least 5 reviews.

*See `phase1/plots/04_reviews_per_user_hist.png`.*

#### Review Length Analysis

| Metric | Value |
|---|---|
| Mean length | 175.38 words |
| Median length | 98.0 words |
| Max length | 5,118 words |
| Min length | 0 words |
| Std dev | 257.13 words |
| IQR outlier upper bound | 503 words |
| Reviews above upper bound (outliers) | 869 (6.8%) |

Review lengths are highly right-skewed (mean >> median), indicating a long tail of very verbose reviews. There is 1 review with 0 words (empty `reviewText`).

**Average review length by rating:**

| Rating | Mean (words) | Median (words) |
|---|---|---|
| 1 | 180.96 | 115.0 |
| 2 | 203.87 | 140.0 |
| 3 | 234.64 | 148.5 |
| 4 | 217.40 | 142.0 |
| 5 | 133.48 | 57.0 |

5-star reviewers write shorter reviews on average; lower-rated reviews tend to be more detailed, likely because dissatisfied customers elaborate on their complaints.

*See `phase1/plots/05_review_length_by_rating.png`, `06_review_length_hist.png`, and `08_avg_length_by_rating.png`.*

#### Duplicates

| Check | Count |
|---|---|
| Exact duplicate rows (excluding unhashable columns) | 820 |
| Duplicate `reviewText` values | 2,199 |

The high number of duplicate `reviewText` entries (2,199 out of 12,805 = 17.2%) warrants attention during preprocessing. Many of these are likely copy-paste or templated reviews.

#### Verified Purchase Breakdown

| Verified | Count |
|---|---|
| False | 7,631 (59.6%) |
| True | 5,174 (40.4%) |

Most reviews are unverified. Unverified reviews may be less trustworthy as sentiment signals.

*See `phase1/plots/07_verified_by_rating.png`.*

#### Helpful Votes

Only 3,902 reviews (30.5%) have a `vote` (helpful vote count) recorded.

#### Key Conclusions

1. The dataset is **strongly positive-skewed**: 70% of reviews are rated 4 or 5 stars. Any classifier must account for class imbalance.
2. **Review text is the primary useful text field**; `summary` is a short secondary signal. Both will be considered during preprocessing.
3. **6.8% of reviews are length outliers** (>503 words); these will be examined during preprocessing.
4. **Duplicate reviews are substantial** (17.2% by text); deduplication should be considered.
5. Lower-rated reviews are significantly longer, suggesting more elaborate negative feedback.

---

### 2. Text Pre-processing (`phase1/02_preprocessing.py`)

#### Column Selection

Two text columns were selected as model inputs:

| Column | Justification |
|---|---|
| `reviewText` | The primary long-form opinion written by the reviewer. It contains the richest and most detailed sentiment signal — specific praise, complaints, comparisons, and recommendations. |
| `summary` | A short reviewer-written headline that often captures the overall sentiment concisely (e.g., "Great product", "Waste of money"). Prepending it to `reviewText` ensures the headline's strong sentiment keywords are always present, even when the body is ambiguous. |

`overall` (the star rating) is used exclusively for ground-truth labeling and is not passed to the models. All other columns (e.g., `image`, `vote`, `style`, `verified`) were dropped: they are either mostly missing, non-textual, or not relevant to lexicon-based sentiment analysis.

#### Sentiment Labeling

Labels are assigned per the project specification:

| Rating | Label |
|---|---|
| 4, 5 | positive |
| 3 | neutral |
  | 1, 2 | negative |

#### Cleaning Steps and Justifications

| Step | Action | Justification |
|---|---|---|
| Drop missing `reviewText` | Remove 1 row with null body | No text means no signal |
| Deduplicate by `reviewText` | Remove 2,199 duplicate bodies | Duplicate reviews inflate class counts and bias evaluation |
| Remove empty reviews | Remove reviews with 0 words | No signal; would produce undefined/neutral scores |
| Outlier removal (IQR upper) | Remove reviews > 531 words (714 removed) | Very long reviews dominate lexicon score averaging and are atypical of real user behaviour |
| Lowercase | Applied for TextBlob input (`combined_text`) | TextBlob's Pattern lexicon is case-insensitive; lowercasing reduces vocabulary noise |
| Remove URLs / HTML | Applied to both model inputs | URLs and HTML tags carry no sentiment |
| Normalize whitespace | Collapse multiple spaces/newlines | Prevents tokenization artefacts |
| Preserve casing for VADER | VADER input (`vader_text`) keeps original case | VADER uses ALL CAPS and punctuation (`!!!`) as emphasis boosters; stripping them would reduce accuracy |

#### Label Distribution After Preprocessing

| Stage | Positive | Neutral | Negative | Total |
|---|---|---|---|---|
| After deduplication | 7,295 | 1,422 | 1,888 | 10,605 |
| After outlier removal | 6,836 | 1,292 | 1,763 | 9,891 |
| 1,000-review stratified sample | 691 | 131 | 178 | 1,000 |

The class imbalance (69% positive) is preserved proportionally in the sample via stratified sampling (random seed 42).

---

### 3. Lexicon Selection (`phase1/03_lexicon_modeling.py`)

The three candidates from the project specification were evaluated:

| Lexicon | Pros | Cons | Selected |
|---|---|---|---|
| **VADER** | Designed for short social/review text; handles punctuation, CAPS, negation, degree modifiers natively; no preprocessing required beyond noise removal; fast | Less suited to formal/technical prose | Yes |
| **TextBlob** | Simple API; good on clean prose; provides both polarity and subjectivity scores | Does not leverage CAPS or punctuation emphasis; based on older Pattern lexicon | Yes |
| SentiWordNet | Fine-grained synset-level scores; useful for formal text | Requires POS tagging and word sense disambiguation; complex pipeline with higher error rate on short informal text | No |

**VADER** and **TextBlob** were chosen because they cover complementary approaches (rule-based with emphasis heuristics vs. lexicon averaging on clean prose) and are well-matched to the review domain, making their comparison informative.

#### Pre-processing per Model

**VADER** (`vader_text`): `summary` + `reviewText` with original casing and punctuation preserved. Only URLs, HTML tags, non-ASCII characters, and excess whitespace are removed. VADER's compound score heuristics rely on capitalization and punctuation — normalizing these would discard valid signal.

**TextBlob** (`combined_text`): `summary` + `reviewText` fully lowercased with URLs, HTML, and whitespace cleaned. TextBlob's Pattern lexicon performs lookup on lowercased tokens; clean, uniform text reduces noise.

#### Thresholds

Both models use the same threshold convention (following Hutto & Gilbert, 2014 for VADER):

| Score range | Predicted label |
|---|---|
| >= 0.05 | positive |
| <= -0.05 | negative |
| (-0.05, 0.05) | neutral |

#### Prediction Distributions on 1,000-Review Sample

| Label | True | VADER | TextBlob |
|---|---|---|---|
| positive | 691 | 818 | 824 |
| neutral | 131 | 38 | 121 |
| negative | 178 | 144 | 55 |

Both models over-predict positive, consistent with the high positive base rate. VADER severely under-predicts neutral (38 vs. 131 true), collapsing most neutral reviews into positive. TextBlob distributes neutral predictions more faithfully but under-predicts negative.

---

### 4. Model Evaluation (`phase1/04_evaluation.py`)

#### Metrics (weighted averages, 1,000-review sample)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| VADER | **0.7330** | 0.6674 | **0.7330** | **0.6916** |
| TextBlob | 0.6820 | **0.6669** | 0.6820 | 0.6476 |

*See `phase1/plots/11_model_comparison.png`.*

#### Per-class Results

**VADER:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| positive | 0.78 | 0.92 | 0.85 | 691 |
| neutral | 0.13 | 0.04 | 0.06 | 131 |
| negative | 0.62 | 0.51 | 0.56 | 178 |

**TextBlob:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| positive | 0.76 | 0.91 | 0.83 | 691 |
| neutral | 0.13 | 0.12 | 0.13 | 131 |
| negative | 0.69 | 0.21 | 0.33 | 178 |

*See `phase1/plots/09_vader_confusion_matrix.png` and `10_textblob_confusion_matrix.png`.*

#### Conclusions

1. **VADER outperforms TextBlob** on all metrics: +5.1 pp accuracy, +4.4 pp F1.
2. **Both models handle positive reviews well** (F1 ~0.83–0.85) due to the high base rate; they struggle with neutral and negative classes.
3. **Neutral is the hardest class for both models.** Neither VADER nor TextBlob has strong coverage of neutral language — ambiguous or factual software reviews with no strong polarity words score near zero and are misclassified.
4. **VADER's recall for negative (0.51) beats TextBlob's (0.21)**: VADER better captures negative sentiment through its punctuation/caps heuristics, which software reviewers often use when expressing frustration.
5. **TextBlob achieves slightly higher precision on negative (0.69 vs. 0.62)**: when it does predict negative, it is more often correct — but it misses the majority of negative reviews entirely.
6. The strong positive class imbalance (69%) inflates accuracy for both models; macro-average F1 (VADER: 0.49, TextBlob: 0.43) gives a more honest picture of overall performance across all three classes.

---

## Phase 2

### 5. Machine Learning Sentiment Analysis (`phase2/03_ml_modeling.py`)

Phase 2 extends the lexicon analysis with supervised machine learning models trained on the same cleaned and labeled Amazon Software review data.

#### Phase 2 Sample and Split

The Phase 2 preprocessing pipeline requested a 20,000-review sample, but after cleaning only 9,891 usable reviews remained. Instead of downsampling further, the full cleaned dataset was used.

| Item | Value |
|---|---:|
| Cleaned reviews available | 9,891 |
| Training set (70%) | 6,923 |
| Test set (30%) | 2,968 |

**Class distribution in cleaned sample:**

| Sentiment | Count |
|---|---:|
| Positive | 6,836 |
| Negative | 1,763 |
| Neutral | 1,292 |

**Class distribution in test set:**

| Sentiment | Count |
|---|---:|
| Positive | 2,051 |
| Negative | 529 |
| Neutral | 388 |

#### Text Representation

TF-IDF was used for machine learning because it is well suited to sparse sentiment classification and works especially well with linear classifiers.

To reduce overfitting, the representation was simplified compared with a larger, noisier feature space:

| Parameter | Value | Reason |
|---|---|---|
| `max_features` | 3000 | Reduces memorization of rare words |
| `ngram_range` | (1,1) | Unigrams generalize better than a larger unigram+bigram space here |
| `min_df` | 3 | Removes rare noisy tokens |
| `max_df` | 0.90 | Removes overly common terms |
| `sublinear_tf` | True | Dampens the effect of repeated words |

Resulting matrix shapes:

| Matrix | Shape |
|---|---|
| Training TF-IDF | (6923, 3000) |
| Test TF-IDF | (2968, 3000) |

#### Models Trained

Four models were trained and tuned:

1. Logistic Regression
2. Linear SVM (`LinearSVC`)
3. LightGBM
4. MLP

The machine learning tuning focused on reducing overfitting:

- Logistic Regression used smaller `C` values and optional class weighting.
- SVM used a linear margin classifier with smaller `C` values.
- LightGBM used shallower trees, fewer leaves, subsampling, and L2 regularization.
- MLP used smaller hidden layers and stronger regularization (`alpha`).

Three-fold cross-validation was used during grid search.

#### Training Results Summary

*See `phase2/data/training_summary.csv`.*

| Model | Best CV Accuracy | Training Accuracy | Overfitting Gap |
|---|---:|---:|---:|
| Logistic Regression | 0.7817 | 0.8339 | 0.0521 |
| SVM | **0.8059** | 0.9249 | 0.1190 |
| LightGBM | 0.7751 | 0.8740 | 0.0989 |
| MLP | 0.7718 | 1.0000 | 0.2282 |

**Interpretation:**

- Logistic Regression is the most stable model and has the smallest overfitting gap.
- SVM achieves the best cross-validation accuracy, but still shows noticeable overfitting.
- LightGBM is competitive but still learns the training set more strongly than Logistic Regression.
- MLP remains heavily overfit despite simplification and regularization.

---

### 6. Phase 2 Testing and Comparison (`phase2/04_evaluation.py`)

All six models were tested on the exact same 2,968-review test set so that the lexicon and machine learning approaches could be compared fairly.

#### Overall Test Results

*See `phase2/data/model_comparison.csv`.*

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7911 | 0.7783 | 0.7911 | 0.7477 |
| **SVM** | **0.7989** | 0.7719 | **0.7989** | **0.7782** |
| LightGBM | 0.7850 | 0.7561 | 0.7850 | 0.7447 |
| MLP | 0.7662 | 0.7622 | 0.7662 | 0.7641 |
| VADER | 0.7311 | 0.6733 | 0.7311 | 0.6898 |
| TextBlob | 0.6900 | 0.6732 | 0.6900 | 0.6588 |

The machine learning models clearly outperform the lexicon baselines on the shared test set.

**Average weighted F1:**

- ML models: **0.7587**
- Lexicon models: **0.6743**
- ML improvement over lexicon: **12.5%**

#### Plot-Based Analysis

The following discussion converts the Phase 2 plots into report-ready text while preserving placeholders for later image insertion.

##### Logistic Regression Confusion Matrix

*See `phase2/plots/09_lr_confusion_matrix.png`.*

[insert plot here]

Logistic Regression predicts positive reviews very well: 2,006 of 2,051 positive reviews are classified correctly. Its main weakness is the neutral class. Only 42 of 388 neutral reviews are predicted correctly, while 296 are incorrectly pushed into the positive class. Negative reviews are handled better than neutral ones, with 300 correct predictions out of 529.

##### SVM Confusion Matrix

*See `phase2/plots/10_svm_confusion_matrix.png`.*

[insert plot here]

The SVM has the strongest balance across classes. It correctly classifies 1,933 positive, 349 negative, and 89 neutral reviews. Compared with Logistic Regression, it sacrifices a small number of positive predictions to improve negative and neutral detection, which leads to the best weighted F1 score overall.

##### LightGBM Confusion Matrix

*See `phase2/plots/11_lgbm_confusion_matrix.png`.*

[insert plot here]

LightGBM behaves similarly to Logistic Regression. It classifies positive reviews well (2,003 correct), but neutral reviews remain difficult (49 correct out of 388). It also confuses many negative reviews with positive ones (224 cases), which hurts negative-class performance.

##### MLP Confusion Matrix

*See `phase2/plots/12_mlp_confusion_matrix.png`.*

[insert plot here]

The MLP gives the best neutral-class detection among all models, with 135 correct neutral predictions. However, it makes noticeably more mistakes on positive reviews than the linear models. This explains why the MLP performs reasonably on the test set but is still not the best final choice, especially given its severe overfitting during training.

##### VADER Confusion Matrix

*See `phase2/plots/13_vader_confusion_matrix.png`.*

[insert plot here]

VADER performs adequately on positive reviews (1,908 correct) but struggles heavily with neutral reviews, identifying only 25 of 388 correctly. Many neutral and negative reviews are predicted as positive, showing the limitations of rule-based lexicons on domain-specific software reviews.

##### TextBlob Confusion Matrix

*See `phase2/plots/14_textblob_confusion_matrix.png`.*

[insert plot here]

TextBlob is the weakest model overall. It still identifies positive reviews fairly well (1,861 correct), but it has weak recall on both negative and neutral classes. Only 127 of 529 negative reviews are classified correctly.

##### Model Comparison Chart

*See `phase2/plots/15_model_comparison.png`.*

[insert plot here]

This chart compares accuracy, precision, recall, and weighted F1 for all six models. The SVM leads overall, while the lexicon models form a clearly weaker group. The chart confirms that machine learning learns the review domain better than the fixed lexicon rules.

##### ML vs Lexicon Chart

*See `phase2/plots/16_ml_vs_lexicon.png`.*

[insert plot here]

Separating the machine learning models from the lexicon models makes the performance gap easier to interpret. The lexicon models are useful baselines, but they cannot adapt to software-specific terms and mixed sentiment as effectively as supervised models.

##### ML-Only Comparison Chart

*See `phase2/plots/17_ml_models_comparison.png`.*

[insert plot here]

Among the machine learning models, SVM is the strongest overall, followed by MLP on weighted F1. However, once model stability is considered, Logistic Regression and SVM remain safer final choices than MLP because they are easier to interpret and show less extreme training behavior.

##### Per-Class F1 Chart

*See `phase2/plots/18_per_class_f1.png`.*

[insert plot here]

The per-class F1 chart highlights that neutral sentiment is the hardest class in the dataset. SVM achieves the best positive F1 (0.8922) and negative F1 (0.6816), while MLP achieves the best neutral F1 (0.3581). This suggests that different models capture different error trade-offs, but SVM remains the best overall compromise.

#### Per-Class Results

*See `phase2/data/per_class_metrics.csv`.*

| Model | Positive F1 | Neutral F1 | Negative F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.8777 | 0.1871 | 0.6550 |
| **SVM** | **0.8922** | 0.3074 | **0.6816** |
| LightGBM | 0.8768 | 0.2021 | 0.6304 |
| MLP | 0.8726 | **0.3581** | 0.6414 |
| VADER | 0.8441 | 0.0973 | 0.5261 |
| TextBlob | 0.8314 | 0.1615 | 0.3547 |

Neutral reviews are the most difficult class for every model. This is expected because neutral software reviews often mix praise and criticism or use factual language with weak explicit polarity.

#### Phase 2 Conclusions

1. Using all 9,891 cleaned reviews improved the training base compared with a much smaller sample.
2. TF-IDF with a reduced unigram vocabulary was an effective representation for this task.
3. SVM is the best overall test-time model.
4. Logistic Regression is the most stable model with the smallest overfitting gap.
5. MLP improved neutral detection but remained strongly overfit.
6. All machine learning models outperform VADER and TextBlob on the same test data.
7. Neutral sentiment remains the hardest class for both lexicon and machine learning approaches.
