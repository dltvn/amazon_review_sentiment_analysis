# 15. Sentiment-Enhanced Recommender System Report

## 15a. Rating Enhancement Through Sentiment Analysis

### Problem Statement

Star ratings alone provide a coarse, discrete signal (integers 1-5) that fails to capture the nuance expressed in review text. A user who writes "This software is decent but has frustrating bugs" might rate it 3 stars, the same as someone who writes "Average product, nothing special." The text carries richer sentiment information that the rating collapses.

Sentiment-enhanced rating adjustment addresses this by blending quantitative ratings with qualitative text signals, producing continuous-valued ratings that better reflect reviewer intent (Chen et al., 2015).

### Sentiment Signals Used

We employ two complementary sentiment scoring approaches:

1. **VADER (Valence Aware Dictionary and sEntiment Reasoner):** A lexicon-based tool that produces a compound score in [-1, +1]. VADER is particularly effective for social media and review text because it handles capitalization ("GREAT!"), punctuation emphasis ("good!!!"), and emoji natively. It requires no training data and provides a continuous-valued signal.

2. **SVM (Support Vector Machine):** A supervised classifier trained on our dataset (Phase 2, Script 03) to predict 3-class sentiment labels (negative, neutral, positive). These are mapped to numeric values: negative=1.0, neutral=3.0, positive=5.0. Unlike VADER, this captures domain-specific patterns learned from the Amazon Software review corpus.

### Enhancement Formulas

We apply two combination strategies to each sentiment signal, creating a 2x2 matrix of 4 enhanced rating variants:

**Linear Blend:** A weighted average of the original rating and the sentiment-derived score:

    enhanced = alpha * rating + (1 - alpha) * sentiment_scaled

where alpha = 0.7 gives majority weight to the original rating. VADER compound is scaled from [-1, 1] to [1, 5] via: `scaled = 1 + (compound + 1) * 2`.

**Additive Adjustment:** The original rating is shifted by a fraction of the sentiment signal:

    enhanced = clamp(rating + beta * sentiment_signal, 1, 5)

where beta = 0.5. For VADER, the compound score [-1, 1] is used directly. For SVM, the 3-class score is normalized: `(ml_score - 3) / 2` maps {1, 3, 5} to {-1, 0, 1}.

The linear blend produces a smooth, continuous rating distribution (see Plot 23, `23_rating_distributions.png`), while the additive approach preserves the shape of the original distribution with smaller perturbations.

---

## 15b. System Architecture and Pseudocode

### Architecture

```
Raw JSON Dataset (Software_5.json, 12,805 reviews)
        |
        v
+-------------------+
| Data Loading      |  Keep: reviewerID, asin, overall, reviewText, summary
+-------------------+
        |
        v
+-------------------+     +-------------------+
| Text Cleaning     |---->| clean_text()      |  For ML: lowercase, demojize, strip URLs/HTML
| (two pipelines)   |---->| clean_for_vader() |  For VADER: preserve case/emoji, strip URLs/HTML
+-------------------+     +-------------------+
        |                         |
        v                         v
+-------------------+     +-------------------+
| ML Scoring        |     | VADER Scoring     |
| TF-IDF + SVM      |     | Compound [-1,+1]  |
| -> {neg,neu,pos}  |     |                   |
| -> {1.0,3.0,5.0}  |     |                   |
+-------------------+     +-------------------+
        |                         |
        v                         v
+-----------------------------------------------+
| Rating Enhancement (2 formulas x 2 signals)   |
|                                                |
|  1. VADER + Linear Blend                       |
|  2. VADER + Additive                           |
|  3. ML (SVM) + Linear Blend                    |
|  4. ML (SVM) + Additive                        |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
| SVD Matrix Factorization (Cornac)              |
| k=100 latent factors, 20 epochs                |
| 5-fold cross-validation per variant            |
| Metrics: RMSE, MAE                             |
+-----------------------------------------------+
        |
        v
+-------------------+
| Comparison Table  |  CSV + 5 plots (19-23)
| & Visualization   |
+-------------------+
```

### Pseudocode

```
LOAD dataset from JSON (12,805 reviews)
KEEP columns: reviewerID, asin, overall, reviewText, summary

FOR EACH review:
    combined_text = clean_text(summary) + " " + clean_text(reviewText)
    vader_text    = clean_for_vader(summary) + " " + clean_for_vader(reviewText)

# Sentiment scoring
vader_compound = VADER.polarity_scores(vader_text).compound   # [-1, 1]
ml_pred        = SVM.predict(TF-IDF.transform(combined_text)) # {neg, neu, pos}
ml_score       = MAP(ml_pred, {neg:1, neu:3, pos:5})

# Rating enhancement (4 variants)
vader_scaled         = 1 + (vader_compound + 1) * 2           # [1, 5]
rating_vader_blend   = 0.7 * overall + 0.3 * vader_scaled
rating_ml_blend      = 0.7 * overall + 0.3 * ml_score
rating_vader_additive = CLAMP(overall + 0.5 * vader_compound, 1, 5)
rating_ml_additive    = CLAMP(overall + 0.5 * (ml_score - 3) / 2, 1, 5)

# SVD evaluation
FOR EACH variant IN [baseline, vader_blend, vader_additive, ml_blend, ml_additive]:
    data = (reviewerID, asin, variant_rating)
    model = SVD(k=100, max_iter=20)
    results[variant] = CrossValidation(data, n_folds=5, metrics=[RMSE, MAE])

COMPARE results, GENERATE plots 19-23
```

### SVD in Collaborative Filtering

SVD (Singular Value Decomposition) decomposes the sparse user-item rating matrix R into low-rank factor matrices: R ~ U * S * V^T. In practice, Cornac's SVD implementation uses Funk-style stochastic gradient descent to learn latent factor vectors for each user and item, plus bias terms. The predicted rating for user u and item i is:

    r_hat(u, i) = mu + b_u + b_i + p_u^T * q_i

where mu is the global mean, b_u and b_i are user/item biases, and p_u, q_i are k-dimensional latent factor vectors. With k=100 factors, the model captures 100 latent dimensions of user preference and item characteristics.

---

## 15c. Results and Analysis

### Comparison Table

| Model | RMSE | +/- | MAE | +/- |
|-------|------|-----|-----|-----|
| VADER + Linear Blend | 0.6963 | 0.0166 | 0.6550 | 0.0171 |
| VADER + Additive | 0.7951 | 0.0198 | 0.7476 | 0.0198 |
| ML (SVM) + Linear Blend | 0.8042 | 0.0171 | 0.7561 | 0.0169 |
| Baseline (Raw Ratings) | 0.8320 | 0.0211 | 0.7844 | 0.0205 |
| ML (SVM) + Additive | 0.8393 | 0.0186 | 0.7890 | 0.0180 |

### Improvement Over Baseline

| Model | RMSE Improvement | % Change |
|-------|-----------------|----------|
| VADER + Linear Blend | -0.1358 | 16.32% better |
| VADER + Additive | -0.0369 | 4.44% better |
| ML (SVM) + Linear Blend | -0.0279 | 3.35% better |
| ML (SVM) + Additive | +0.0073 | 0.87% worse |

### Plot Descriptions

- **Plot 19** (`19_recommender_rmse_comparison.png`): Bar chart comparing RMSE across all 5 models with error bars showing cross-validation standard deviation. VADER + Linear Blend clearly achieves the lowest RMSE.

- **Plot 20** (`20_recommender_mae_comparison.png`): Same structure as Plot 19 but for MAE. The ranking is identical, confirming the RMSE findings.

- **Plot 21** (`21_formula_comparison.png`): Grouped bar chart comparing the two enhancement formulas (Linear Blend vs. Additive) for each sentiment signal. Linear Blend outperforms Additive for both VADER and ML signals.

- **Plot 22** (`22_signal_comparison.png`): Grouped bar chart comparing the two sentiment signals (VADER vs. ML) for each formula. VADER consistently outperforms ML (SVM) regardless of formula choice.

- **Plot 23** (`23_rating_distributions.png`): Four-panel histogram showing raw vs. enhanced rating distributions. The linear blend creates smoother, more continuous distributions, while the additive approach keeps the distribution closer to the original discrete shape.

### Analysis

**Best overall model: VADER + Linear Blend** achieves a 16.32% RMSE improvement over the baseline (0.6963 vs. 0.8320). This is a substantial gain from a simple enhancement strategy.

**Formula comparison:** Linear Blend consistently outperforms Additive adjustment across both sentiment signals (Plot 21). This is expected: the blend creates a genuinely new rating scale by mixing two independent signals, while the additive approach makes smaller perturbations that barely shift the distribution. The blend's 30% weight on sentiment is enough to inject meaningful text-derived information.

**Signal comparison:** VADER outperforms ML (SVM) for both formulas (Plot 22). This is likely because:
1. VADER produces a continuous compound score [-1, 1] that captures sentiment intensity, while SVM collapses to just 3 discrete values (1, 3, 5).
2. VADER preserves fine-grained sentiment distinctions (e.g., "good" vs. "excellent" vs. "EXCELLENT!!!"), while SVM groups all positive reviews into a single class.
3. The SVM was trained on labels derived from the same star ratings we're trying to enhance, creating a circular dependency that limits its ability to add new information.

**ML (SVM) + Additive performed slightly worse than baseline** (0.87% worse RMSE). This combination suffers from both the discrete signal problem and the small perturbation size, resulting in noise rather than useful adjustment.

### Limitations

- **Data sparsity:** The dataset has 12,805 reviews across 1,826 users and 802 products, meaning the user-item matrix is ~99% sparse. SVD handles this well, but denser data would likely show larger differentiation between models.
- **Single domain:** Results are specific to Amazon Software reviews. Other product categories may show different sentiment-rating relationships.
- **No top-N evaluation:** We only measure rating prediction accuracy (RMSE/MAE), not ranking quality (precision@k, recall@k), which matters more for real recommendation use cases.
- **Fixed hyperparameters:** Alpha=0.7 and beta=0.5 were chosen as reasonable defaults. Tuning these per-model could improve results further.
- **Duplicate handling:** Cornac automatically removes duplicate (user, item) pairs (~595 duplicates), which slightly reduces the effective dataset size.

### Future Work

- Grid search over alpha/beta parameters for each sentiment signal
- Explore additional sentiment signals (TextBlob, transformer-based models)
- Add top-N ranking evaluation metrics
- Test on datasets from other product categories
- Investigate hybrid approaches combining content-based and collaborative filtering
