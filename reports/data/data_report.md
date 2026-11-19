# 1. Data Analysis and Preparation

## 1.1. Dataset Overview

The project uses the Digikala customer reviews dataset, consisting of two raw CSV files:

| Dataset | File Size | Number of Records | Number of Columns |
|---|---:|---:|---:|
| Comments | 1,219.3 MB | 6,156,289 | 15 |
| Products | 246.7 MB | 1,283,496 | 12 |

The comments dataset contains customer reviews, ratings, recommendation statuses, product identifiers, and additional metadata.

The original dataset was encoded using `UTF-8-SIG` with a comma delimiter. Polars was used for the initial data inspection and schema inference.

The primary columns relevant to this project include:

- `id`: Unique review identifier.
- `body`: Main review text.
- `rate`: Numerical rating assigned by the customer.
- `recommendation_status`: Customer recommendation label.
- `product_id`: Identifier of the reviewed product.
- `is_buyer`: Indicates whether the reviewer purchased the product.

**Important distinction:** The `rate` column in the comments dataset ranges from 0 to 5, whereas the `Rate` column in the products dataset ranges from 0 to 100. These represent different variables and were treated separately.

## 1.2. Dataset Structure and Coverage

The initial analysis identified:

- 6,156,289 customer reviews.
- 331,548 unique products with at least one review.
- 1,283,496 products in the complete product catalog.

Approximately 25.8% of the products in the catalog have at least one customer review.

The distribution of reviews per product is highly uneven:

| Statistic | Number of Reviews |
|---|---:|
| Minimum | 1 |
| 25th Percentile | 1 |
| Median | 4 |
| 75th Percentile | 13 |
| 90th Percentile | 46 |
| 99th Percentile | 200 |
| Maximum | 400 |
| Mean | 18.57 |

An interesting observation was the maximum of exactly 400 reviews per product. Three products reached this limit, suggesting a possible artificial collection or publication cap rather than a naturally occurring maximum.

Furthermore, 91,944 products had only one review, while 198,458 products had five or fewer reviews.

This distribution highlights the importance of considering product-level grouping during dataset splitting.

## 1.3. Missing Value Analysis

A significant finding during data exploration was that missing values in several string columns were represented by the literal string `"nan"` rather than actual null values.

Consequently, a sentinel-aware missing value analysis was performed.

| Column | Missing Values | Percentage |
|---|---:|---:|
| `true_to_size_rate` | 6,065,516 | 98.53% |
| `disadvantages` | 5,744,017 | 93.30% |
| `advantages` | 5,448,084 | 88.50% |
| `title` | 2,865,062 | 46.54% |
| `recommendation_status` | 894,426 | 14.53% |
| `seller_title` | 302,181 | 4.91% |
| `seller_code` | 302,181 | 4.91% |
| `body` | 637 | 0.01% |

The missingness analysis revealed that missing values in `recommendation_status` were not randomly distributed.

In particular:

- 100% of non-buyers (`is_buyer=False`) had missing recommendation statuses.
- Among buyers, approximately 10.1% had missing recommendation statuses.
- Reviews without recommendation labels had a higher average rating (4.04) than reviews with available labels (3.58).

This indicates a structural relationship between the availability of recommendation labels and user purchasing behavior.

**Implication:** Supervised training data may be biased toward buyers who actively provide recommendation feedback. This limitation should be considered when interpreting model performance and generalization.

## 1.4. Target Variable Distribution

The project considers two distinct prediction targets.

### 1.4.1. Recommendation Status

The distribution of available recommendation labels is:

| Class | Number of Records | Percentage |
|---|---:|---:|
| `recommended` | 4,162,839 | 79.11% |
| `no_idea` | 594,121 | 11.29% |
| `not_recommended` | 504,903 | 9.60% |

The dataset exhibits substantial class imbalance, with an approximate majority-to-minority ratio of 8.2:1.

This imbalance is an important consideration for model selection, training, and evaluation.

### 1.4.2. Numerical Rating

The original `rate` variable has the following characteristics:

- Minimum: 0
- Maximum: 2500
- Mean: 3.65
- Median: 4.0
- Standard deviation: 1.86
- 3.71% of values are fractional.

Only one observation falls outside the expected rating range of [0, 5], with a value of 2500.

For the rating classification task, a three-class variable named `rating_band` was constructed using fixed thresholds:

| Class | Definition | Number of Records | Percentage |
|---|---|---:|---:|
| `low` | 0 ≤ rate < 2.5 | 1,092,329 | 17.74% |
| `medium` | 2.5 ≤ rate < 4.0 | 1,059,054 | 17.20% |
| `high` | 4.0 ≤ rate ≤ 5.0 | 4,004,905 | 65.05% |
| `invalid` | Outside valid range | 1 | <0.01% |

The resulting class distribution has an approximate imbalance ratio of 3.8:1.

### 1.4.3. Relationship Between the Two Targets

An important observation was the relationship between numerical ratings and recommendation statuses.

| Recommendation Status | Mean Rating | Median Rating | Standard Deviation |
|---|---:|---:|---:|
| `recommended` | 3.93 | 4.0 | 1.94 |
| `no_idea` | 2.92 | 3.0 | 0.91 |
| `not_recommended` | 1.45 | 1.0 | 0.98 |

Although the two targets are correlated, they are not equivalent.

For example, reviews labeled `no_idea` tend to have ratings concentrated around three stars.

This observation supports treating recommendation prediction and rating classification as separate prediction tasks rather than assuming that one target can directly replace the other.

## 1.5. Text Length Analysis

The `body` column was selected as the primary textual input for the NLP models.

The text length distribution was examined using both character and word counts.

| Percentile | Characters | Words |
|---|---:|---:|
| Minimum | 1 | 1 |
| 50th | 29 | 6 |
| 90th | 106 | 22 |
| 95th | 152 | 31 |
| 99th | 297 | 61 |
| Maximum | 4,000 | 908 |

Additional observations:

- 10.9% of reviews contain fewer than 10 characters.
- 34.1% contain fewer than 20 characters.
- 16.7% contain fewer than three words.
- Only 0.96% contain more than 300 characters.
- Approximately 0.23% contain more than 500 characters.

A notable relationship was identified between review length and recommendation status.

| Recommendation Status | Mean Characters | Median Characters | 90th Percentile |
|---|---:|---:|---:|
| `recommended` | 44.25 | 27 | 95 |
| `no_idea` | 59.30 | 38 | 129 |
| `not_recommended` | 73.26 | 48 | 159 |

Negative reviews tend to be substantially longer than positive reviews.

This introduces a potential confounding factor: a model might partially exploit review length instead of learning meaningful semantic patterns.

Therefore, model evaluation should consider whether predictions rely on textual meaning or superficial characteristics such as document length.

Based on the 99th percentile of 61 words, a maximum sequence length of 128 tokens was selected as the initial configuration for transformer-based models. This provides sufficient coverage for the vast majority of reviews while limiting computational requirements.

## 1.6. Duplicate Records and Label Inconsistency

Duplicate analysis was performed at both the review identifier and text levels.

### Duplicate IDs

The initial dataset contained:

- 3,226 duplicated review IDs.
- 6,455 records associated with these duplicated IDs.
- 2,911 IDs with exact duplicate records.
- 315 IDs with conflicting record contents.

Both exact and divergent duplicate IDs were removed during preprocessing to preserve review-level identifier integrity.

### Duplicate Texts

Approximately 36% of the records contained duplicated review bodies.

Specifically:

- 197,745 unique text values appeared more than once.
- These texts accounted for 2,217,776 records.
- The most frequently repeated text appeared 121,349 times.

Unlike duplicate IDs, duplicate text bodies were retained.

The reason is that identical short reviews may represent genuine feedback from different users. Removing them would alter the original data distribution and potentially introduce an unrealistic evaluation dataset.

### Conflicting Labels

A particularly important finding was the presence of identical review bodies associated with different recommendation labels.

- 31,942 unique review bodies had multiple recommendation labels.
- These conflicting texts represented 1,221,047 records.
- Approximately 19.8% of supervised records were affected.

Examples include common expressions such as:

- `عالی`
- `خوب بود`
- `خوبه`
- `خوب`
- `عالی بود`

These texts appeared with all three recommendation classes.

This represents an inherent source of label noise: identical textual inputs can correspond to different target labels.

Such inconsistencies establish a practical limitation on the achievable classification performance, particularly for models relying exclusively on review text.

These records were retained to avoid changing the original target distribution.

## 1.7. Product-Level Review Concentration

The distribution of reviews across products was also examined.

| Percentage of Reviews | Percentage of Products |
|---|---:|
| 25% | 2.31% |
| 50% | 5.34% |
| 80% | 17.67% |
| 90% | 31.38% |

The results indicate moderate concentration of reviews across products.

This reinforced the decision to use product-grouped dataset splitting rather than randomly splitting individual reviews.

---

# 2. Data Cleaning and Preprocessing

## 2.1. Persian Text Normalization

Persian text normalization was implemented to address inconsistencies in character representation and text formatting.

A preliminary inspection of 5,000 randomly sampled records (seed = 2026) identified the following patterns:

| Pattern | Frequency |
|---|---:|
| Arabic Yeh (`ي`) | 0.98% |
| Arabic Kaf (`ك`) | 0.42% |
| Zero-width non-joiner (ZWNJ) | 3.00% |
| URLs | 0% |
| HTML tags | 0% |
| Emojis | 0.76% |
| ASCII digits | 1.72% |
| Persian digits | 3.52% |
| Arabic digits | 0.04% |
| Other zero-width controls | 3.02% |
| Newline characters | 10.18% |

A custom `normalize_persian` function was implemented using the Hazm normalization library.

The normalization pipeline included:

1. Removing URLs and HTML tags.
2. Removing zero-width space (ZWSP), zero-width joiner (ZWJ), and BOM characters.
3. Preserving zero-width non-joiner (ZWNJ) to maintain Persian word structure.
4. Normalizing Arabic Yeh and Kaf characters.
5. Standardizing whitespace and half-space formatting using `hazm.Normalizer`.
6. Collapsing repeated horizontal whitespace.
7. Removing leading and trailing whitespace.

An additional normalization step converted Persian and Arabic digits into ASCII digits because the selected Hazm version (0.12.1) did not perform this conversion automatically.

An idempotence test was performed on nine test cases, with zero violations observed.

This indicates that applying the normalization function repeatedly does not introduce additional changes to the tested inputs.

## 2.2. Cleaning Operations

The following cleaning operations were applied:

- Converted string-based missing value representations (`"nan"`, `"NaN"`, `"null"`, `"None"`, and empty strings) into actual null values in `recommendation_status` and `body`.
- Removed 3,226 duplicated review IDs.
- Removed unnecessary metadata columns.
- Generated normalized text and rating-related features.

The retained columns were:

| Column | Description |
|---|---|
| `id` | Review identifier |
| `recommendation_status` | Recommendation classification target |
| `rate` | Original numerical rating |
| `product_id` | Product identifier for grouped splitting |
| `is_buyer` | Buyer indicator |
| `rating_band` | Three-class rating target |
| `is_valid_rating_band` | Validity indicator for rating classification |
| `body_normalized` | Normalized Persian review text |

The following columns were removed from the processed dataset:

`title`, `created_at`, `advantages`, `disadvantages`, `likes`, `dislikes`, `seller_title`, `seller_code`, and `true_to_size_rate`.

The processed dataset was saved as:

`data/processed/comments_clean.parquet`

| Property | Value |
|---|---:|
| Records | 6,149,834 |
| Columns | 8 |
| File Size | 178.5 MB |
| Compression | Zstandard (Zstd) |

Post-cleaning validation confirmed that:

- 637 records still had missing normalized review bodies, consistent with the initial analysis.
- Recommendation label counts decreased slightly due to duplicate ID removal.
- Only one record had an invalid rating band.

---

# 3. Dataset Splitting

## 3.1. Product-Grouped Stratified Splitting

To prevent data leakage between training and evaluation datasets, splitting was performed at the product level.

All reviews belonging to the same product were assigned to the same split.

The splitting strategy used a 70/15/15 allocation:

- 70% Training
- 15% Validation
- 15% Test

Stratification was performed using each product's majority recommendation class.

Products without any available recommendation labels were assigned to a separate `__none__` group.

The majority-class distribution across products was:

| Majority Class | Number of Products |
|---|---:|
| `recommended` | 258,699 |
| `not_recommended` | 27,513 |
| `no_idea` | 22,965 |
| `__none__` | 22,335 |

The resulting product distribution was:

| Split | Products | Percentage |
|---|---:|---:|
| Train | 232,057 | 70% |
| Validation | 49,727 | 15% |
| Test | 49,728 | 15% |

Disjointness assertions confirmed that no product appeared in more than one split.

The class distributions also remained sufficiently representative, with a maximum difference of approximately two percentage points between splits.

This strategy provides a more realistic evaluation of generalization to previously unseen products.

## 3.2. Semi-Supervised Data Masking

To support the semi-supervised learning component, a masking procedure was designed to move a portion of labeled training examples into an unlabeled training subset.

The masking configuration was:

| Parameter | Value |
|---|---|
| Masking Rate | 20% |
| Masking Seed | `digikala-semi-supervised-v1` |
| Hash Algorithm | SHA-256 |
| Selection Strategy | Class-wise deterministic selection |

Masking was performed independently within each recommendation class.

For reproducibility, each review ID was assigned a stable hash:

`SHA256(MASK_SEED + "|" + row_id)`

Records were sorted according to their hash values, and the first 20% of each class were selected for masking.

The results were:

| Class | Original Training Records | Masked Records | Masking Rate |
|---|---:|---:|---:|
| `recommended` | 2,921,036 | 584,207 | 20.00% |
| `not_recommended` | 351,901 | 70,380 | 20.00% |
| `no_idea` | 415,723 | 83,144 | 20.00% |
| **Total** | **3,688,660** | **737,731** | **20.00%** |

The resulting `train_pseudo_unlabeled` dataset contains the masked review IDs and their normalized text.

**Note:** The original execution summary contains a discrepancy regarding whether masked records were removed from `train_supervised`. This must be verified against `split_manifest.json` before treating the final dataset counts as definitive.

## 3.3. Final Dataset Files

The pipeline generated the following standardized files:

| File | Records | Size |
|---|---:|---:|
| `train_supervised.csv` | 3,578,692 | 458.1 MB |
| `train_pseudo_unlabeled.csv` | 737,731 | 68.8 MB |
| `natural_unlabeled.csv` | 627,763 | 60.9 MB |
| `validation.csv` | 913,570 | 117.4 MB |
| `test.csv` | 919,841 | 118.0 MB |
| `split_manifest.json` | — | 2 KB |

The `split_manifest.json` file records the main configuration and metadata required to reproduce the data preparation process, including:

- Random seeds.
- Rating band thresholds.
- Ordinal encoding.
- Split strategy.
- Dataset counts.
- Per-class masking statistics.

---

# 4. Key Data Preparation Decisions

The following decisions define the final data preparation strategy:

| Decision | Rationale |
|---|---|
| Use only `body_normalized` as model input | Focus on textual information rather than metadata. |
| Use `product_id` only for splitting | Prevent product-level data leakage. |
| Remove duplicate review IDs | Maintain review-level identifier integrity. |
| Preserve duplicate review bodies | Avoid artificially changing the original text distribution. |
| Preserve conflicting labels | Retain the original label distribution and document inherent label noise. |
| Retain the invalid rating record | Preserve the review text while excluding the invalid rating from valid rating classes. |
| Use product-grouped stratified splitting | Ensure product-level separation and representative class distributions. |
| Apply deterministic 20% masking | Construct a reproducible semi-supervised training subset. |
| Preserve Persian half-spaces and negations | Avoid damaging linguistic information during normalization. |
| Use a maximum sequence length of 128 tokens | Cover the vast majority of reviews while controlling computational cost. |

---

# 5. Limitations and Data-Related Challenges

Several limitations were identified during data analysis.

**1. Structural Missingness**

Recommendation labels are unavailable for all non-buyers. Therefore, supervised training data does not represent every reviewer equally.

**2. Class Imbalance**

The recommendation target is dominated by the `recommended` class, requiring appropriate evaluation metrics and potentially class-imbalance handling techniques.

**3. Inherent Label Noise**

Approximately 19.8% of supervised records are associated with text bodies that appear with multiple recommendation labels. This introduces ambiguity that cannot always be resolved using text alone.

**4. Text-Length Confounding**

Negative reviews tend to be longer than positive reviews. Models may exploit this correlation rather than learning semantic features.

**5. Product-Level Generalization**

Reviews from the same product can share contextual information. Product-grouped splitting reduces the risk of information leakage and provides a more demanding evaluation setting.

**6. Masking and Dataset Accounting**

The final relationship between `train_supervised` and `train_pseudo_unlabeled` requires verification against the split manifest to ensure that the semi-supervised dataset construction was performed correctly.

---

# 6. Conclusion

The data preparation stage transformed the original Digikala review dataset into a structured and reproducible dataset suitable for NLP experimentation.

Starting with 6.15 million raw reviews, the pipeline performed data inspection, missing value analysis, target distribution analysis, Persian text normalization, duplicate handling, and product-grouped stratified splitting.

The final preparation strategy also incorporated deterministic class-wise masking to support semi-supervised learning.

Several important dataset characteristics were identified, including structural missingness, class imbalance, repeated short reviews, conflicting labels, and correlations between review length and sentiment-related targets.

These findings directly influenced the preprocessing and splitting decisions and establish the experimental context for the subsequent modeling stage.