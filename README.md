# Porter Delivery Time Prediction with Linear Regression

A regression project that estimates delivery time at the moment an order is placed, so Porter, India's largest intra-city logistics marketplace, can show customers an accurate ETA and understand what causes delays. Linear regression was chosen because the business needs to know which factors add or remove minutes.

## Business problem

- Show customers an accurate **ETA** when an order is placed.
- Understand **which factors add or remove minutes** from delivery time.
- **Target:** delivery duration in minutes (continuous), so this is a **regression** problem.
- **Success metrics:** R², MAE, RMSE.

## Data

About 197,000 orders. The delivery duration column did not exist, so it was **engineered**:

`delivery_minutes = (actual_delivery_time − created_at)` converted to minutes

Rows with missing timestamps were dropped, and impossible durations (negative values, or above 4 hours) were removed.

> **Data source:** _add where the dataset comes from (for example, a public dataset link)._

## Approach

1. **Target creation and sanity filtering** as described above.
2. **EDA:** the target is right-skewed (skewness 1.874). Most numeric features are also right-skewed. `store_primary_category` has 74 categories, and the top 20 cover about 86% of orders.
3. **Outliers:** capped with the IQR method on the **features only**. The target was not capped, because slow deliveries are real cases the model needs to learn from.
4. **Missing values:** mode for categorical columns, KNN imputation for numeric columns.
5. **Feature engineering**
   - Time features: hour, day of week, weekend flag, part of day.
   - Load features: `busy_ratio` (busy partners ÷ on-shift partners) and `orders_per_free_partner` (outstanding orders ÷ (free partners + 1)). They capture network load relative to capacity.
   - `store_id` frequency-encoded. Categories grouped (top 20 plus "other") and one-hot encoded.
6. **Statistical tests:** ANOVA, t-test and chi-square to check which categorical features matter.
7. **Split and scaling:** 60/20/20 train/validation/test, `StandardScaler` fitted on the training data only.
8. **Assumption checks:** VIF, Breusch-Pagan, Q-Q plot.
9. **Model and validation:** OLS, 5-fold cross-validation, Ridge, Lasso and Elastic Net.

## Results

### Model performance

| Metric | Value |
|---|---|
| R² (train / validation / test) | 0.1965 / 0.2031 / 0.1956 |
| 5-fold CV R² | mean 0.198 (std 0.005) |
| Test MAE | about 12.0 minutes |
| Test RMSE | about 16.4 minutes |

The scores are almost identical on every split, so the model is **not overfitting**. The low scores mean it **underfits**: the available features carry little signal.

### Assumption checks

- **Multicollinearity:** the partner-count features correlate at about 0.94 (VIF flagged them).
- **Heteroscedasticity:** Breusch-Pagan detected non-constant variance.
- **Normality:** the Q-Q plot shows right-skewed residuals.

### Regularization

| Model | Test R² |
|---|---|
| Linear regression | about 0.20 |
| Ridge | about 0.20 (matches linear regression) |
| Lasso (alpha = 1.0) | about 0.09 to 0.10 (zeroed 41 of 48 features) |
| Elastic Net (alpha = 1.0) | about 0.09 to 0.10 |

Regularization fixes overfitting, and this model was not overfitting, so it could not help. The default alpha was too strong for Lasso, so it should be tuned with cross-validation.

## Key findings

1. **No single feature correlates with delivery time above 0.21.** Delay comes from many small effects.
2. **Network load matters more than order details.** The engineered load ratios were stronger predictors than the raw partner counts.
3. **`order_protocol` has a visible effect** (protocol 6 slower, protocol 7 faster). `market_id` has almost none.
4. **About 80% of the variation is unexplained**, probably because traffic, weather, distance and driver behaviour are not in the data.

## Business takeaway

The model gives a **reliable baseline ETA for typical deliveries, typically within about 12 minutes**. It is less reliable during busy periods and for unusually slow orders, because predictions are compressed toward the middle and very slow deliveries are underpredicted. Porter should **add a time buffer at peak hours** and should not use the model alone to flag at-risk deliveries.

## Limitations and next steps

- Log-transform the target to reduce skew and heteroscedasticity (predictions must then be back-transformed).
- Tune alpha with `LassoCV` and `RidgeCV`.
- Drop or combine redundant, collinear features.
- Add distance, traffic and weather data.
- Benchmark a tree-based model (Random Forest or XGBoost) to capture interactions, such as a high backlog combined with few free partners.
- Fit imputers and encoders on the training data only, ideally inside a scikit-learn `Pipeline`, to avoid leakage.

## How to run

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn
```

1. Put the dataset CSV in the same folder as the notebook and update the file path in the first cells.
2. Run all cells: `jupyter notebook <your_notebook_name>.ipynb`

## Repository contents

| File | Description |
|---|---|
| `<your_notebook_name>.ipynb` | Full analysis, code and outputs |
| `<dataset_name>.csv` | Dataset |
| `README.md` | This file |
