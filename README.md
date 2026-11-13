# Melbourne Housing Price Analysis

End-to-end analysis of the Melbourne Housing Snapshot dataset (Kaggle): data cleaning, SQL analysis, clustering, visualization and regression modeling.
The regression part is the basis of the paper **"The Impact of Suburb-Level Encoding on Melbourne Housing Price Prediction"** (SCI-2026, accepted / under publication; see `Realestate (3).pdf`).

## Notebooks

| # | Notebook | Content |
|---|---|---|
| 1 | Business Understanding and Analytic Approach | Problem statement and plan |
| 2 | Data Collection, Understanding, Preparation | Cleaning (13,580 -> 10,480 rows), encoding, outlier removal |
| 3 | Data Analysis with SQL | SQL queries on `melb_real_estate.db` |
| 4 | Data Analysis with Python | KMeans, PCA and exploratory analysis |
| 5 | Data Visualization | Charts of price vs. property features |
| 6 | Regression Analysis | 7 models, suburb ablation, shuffled-suburb control, SHAP |

## Method (notebook 6)

- Target: `log(Price)`; 80/20 train/test split, 5-fold CV on the training set (`random_state=42`).
- Suburb handled with target encoding (smoothing=10, min_samples_leaf=20), fitted inside each fold to avoid leakage.
- Models: Linear, Ridge, Lasso, Random Forest, Extra Trees, XGBoost, LightGBM.
- Locality ablation with Random Forest: without suburb / with suburb / shuffled suburb.

## Results reported in the paper (log-price scale)

| Model | 5-fold CV R^2 |
|---|---|
| Linear / Ridge | 0.736 |
| Lasso | 0.729 |
| Random Forest | 0.819 |
| Extra Trees | 0.808 |
| XGBoost / LightGBM | 0.843 |

| Random Forest setting | Train R^2 | Test R^2 |
|---|---|---|
| Without Suburb | 0.847 | 0.815 |
| With Suburb | 0.871 | 0.835 |
| Shuffled Suburb | 0.851 | 0.810 |

SHAP on the XGBoost model ranks Suburb, Distance, Landsize and Rooms as the most influential features.

> Reproducibility note: re-running the code in this repository gives values within about 0.006 R^2 of the table above (for example XGBoost CV R^2 0.847, Random Forest test R^2 0.809 without and 0.830 with suburb). The conclusions are unchanged. The shuffle test uses `np.random.permutation` without a fixed seed, so it varies slightly between runs.

## Run

```bash
pip install pandas numpy scikit-learn category_encoders xgboost lightgbm shap matplotlib seaborn jupyter
jupyter notebook
```
Run the notebooks in order; notebook 6 reads `melb_data_cleaned.csv`.

## Files

- `melb_data.csv`: raw data (13,580 rows)
- `melb_data_cleaned.csv`: cleaned data (10,480 rows)
- `melb_real_estate.db`: SQLite copy of the cleaned data
