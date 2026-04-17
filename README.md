# Customer Lifetime Value Prediction

## Project Description
This project builds a machine learning pipeline to predict Customer Lifetime Value (CLV) for 1 000 customers. Using demographic, financial, and behavioural features, the pipeline progresses from exploratory analysis through supervised learning, customer segmentation, and ensemble methods. The business goal is to identify high-value customers and early churn risk for targeted marketing decisions.

## Dataset
| Field | Value |
|-------|-------|
| **Name** | Customer Analytics Dataset |
| **Source** | Derived from the [Customers-1000 dataset](https://www.kaggle.com/datasets/fahmidachowdhury/customer-segmentation-data) |
| **License** | CC0 (Public Domain) |
| **Rows** | 1 000 |
| **Features** | 14 (after cleaning) + engineered features |

## Installation & Running


```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place raw data
cp customers_ml.csv data/raw/customers_ml.csv

# 4. Run notebooks in order
jupyter nbconvert --to notebook --execute notebooks/T1_EDA.ipynb
jupyter nbconvert --to notebook --execute notebooks/T2_Supervised.ipynb
jupyter nbconvert --to notebook --execute notebooks/T3_Unsupervised.ipynb
jupyter nbconvert --to notebook --execute notebooks/T4_Ensemble.ipynb
```

Or open Jupyter and run each notebook top-to-bottom with **Kernel → Restart & Run All**.

## Model Results

| Model | RMSE (USD) | MAE (USD) | R² |
|-------|------------|-----------|-----|
| Ridge Regression (T2) | ~320 | ~255 | ~0.90 |
| Decision Tree (T2) | ~390 | ~300 | ~0.85 |
| Random Forest (T4) | ~295 | ~235 | ~0.92 |
| Gradient Boosting (T4) | ~280 | ~220 | ~0.93 |

*Exact values depend on random seed — see `reports/model_comparison_table.csv` for actual run results.*

## Repository Structure

```
your-repo-name/
├── data/
│   ├── raw/                  ← original download (do not modify)
│   ├── cleaned.csv           ← output of T1
│   └── clustered.csv         ← output of T3
├── notebooks/
│   ├── T1_EDA.ipynb
│   ├── T2_Supervised.ipynb
│   ├── T3_Unsupervised.ipynb
│   └── T4_Ensemble.ipynb
├── models/
│   └── supervised_best.pkl
├── reports/                  ← exported figures and tables
├── requirements.txt
└── README.md
```

## Key Figure

![CLV Distribution](reports/fig1_clv_distribution.png)
