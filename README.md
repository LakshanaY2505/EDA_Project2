# EDA_Project2

# Churn Prediction Project

Predict which telecom customers are likely to churn using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Business objective
Customer acquisition costs 5–7× more than retention. By identifying high-risk customers early, the retention team can act proactively.

## ML objective
Binary classification: predict `Churn` (1 = churned, 0 = stayed).

## Folder structure
```
churn-project/
├── data/
│   ├── raw/                   # Original CSV from Kaggle (do not edit)
│   └── processed/
│       ├── cleaned_churn.csv  # After cleaning (notebook 02)
│       └── featured_churn.csv # Model-ready (notebook 04)
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_churn_eda.ipynb
│   └── 04_feature_engineering.ipynb
├── images/                    # All saved plots
├── src/
│   ├── data_cleaning.py       # clean_data()
│   ├── eda_plots.py           # plot_all() and individual plot functions
│   └── feature_engineering.py # build_features(), split_X_y()
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
jupyter notebook
```

## Dataset
| Column | Description |
|---|---|
| customerID | Unique identifier (dropped) |
| gender | Male / Female |
| SeniorCitizen | 0 or 1 |
| Partner, Dependents | Yes / No |
| tenure | Months with company |
| PhoneService … StreamingMovies | Service subscriptions |
| Contract | Month-to-month / One year / Two year |
| PaperlessBilling | Yes / No |
| PaymentMethod | 4 options including e-check |
| MonthlyCharges, TotalCharges | Billing amounts |
| **Churn** | **Target: Yes / No → 1 / 0** |

## Key EDA findings
- Month-to-month contracts churn at ~43% vs 3% for two-year contracts.
- Churned customers have median tenure of ~10 months vs ~38 months for stayers.
- Churned customers pay more per month on average (~$74 vs $61).
- Electronic check users churn at ~45% — highest of any payment method.
- Tenure and TotalCharges are highly correlated (r > 0.8); TotalCharges dropped.

## Engineered features
| Feature | Description |
|---|---|
| `tenure_bucket` | new (0–12 mo) / mid (13–48 mo) / long (49+ mo) |
| `high_charges` | 1 if MonthlyCharges > median |
| `num_services` | Count of add-on services (0–6) |

## Class imbalance
~26.5% churned / ~73.5% stayed. Use **ROC-AUC or F1** as evaluation metric, not accuracy.