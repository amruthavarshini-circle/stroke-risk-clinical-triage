# Stroke Risk Factors & Clinical Triage App

**Course:** Introduction to Data Science (25AI2104)  
**Project allocation:** No. 49  
**Dataset:** Stroke Prediction Dataset (Kaggle)

## Problem statement
Impute BMI and smoking attributes, investigate hypertension/glucose relationships using Chi-Square analysis and Logistic Regression, and deploy a Streamlit stroke-risk screening application.

## Important medical limitation
This is an educational data-science demonstration using a historical dataset. It is not a medical device, diagnostic system, or substitute for professional medical advice. Predictions are uncertain and may be affected by dataset bias, class imbalance, and limitations in the source data.

## Dataset
Download the allocated dataset from:
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

Place the CSV at:
`data/raw/healthcare-dataset-stroke-data.csv`

Do not upload the raw CSV to GitHub. The supplied college Git workflow guide says to exclude CSV/XLSX files and large datasets.

## Ubuntu / VS Code setup
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip git -y
mkdir -p ~/Stroke-Risk-Project
cd ~/Stroke-Risk-Project
# Copy/extract this project into this folder
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the analysis
```bash
python -m src.data_audit
python -m src.eda
python -m src.statistics
python -m src.train_model
```

Then launch the application:
```bash
streamlit run app.py
```

## Project stages
1. Dataset audit and data dictionary
2. Data cleaning and missing-value analysis
3. Exploratory data analysis
4. Chi-Square statistical tests
5. Feature preprocessing and model training
6. Model evaluation
7. Streamlit application
8. Report, presentation, GitHub and viva

## Interpretation notes
- `stroke` is the binary target (0 = no recorded stroke, 1 = recorded stroke).
- `id` is an identifier and is excluded from model features.
- `Unknown` in `smoking_status` is an explicit category in this dataset, not automatically a null. The pipeline preserves it as a category; actual null values are imputed using the most frequent training category.
- BMI missing values are imputed with the median learned from the training data only.
- The target is imbalanced. Accuracy alone is not an adequate evaluation metric. Report recall, precision, F1, ROC-AUC, average precision and the confusion matrix.
- A Chi-Square test requires categorical variables. For glucose, the analysis script creates data-derived quartile groups for an exploratory independence test. Logistic Regression also models glucose as a continuous predictor.
- Statistical association does not prove causation.

## Folder layout
```text
data/raw/             Original downloaded CSV (not committed)
data/processed/       Optional cleaned data (not committed)
notebooks/            Guided Jupyter notebooks
src/                  Reusable analysis and model code
models/               Trained model artifact (not committed by default)
reports/figures/      Generated charts
docs/                 Final report PDF and PPT
app.py                Streamlit application
```
