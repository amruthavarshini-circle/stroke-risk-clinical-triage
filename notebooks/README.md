# Notebook workflow

Open VS Code, open the `notebooks` folder, and create Jupyter notebooks in this order:

1. `01_data_understanding.ipynb` — load the CSV, inspect shape, columns, dtypes, missing values, duplicates and target counts.
2. `02_data_cleaning.ipynb` — inspect missing BMI and smoking-status values; distinguish true nulls from the literal `Unknown` category. Do not delete rows without justification.
3. `03_exploratory_data_analysis.ipynb` — univariate, bivariate and multivariate charts. Save selected figures to `reports/figures/`.
4. `04_statistical_analysis.ipynb` — hypotheses, Chi-Square test for hypertension vs stroke, quartile-based glucose Chi-Square exploration, and an appropriate two-group glucose comparison.
5. `05_feature_engineering.ipynb` — explain numeric/categorical features, one-hot encoding, scaling and why preprocessing must be fitted on training data only.
6. `06_model_training_evaluation.ipynb` — stratified split, Logistic Regression, comparison model, evaluation metrics and error analysis.

The reusable commands in `src/` run the core pipeline. Use the notebooks to show your learning, reasoning, outputs and written interpretations. Do not submit notebooks with unexplained copied code.
