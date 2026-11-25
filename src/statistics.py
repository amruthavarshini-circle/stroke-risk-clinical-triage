import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from src.config import DATA_PATH, TARGET

def chi_square_report(df, feature, target=TARGET):
    table = pd.crosstab(df[feature], df[target])
    chi2, p_value, dof, expected = chi2_contingency(table)
    print(f"\nCHI-SQUARE: {feature} vs {target}")
    print(table)
    print(f"Chi-square statistic = {chi2:.4f}")
    print(f"Degrees of freedom = {dof}")
    print(f"P-value = {p_value:.6g}")
    print("Expected frequencies:")
    print(pd.DataFrame(expected, index=table.index, columns=table.columns).round(2))
    print("Decision at alpha=0.05:",
          "Evidence of association (reject H0)" if p_value < 0.05
          else "Insufficient evidence of association (fail to reject H0)")
    return table, chi2, p_value, dof, expected

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Place the dataset at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, na_values=["N/A", "NA", ""])

    # H0: hypertension and recorded stroke outcome are independent.
    chi_square_report(df, "hypertension")

    # Chi-Square needs categorical variables. Create quartiles from observed
    # glucose values for an exploratory association test.
    glucose_data = df[["avg_glucose_level", TARGET]].dropna().copy()
    glucose_data["glucose_quartile"] = pd.qcut(
        glucose_data["avg_glucose_level"], q=4,
        labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"],
        duplicates="drop"
    )
    chi_square_report(glucose_data, "glucose_quartile")

    # Glucose is continuous; compare its distributions between the two
    # independent outcome groups with a non-parametric Mann-Whitney U test.
    no_stroke = df.loc[df[TARGET] == 0, "avg_glucose_level"].dropna()
    stroke = df.loc[df[TARGET] == 1, "avg_glucose_level"].dropna()
    u_stat, p_value = mannwhitneyu(no_stroke, stroke, alternative="two-sided")
    print("\nMANN-WHITNEY U: glucose distribution between outcome groups")
    print(f"U statistic = {u_stat:.4f}")
    print(f"P-value = {p_value:.6g}")
    print("This tests distributional differences, not causation.")

if __name__ == "__main__":
    main()
