import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import DATA_PATH, FIGURES_DIR, TARGET

sns.set_theme(style="whitegrid")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Place the dataset at {DATA_PATH}")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, na_values=["N/A", "NA", ""])

    # 1. Target balance
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET)
    plt.title("Recorded Stroke Outcome Counts")
    plt.xlabel("Stroke (0 = No, 1 = Yes)")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_target_distribution.png", dpi=200)
    plt.close()

    # 2. Age distribution by outcome
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="age", hue=TARGET, bins=30,
                 stat="density", common_norm=False, element="step")
    plt.title("Age Distribution by Recorded Stroke Outcome")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_age_distribution.png", dpi=200)
    plt.close()

    # 3. Glucose distribution by outcome
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=TARGET, y="avg_glucose_level")
    plt.title("Average Glucose Level by Stroke Outcome")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_glucose_boxplot.png", dpi=200)
    plt.close()

    # 4. BMI distribution (missing values excluded for this plot only)
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="bmi", bins=30, kde=True)
    plt.title("BMI Distribution (Observed Values)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_bmi_distribution.png", dpi=200)
    plt.close()

    # 5. Hypertension vs stroke
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="hypertension", hue=TARGET)
    plt.title("Hypertension Status by Recorded Stroke Outcome")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_hypertension_by_stroke.png", dpi=200)
    plt.close()

    # 6. Smoking status
    plt.figure(figsize=(9, 5))
    sns.countplot(data=df, x="smoking_status", hue=TARGET)
    plt.title("Smoking Status by Recorded Stroke Outcome")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_smoking_by_stroke.png", dpi=200)
    plt.close()

    # 7. Numeric correlation heatmap
    numeric = df.select_dtypes(include="number").drop(columns=["id"], errors="ignore")
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm",
                fmt=".2f", center=0)
    plt.title("Correlation Heatmap of Numeric Variables")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "07_correlation_heatmap.png", dpi=200)
    plt.close()

    print(f"EDA charts saved in: {FIGURES_DIR}")
    print("\nSuggested analysis: describe patterns visible in the charts, "
          "but do not claim that correlation proves causation.")

if __name__ == "__main__":
    main()
