import pandas as pd
from src.config import DATA_PATH, TARGET

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download the allocated CSV and place it in data/raw/."
        )

    df = pd.read_csv(DATA_PATH, na_values=["N/A", "NA", ""])
    print("\n--- DATASET SHAPE ---")
    print(df.shape)
    print("\n--- FIRST FIVE ROWS ---")
    print(df.head())
    print("\n--- COLUMN TYPES ---")
    print(df.dtypes)
    print("\n--- MISSING VALUES ---")
    print(df.isna().sum().sort_values(ascending=False))
    print("\n--- DUPLICATE ROWS ---")
    print(df.duplicated().sum())
    print("\n--- TARGET COUNTS ---")
    print(df[TARGET].value_counts(dropna=False))
    print("\n--- TARGET PROPORTIONS ---")
    print(df[TARGET].value_counts(normalize=True, dropna=False).round(4))
    print("\n--- CATEGORICAL VALUES ---")
    for col in df.select_dtypes(include="object").columns:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False))

if __name__ == "__main__":
    main()
