import json
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from src.config import (
    DATA_PATH, FEATURES, TARGET, NUMERIC_FEATURES,
    CATEGORICAL_FEATURES, MODELS_DIR, MODEL_PATH, METRICS_PATH
)

def build_preprocessor():
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

def evaluate(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, zero_division=0, output_dict=True
        )
    }

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Place the dataset at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, na_values=["N/A", "NA", ""])
    X = df[FEATURES].copy()  # id is intentionally excluded
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        )
    }

    results = []
    fitted = {}
    for name, classifier in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier)
        ])
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        result = evaluate(name, pipeline, X_test, y_test)
        results.append(result)
        print(f"\n--- {name} ---")
        for metric in ["accuracy", "precision", "recall", "f1",
                       "roc_auc", "average_precision"]:
            print(f"{metric}: {result[metric]:.4f}")
        print("Confusion matrix:")
        print(result["confusion_matrix"])

    # The assigned problem explicitly names Logistic Regression.
    # Save that model as the required, interpretable baseline application model.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted["Logistic Regression"], MODEL_PATH)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved Logistic Regression pipeline: {MODEL_PATH}")
    print(f"Saved evaluation metrics: {METRICS_PATH}")
    print("\nImportant: do not select a model using accuracy alone. "
          "Discuss recall, precision, F1, ROC-AUC and average precision, "
          "especially because the positive class is rare.")

if __name__ == "__main__":
    main()
