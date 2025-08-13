import argparse
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# -------------------------------------------------------
# Function to generate a schema.json file for Streamlit UI
# -------------------------------------------------------
def infer_feature_schema(df, target):
    schema = {"features": []}

    for col in df.columns:
        if col == target:
            continue  # Skip target column

        if pd.api.types.is_numeric_dtype(df[col]):  
            # Detect numeric type (float or int)
            ftype = "number" if not pd.api.types.is_integer_dtype(df[col]) else "integer"
            schema["features"].append({
                "name": col,
                "type": ftype,
                "default": float(df[col].mean()) if ftype == "number" else int(df[col].mode()[0]),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            })
        else:
            # Categorical column: store all unique values
            categories = df[col].dropna().unique().tolist()
            schema["features"].append({
                "name": col,
                "type": "categorical",
                "categories": categories,
                "default": categories[0] if categories else ""
            })

    # Store the target column name
    schema["target"] = target
    return schema

# -------------------------------------------------------
# Main function to train model and save artifacts
# -------------------------------------------------------
def main():
    # CLI arguments: dataset path and target column
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.data)

    # Split into features (X) and target (y)
    X = df.drop(columns=[args.target])
    y = df[args.target]

    # Identify numerical and categorical feature columns
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),  # Fill missing values with mean
        ("scaler", StandardScaler())                  # Standardize values
    ])

    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing with most frequent
        ("encoder", OneHotEncoder(handle_unknown="ignore"))    # Encode categories
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features)
        ]
    )

    # Create final pipeline: preprocessing + model
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))  # Change model here if needed
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate accuracy
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model trained. Accuracy: {acc:.4f}")

    # Save trained model
    joblib.dump(model, "model.pkl")

    # Generate and save feature schema for Streamlit
    schema = infer_feature_schema(df, args.target)
    with open("schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=4)

    print("Artifacts saved: model.pkl, schema.json")

# -------------------------------------------------------
# Script entry point
# -------------------------------------------------------
if __name__ == "__main__":
    main()
