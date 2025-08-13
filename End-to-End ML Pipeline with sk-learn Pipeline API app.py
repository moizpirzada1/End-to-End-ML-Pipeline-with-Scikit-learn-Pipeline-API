import streamlit as st
import pandas as pd
import joblib
from io import StringIO
import os

# Path to model file
MODEL_PATH = "models/churn_pipeline.pkl"  # Adjust as needed

@st.cache_resource
def load_model():
    """Load the trained model from disk."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    if not hasattr(model, "feature_names_in_"):
        st.error("Model does not contain feature name metadata. Cannot auto-generate form.")
        st.stop()

    return model

# Single input form (generated from model's feature names)
def single_input_form(feature_names):
    st.subheader("Single Prediction Input")
    input_data = {}
    for feat in feature_names:
        input_data[feat] = st.text_input(feat, value="")
    return input_data

# Prediction functions
def predict_single(model, row):
    df = pd.DataFrame([row])
    return model.predict(df)[0]

def predict_batch(model, df):
    preds = model.predict(df)
    df["prediction"] = preds
    return df

# Main app
def main():
    st.title("ML Pipeline Prediction App")

    model = load_model()
    feature_names = list(model.feature_names_in_)

    tab1, tab2 = st.tabs(["Single Input", "Batch CSV Upload"])

    # Single Prediction
    with tab1:
        row = single_input_form(feature_names)
        if st.button("Predict Single"):
            try:
                df_row = pd.DataFrame([row])
                # Convert to numeric where possible
                for col in df_row.columns:
                    df_row[col] = pd.to_numeric(df_row[col], errors="ignore")
                result = predict_single(model, df_row.iloc[0])
                st.success(f"Prediction: {result}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # Batch Prediction
    with tab2:
        st.subheader("Upload CSV for Batch Predictions")
        st.caption(f"Required columns: {feature_names}")
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                missing_cols = [col for col in feature_names if col not in df.columns]

                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                else:
                    result_df = predict_batch(model, df)
                    st.dataframe(result_df)

                    csv_buf = StringIO()
                    result_df.to_csv(csv_buf, index=False)
                    st.download_button(
                        "Download Predictions",
                        data=csv_buf.getvalue(),
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

if __name__ == "__main__":
    main()
