import streamlit as st
import pandas as pd
import joblib
import json
from io import StringIO

MODEL_PATH = "model.pkl"
SCHEMA_PATH = "schema.json"  # saved from train.py with feature metadata

@st.cache_resource
def load_model_and_schema():
    model = joblib.load(MODEL_PATH)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return model, schema

def single_input_form(schema):
    st.subheader("Single Prediction Input")
    input_data = {}
    for feat in schema["features"]:
        name = feat["name"]
        ftype = feat["type"]
        if ftype == "number":
            input_data[name] = st.number_input(name, value=float(feat.get("default", 0)))
        elif ftype == "integer":
            input_data[name] = st.number_input(name, value=int(feat.get("default", 0)), step=1)
        else:
            input_data[name] = st.text_input(name, value=str(feat.get("default", "")))
    return input_data

def predict_single(model, row):
    df = pd.DataFrame([row])
    return model.predict(df)[0]

def predict_batch(model, df):
    preds = model.predict(df)
    df["prediction"] = preds
    return df

def main():
    st.title("ML Pipeline Prediction App")

    try:
        model, schema = load_model_and_schema()
    except Exception as e:
        st.error(f"Error loading model/schema: {e}")
        st.stop()

    tab1, tab2 = st.tabs(["Single Input", "Batch CSV Upload"])

    with tab1:
        row = single_input_form(schema)
        if st.button("Predict Single"):
            try:
                result = predict_single(model, row)
                st.success(f"Prediction: {result}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with tab2:
        st.subheader("Upload CSV for Batch Predictions")
        st.caption(f"Required columns: {[f['name'] for f in schema['features']]}")
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                missing_cols = [f['name'] for f in schema['features'] if f['name'] not in df.columns]
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                else:
                    result_df = predict_batch(model, df)
                    st.dataframe(result_df)
                    csv_buf = StringIO()
                    result_df.to_csv(csv_buf, index=False)
                    st.download_button("Download Predictions", data=csv_buf.getvalue(),
                                       file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

if __name__ == "__main__":
    main()
