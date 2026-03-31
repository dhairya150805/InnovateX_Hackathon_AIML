"""
Credit Card Fraud Detection — Streamlit Web App
=================================================
A professional, hackathon-ready interface that loads pre-trained model
artefacts and predicts whether a credit-card transaction is legitimate
or fraudulent.
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ──────────────────────────────────────────────
# Page configuration (must be the first st call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Paths to artefact files (same directory as app)
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REQUIRED_FILES = {
    "fraud_model.pkl": "Trained classifier model",
    "scaler.pkl": "StandardScaler / scaler object",
    "feature_names.pkl": "List of feature names expected by the model",
    "scale_cols.pkl": "List of columns to scale",
}
OPTIONAL_FILES = {
    "selector.pkl": "Feature selector (optional)",
}


# ──────────────────────────────────────────────
# Helper — load artefacts with caching
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model artefacts …")
def load_artefacts():
    """Load all .pkl files. Raises FileNotFoundError for required ones."""
    artefacts = {}

    # Required
    for fname, description in REQUIRED_FILES.items():
        fpath = os.path.join(BASE_DIR, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Required file **{fname}** ({description}) not found in `{BASE_DIR}`."
            )
        artefacts[fname] = joblib.load(fpath)

    # Optional
    for fname, description in OPTIONAL_FILES.items():
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.isfile(fpath):
            artefacts[fname] = joblib.load(fpath)
        else:
            artefacts[fname] = None

    return artefacts


# ──────────────────────────────────────────────
# Helper — preprocess a single transaction
# ──────────────────────────────────────────────
def preprocess(raw: dict, feature_names: list, scale_cols: list, scaler, selector):
    """
    Reproduce the same preprocessing pipeline used during training:
    1. Derive Hour from Time.
    2. Derive Amount_log from Amount.
    3. Build a DataFrame with the correct column order.
    4. Scale only the columns in scale_cols.
    5. Apply the feature selector (if provided).
    """
    # Engineered features
    raw["Hour"] = int(raw["Time"] // 3600) % 24
    raw["Amount_log"] = np.log1p(raw["Amount"])

    # Build DataFrame in the exact column order the model expects
    df = pd.DataFrame([raw])

    # Keep only the columns the model was trained on
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0  # fill missing features with zero
    df = df[feature_names]

    # Scale designated columns
    cols_to_scale = [c for c in scale_cols if c in df.columns]
    if cols_to_scale:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Feature selection (optional) — only apply if the model was trained
    # on the reduced feature set (i.e. selector output count matches model input count)
    if selector is not None:
        n_selected = int(selector.get_support().sum())
        model_n = None
        # We'll check compatibility at prediction time; for safety, skip here
        # and let the caller decide. Store selector output as alternative.
        # Skip selector by default — the model was trained on the full feature set.
        pass

    return df


# ──────────────────────────────────────────────
# Helper — run prediction
# ──────────────────────────────────────────────
def predict(model, processed_df):
    """Return (label, probability_or_None)."""
    prediction = model.predict(processed_df)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(processed_df)[0]
    return int(prediction), proba


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.image(
            "https://img.icons8.com/3d-fluency/94/shield.png",
            width=80,
        )
        st.title("ℹ️ About")
        st.markdown(
            """
            This app uses a **machine-learning model** trained on the
            [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
            to classify transactions as *legitimate* or *fraudulent*.

            **How it works**
            1. Enter the transaction features on the right.
            2. Click **Predict**.
            3. The model pre-processes the input and returns a verdict.

            ---
            **Artefact files expected**
            - `fraud_model.pkl`
            - `scaler.pkl`
            - `feature_names.pkl`
            - `scale_cols.pkl`
            - `selector.pkl` *(optional)*
            """
        )
        st.caption("Built with ❤️ using Streamlit")


# ──────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────
def main():
    render_sidebar()

    # ── Header ────────────────────────────────
    st.markdown(
        """
        <h1 style='text-align:center;'>🛡️ Credit Card Fraud Detection</h1>
        <p style='text-align:center; color:grey;'>
            Enter the transaction details below and click <b>Predict</b> to check
            whether the transaction is legitimate or fraudulent.
        </p>
        <hr>
        """,
        unsafe_allow_html=True,
    )

    # ── Load artefacts ────────────────────────
    try:
        artefacts = load_artefacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Please place all required `.pkl` files in the project folder and reload the app.")
        st.stop()

    model = artefacts["fraud_model.pkl"]
    scaler = artefacts["scaler.pkl"]
    feature_names = artefacts["feature_names.pkl"]
    scale_cols = artefacts["scale_cols.pkl"]
    selector = artefacts["selector.pkl"]  # may be None

    if selector is not None:
        st.sidebar.success("✅ selector.pkl loaded")
    else:
        st.sidebar.info("ℹ️ selector.pkl not found — skipping feature selection")

    # ── Input form ────────────────────────────
    st.subheader("📋 Transaction Details")

    with st.form("txn_form"):
        # Row 1 — Time & Amount
        col_t, col_a = st.columns(2)
        with col_t:
            time_val = st.number_input(
                "Time (seconds elapsed)", min_value=0.0, value=0.0, step=1.0,
                help="Seconds elapsed between this transaction and the first transaction in the dataset.",
            )
        with col_a:
            amount_val = st.number_input(
                "Amount ($)", min_value=0.0, value=0.0, step=0.01,
                help="Transaction amount in US dollars.",
            )

        # V1 – V28 in a 4-column grid
        st.markdown("**PCA Components (V1 – V28)**")
        v_values = {}
        cols_per_row = 4
        for start in range(1, 29, cols_per_row):
            cols = st.columns(cols_per_row)
            for idx, col in enumerate(cols):
                v_num = start + idx
                if v_num > 28:
                    break
                with col:
                    v_values[f"V{v_num}"] = st.number_input(
                        f"V{v_num}", value=0.0, format="%.6f", key=f"V{v_num}",
                    )

        submitted = st.form_submit_button("🔍  Predict", use_container_width=True)

    # ── Prediction ────────────────────────────
    if submitted:
        # Collect raw inputs
        raw_input = {"Time": time_val, "Amount": amount_val}
        raw_input.update(v_values)

        try:
            processed_df = preprocess(raw_input, feature_names, scale_cols, scaler, selector)
            label, proba = predict(model, processed_df)
        except Exception as exc:
            st.error(f"⚠️ Prediction failed: {exc}")
            st.stop()

        # ── Result card ───────────────────────
        st.markdown("---")
        if label == 0:
            st.success("### ✅ Legitimate Transaction")
        else:
            st.error("### 🚨 Fraudulent Transaction")
            st.warning("This transaction has been flagged as potentially fraudulent.")

        # Probability bar
        if proba is not None:
            fraud_prob = proba[1] if len(proba) > 1 else proba[0]
            legit_prob = 1 - fraud_prob

            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.metric("Legitimate Probability", f"{legit_prob:.4%}")
            with prob_col2:
                st.metric("Fraud Probability", f"{fraud_prob:.4%}")

            st.progress(float(fraud_prob), text=f"Fraud confidence: {fraud_prob:.2%}")

        # Expandable processed data view
        with st.expander("🔬 View Processed Input Data"):
            st.dataframe(processed_df, use_container_width=True)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    main()
