# 🛡️ Credit Card Fraud Detection — Streamlit App

A professional, hackathon-ready web application that predicts whether a credit-card transaction is **legitimate** or **fraudulent** using a pre-trained machine-learning model.

---

## 📂 Project Structure

```
Samatrix/
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── creditcard.csv         # Raw dataset (not used at runtime)
├── fraud_model.pkl        # Trained classifier (you provide)
├── scaler.pkl             # Fitted scaler (you provide)
├── feature_names.pkl      # List of feature column names (you provide)
├── scale_cols.pkl         # List of columns to scale (you provide)
└── selector.pkl           # Feature selector — optional (you provide)
```

## ⚙️ Prerequisites

- Python 3.9 or newer
- The following `.pkl` files placed in the project folder:
  - `fraud_model.pkl`
  - `scaler.pkl`
  - `feature_names.pkl`
  - `scale_cols.pkl`
  - `selector.pkl` *(optional)*

---

## 🚀 Run Locally

```bash
# 1. Clone / download the project
cd Samatrix

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your .pkl files in the project folder

# 5. Launch the app
streamlit run app.py
```

The app will open in your browser at **http://localhost:8501**.

---

## ☁️ Deploy on Streamlit Community Cloud

1. **Push to GitHub** — Create a public (or private) GitHub repository containing:
   - `app.py`
   - `requirements.txt`
   - `fraud_model.pkl`
   - `scaler.pkl`
   - `feature_names.pkl`
   - `scale_cols.pkl`
   - `selector.pkl` *(if used)*

2. **Sign in** to [Streamlit Community Cloud](https://share.streamlit.io).

3. **New app** → select your repository, branch, and set the main file path to `app.py`.

4. Click **Deploy**. Streamlit will install the requirements and start the app automatically.

> **Tip:** If your `.pkl` files are too large for GitHub, consider using Git LFS or hosting them externally and downloading at startup.

---

## 🧪 How It Works

1. The user enters transaction features (Time, V1–V28, Amount).
2. The app engineers `Hour` and `Amount_log` from the raw inputs.
3. Columns are reordered to match `feature_names.pkl`.
4. Designated columns are scaled using `scaler.pkl`.
5. If `selector.pkl` is present, feature selection is applied.
6. The model predicts the class and (if available) outputs a fraud probability.

---

## 📜 License

This project is for educational and demonstration purposes.
