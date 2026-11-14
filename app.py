import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sales Prediction", page_icon="📈", layout="centered")


@st.cache_resource
def load_model(path: str = "Sales_prediction_model.pkl"):
    return joblib.load(path)


FEATURES = ["youtube", "facebook", "newspaper"]

st.title("📈 Sales Prediction")
st.write("Enter your advertising budgets to predict sales.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model 'Sales_prediction_model.pkl': {e}")
    st.stop()

tab_single, tab_batch = st.tabs(["Single prediction", "Batch (CSV)"])

# ---------- Single prediction ----------
with tab_single:
    col1, col2, col3 = st.columns(3)
    youtube = col1.number_input("YouTube", min_value=0.0, value=100.0, step=1.0)
    facebook = col2.number_input("Facebook", min_value=0.0, value=30.0, step=1.0)
    newspaper = col3.number_input("Newspaper", min_value=0.0, value=20.0, step=1.0)

    if st.button("Predict Sales", type="primary"):
        X = pd.DataFrame([[youtube, facebook, newspaper]], columns=FEATURES)
        pred = model.predict(X)
        value = float(pred.ravel()[0])
        st.success(f"Predicted Sales: **{value:,.2f}**")

    with st.expander("Model details"):
        if hasattr(model, "coef_"):
            coefs = pd.DataFrame(
                {"feature": FEATURES, "coefficient": list(model.coef_.ravel())}
            )
            st.dataframe(coefs, hide_index=True)
            st.write(f"Intercept: {float(pd.Series(model.intercept_).iloc[0]):.4f}")

# ---------- Batch prediction ----------
with tab_batch:
    st.write(f"Upload a CSV with columns: `{', '.join(FEATURES)}`")
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        df.columns = [c.strip().lower() for c in df.columns]
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            df["predicted_sales"] = model.predict(df[FEATURES]).ravel()
            st.dataframe(df)
            st.download_button(
                "Download predictions",
                df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
