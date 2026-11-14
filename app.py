import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Computer Price Predictor", page_icon="💻")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_options():
    df = pd.read_csv("Computer_Data.csv")
    return {c: sorted(df[c].unique().tolist()) for c in ["speed", "ram", "hd", "screen"]}

model = load_model()
opts = load_options()

st.title("💻 Computer Price Predictor")
st.write("Select the specs and get an estimated price.")

col1, col2 = st.columns(2)
with col1:
    speed = st.selectbox("Speed (MHz)", opts["speed"])
    ram = st.selectbox("RAM (MB)", opts["ram"])
with col2:
    hd = st.selectbox("Hard Disk (MB)", opts["hd"])
    screen = st.selectbox("Screen (inches)", opts["screen"])

if st.button("Predict Price"):
    X_new = pd.DataFrame([[speed, ram, hd, screen]],
                         columns=["speed", "ram", "hd", "screen"])
    price = model.predict(X_new)[0]
    st.success(f"Estimated price: ${price:,.0f}")
