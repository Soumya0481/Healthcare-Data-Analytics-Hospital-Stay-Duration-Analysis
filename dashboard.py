import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Healthcare Dashboard", layout="wide")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "healthcare_dataset.csv")

    if not os.path.exists(data_path):
        st.error(f"Dataset not found at: {data_path}")
        st.stop()

    df = pd.read_csv(data_path)

    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    df['Hospital Stay (Days)'] = (
        df['Discharge Date'] - df['Date of Admission']
    ).dt.days

    df['Age Group'] = pd.cut(
        df['Age'],
        bins=[0, 18, 35, 50, 65, 100],
        labels=['0-18', '19-35', '36-50', '51-65', '65+']
    )

    return df
