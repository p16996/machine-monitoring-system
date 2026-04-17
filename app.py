import streamlit as st
import pandas as pd

st.set_page_config(page_title="Machine Monitoring System", layout="wide")

st.title("Machine Monitoring System")
st.subheader("AI-powered machine failure prediction and monitoring")

st.write(
    "This web app uses machine learning to predict machine failure risk "
    "from operational data such as temperature, rotational speed, torque, and tool wear."
)
# Load dataset
df = pd.read_csv("data/predictive_maintenance.csv")
st.markdown("## Dataset Overview")
# Dataset shape
rows, cols = df.shape
col1, col2 = st.columns(2)
col1.metric("Rows", rows)
col2.metric("Columns", cols)

# Preview
st.markdown("### Data Preview")
st.dataframe(df.head())

# Column names
st.markdown("### Columns")
st.write(df.columns.tolist())

# Missing values
st.markdown("### Missing Values")
missing_df = df.isnull().sum().reset_index()
missing_df.columns = ["Column", "Missing Values"]
st.dataframe(missing_df)

# Target distribution
st.markdown("### Target Distribution: Machine Failure")
target_counts = df["Machine failure"].value_counts().reset_index()
target_counts.columns = ["Machine failure", "Count"]
st.dataframe(target_counts)

st.bar_chart(df["Machine failure"].value_counts())

# Notes
st.markdown("### Notes")
st.write("- Target column: `Machine failure`")
st.write("- Likely columns to drop later: `UDI`, `Product ID`")
st.write("- Categorical column to encode later: `Type`")

st.markdown("### Initial Interpretation")
st.write(
    "The dataset is used for a binary classification problem where the goal is to predict "
    "whether a machine will fail. The target appears imbalanced because failures are much less frequent "
    "than non-failures."
)