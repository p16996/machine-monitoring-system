import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

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

st.markdown("### Data Preparation for Modeling")

df_model = df.drop(columns=["UDI", "Product ID"])
df_model = pd.get_dummies(df_model, columns=["Type"], drop_first=True)

X = df_model.drop(columns=["Machine failure"])
y = df_model["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

st.markdown("### Prepared Dataset")

st.write("Feature columns:")
st.write(X.columns.tolist())

st.markdown("### Shapes")

col1, col2 = st.columns(2)
col1.metric("X shape", f"{X.shape[0]} rows, {X.shape[1]} cols")
col2.metric("y shape", f"{y.shape[0]} rows")

st.markdown("### Train/Test Split")

col3, col4 = st.columns(2)
col3.write(f"X_train: {X_train.shape}")
col4.write(f"X_test: {X_test.shape}")

st.write("y_train distribution:")
st.write(y_train.value_counts())

st.write("y_test distribution:")
st.write(y_test.value_counts())

st.markdown("### Interpretation")
st.write(
    "The dataset has been prepared for machine learning by removing identifiers, "
    "encoding categorical variables, and splitting into training and testing sets. "
    "Stratified sampling helps preserve the rare failure cases in both sets."
)