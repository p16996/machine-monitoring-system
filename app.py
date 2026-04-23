import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Machine Failure Monitoring System", layout="wide")

st.title("⚙️ Machine Failure Monitoring System")
st.markdown("AI-powered system to predict machine failure risk using operational data.")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/predictive_maintenance.csv")

df = load_data()

# -------------------------------
# Dataset Overview
# -------------------------------
st.markdown("## 📊 Dataset Overview")

col1, col2 = st.columns(2)
col1.metric("Total Records", df.shape[0])
col2.metric("Total Features", df.shape[1])

st.dataframe(df.head(), use_container_width=True)

# -------------------------------
# Data Preparation
# -------------------------------
df_model = df.drop(columns=["UDI", "Product ID"])
df_model = pd.get_dummies(df_model, columns=["Type"], drop_first=True)

X = df_model.drop(columns=["Machine failure"])
y = df_model["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Model Training
# -------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -------------------------------
# Metrics
# -------------------------------
st.markdown("## 🤖 Model Performance")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
c2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.2f}")
c3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.2f}")
c4.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.2f}")

# -------------------------------
# Confusion Matrix
# -------------------------------
st.markdown("### Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual: No Failure", "Actual: Failure"],
    columns=["Predicted: No Failure", "Predicted: Failure"]
)

st.dataframe(cm_df, use_container_width=True)

# -------------------------------
# Feature Importance
# -------------------------------
st.markdown("## 📌 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.dataframe(importance_df.head(10), use_container_width=True)

fig, ax = plt.subplots()
importance_df_sorted = importance_df.sort_values(by="Importance")

ax.barh(
    importance_df_sorted["Feature"],
    importance_df_sorted["Importance"]
)

ax.set_title("Feature Importance")
ax.set_xlabel("Importance Score")

st.pyplot(fig)

# -------------------------------
# Prediction Section
# -------------------------------
st.markdown("## 🔮 Predict Machine Failure Risk")

air_temp = st.number_input("Air Temperature (K)", value=300.0)
process_temp = st.number_input("Process Temperature (K)", value=310.0)
rpm = st.number_input("Rotational Speed (RPM)", value=1500)
torque = st.number_input("Torque (Nm)", value=40.0)
tool_wear = st.number_input("Tool Wear (min)", value=100)

machine_type = st.selectbox("Machine Type", ["L", "M", "H"])

input_data = pd.DataFrame({
    "Air temperature [K]": [air_temp],
    "Process temperature [K]": [process_temp],
    "Rotational speed [rpm]": [rpm],
    "Torque [Nm]": [torque],
    "Tool wear [min]": [tool_wear],
    "Type_L": [1 if machine_type == "L" else 0],
    "Type_M": [1 if machine_type == "M" else 0]
})

input_data = input_data.reindex(columns=X.columns, fill_value=0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Risk"):

    prob = model.predict_proba(input_data)[0][1]

    st.markdown("### Prediction Result")
    st.metric("Failure Probability", f"{prob:.2%}")

    if prob >= 0.5:
        st.error("🔴 High Risk of Failure")
    elif prob >= 0.2:
        st.warning("🟠 Moderate Risk - Monitor Closely")
    else:
        st.success("🟢 Low Risk - Normal Operation")