import os
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Online Shopping Intent Prediction", layout="centered")

st.title("🛒 Online Shopping Intent Prediction")
st.write("Predict whether a user session will result in a purchase (Revenue = 1).")

@st.cache_data
def load_data():
    # ✅ Always load CSV relative to this app.py file (works on Streamlit Cloud)
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "online_shoppers_intention.csv")

    df = pd.read_csv(file_path, encoding="latin1")

    # drop weird junk PK column if present
    df = df.loc[:, ~df.columns.str.contains("PK", case=False, na=False)]
    df = df.dropna(subset=["Revenue"])
    df["Revenue"] = df["Revenue"].astype(int)
    return df

@st.cache_resource
def train_model(df):
    X = df.drop("Revenue", axis=1)
    y = df["Revenue"]

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(include=np.number).columns

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    model = RandomForestClassifier(n_estimators=200, random_state=42)

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)
    return pipe, X, acc

df = load_data()
pipe, X_template, acc = train_model(df)

st.success(f"Model trained successfully. Test Accuracy: {acc:.3f}")

st.subheader("Enter a new session (inputs)")

user_input = {}
for col in X_template.columns:
    if str(X_template[col].dtype) == "object":
        user_input[col] = st.selectbox(col, sorted(X_template[col].dropna().unique().tolist()))
    else:
        default_val = float(X_template[col].median())
        user_input[col] = st.number_input(col, value=default_val)

input_df = pd.DataFrame([user_input])

if st.button("Predict"):
    pred = pipe.predict(input_df)[0]
    prob = pipe.predict_proba(input_df)[0][1]

    if pred == 1:
        st.success("✅ Prediction: Purchase (Revenue = 1)")
    else:
        st.warning("❌ Prediction: No Purchase (Revenue = 0)")

    st.write(f"Purchase probability: **{prob:.2f}**")
