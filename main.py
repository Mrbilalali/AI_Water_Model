import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Required columns
REQUIRED_FEATURES = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                     'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

st.set_page_config(page_title="Water Potability Prediction", layout="wide")
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Model Training", "Single Prediction", "Batch Prediction"])

@st.cache_resource
def load_and_train_models():
    data = pd.read_csv("water_potability.csv")
    data.fillna(data.mean(), inplace=True)
    X = data[REQUIRED_FEATURES]
    y = data['Potability']

    ros = RandomOverSampler(random_state=42)
    rus = RandomUnderSampler(random_state=42)
    X_over, y_over = ros.fit_resample(X, y)
    X_under, y_under = rus.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "model": model,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "Report": classification_report(y_test, y_pred, output_dict=True),
            "Confusion Matrix": confusion_matrix(y_test, y_pred)
        }
    return results, X_test, y_test

models_data, X_test, y_test = load_and_train_models()

if page == "Model Training":
    st.title("📊 Water Potability Model Training & Evaluation")

    metric_df = pd.DataFrame({k: {m: round(v[m], 2) for m in ['Accuracy', 'Precision', 'Recall', 'F1 Score']} 
                               for k, v in models_data.items()}).T

    st.subheader("Model Metrics Comparison")
    st.dataframe(metric_df.style.background_gradient(cmap='BuGn'))

    st.subheader("📈 Metric Comparison Chart")
    st.bar_chart(metric_df)

    st.subheader("Confusion Matrix")
    selected_model = st.selectbox("Select model to view confusion matrix", list(models_data.keys()))
    cm = models_data[selected_model]['Confusion Matrix']
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{selected_model} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

elif page == "Single Prediction":
    st.title("💧 Single Water Sample Prediction")
    st.markdown("Enter the values below to predict potability:")

    sample = {}
    for col in REQUIRED_FEATURES:
        sample[col] = st.number_input(f"{col}", value=0.0, step=0.1)

    selected_model = st.selectbox("Select Model", list(models_data.keys()))

    if st.button("Predict Potability"):
        model = models_data[selected_model]['model']
        X_input = pd.DataFrame([sample])
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][1]

        st.metric("Probability Safe (%)", f"{prob * 100:.2f}%")
        st.progress(prob)

        if pred == 1:
            st.success("✅ Prediction: POTABLE (Safe to Drink)")
        else:
            st.error("❌ Prediction: NOT POTABLE")

elif page == "Batch Prediction":
    st.title("📁 Batch Water Prediction")
    uploaded = st.file_uploader("Upload a CSV file", type=['csv'])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [col for col in REQUIRED_FEATURES if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            df = df[REQUIRED_FEATURES].apply(pd.to_numeric, errors='coerce')
            df.fillna(df.mean(), inplace=True)

            selected_model = st.selectbox("Select Model", list(models_data.keys()))
            model = models_data[selected_model]['model']
            preds = model.predict(df)

            df['Prediction'] = preds
            df['Label'] = df['Prediction'].map({1: 'Safe', 0: 'Not Safe'})
            st.dataframe(df)

            # Plot distribution
            counts = df['Label'].value_counts()
            st.subheader("Prediction Distribution")
            fig1, ax1 = plt.subplots()
            ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)
            st.bar_chart(counts)

            st.success(f"Total Samples: {len(df)}")
            st.success(f"Safe: {counts.get('Safe', 0)}")
            st.warning(f"Not Safe: {counts.get('Not Safe', 0)}")
