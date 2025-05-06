import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Required features
required_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                     'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Model Training & Comparison", "Single Sample Prediction", "Batch Prediction"])

@st.cache_resource
def load_and_train_models():
    try:
        df_water = pd.read_csv("water_potability.csv")
    except FileNotFoundError:
        st.error("Data file 'water_potability.csv' not found.")
        return None, None, None, None, None, None

    df_water.fillna(df_water.mean(), inplace=True)
    st.subheader("Class Distribution in Dataset")
    fig = plt.figure(figsize=(5, 3))
    sns.countplot(x='Potability', data=df_water)
    st.pyplot(fig)

    x = df_water[required_features]
    y = df_water['Potability']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=101)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    metrics = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        metrics[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1}

    return models, metrics, x_test, y_test, x, y

models, metrics, X_test, y_test, X_data, y_data = load_and_train_models()

if page == "Batch Prediction":
    st.header("📁 Batch Water Sample Prediction")
    if models is None:
        st.stop()

    uploaded_file = st.file_uploader("Upload water samples CSV", type=['csv'])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception:
            st.error("Invalid file.")
            st.stop()

        missing = [col for col in required_features if col not in batch_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        batch_df = batch_df[required_features].apply(pd.to_numeric, errors='coerce')
        batch_df = batch_df.fillna(batch_df.mean())

        model_choice = st.selectbox("Choose a model", list(models.keys()))
        model = models[model_choice]

        try:
            predictions = model.predict(batch_df)
            probs = model.predict_proba(batch_df)[:, 1]
            labels = ["Safe (1)" if p == 1 else "Not Safe (0)" for p in predictions]
            batch_df['Predicted_Potability'] = labels
            batch_df['Probability_Safe'] = probs
            st.subheader("Results")
            st.dataframe(batch_df)

            # Summary pie chart
            counts = batch_df['Predicted_Potability'].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)

            st.bar_chart(counts)

            st.success(f"✅ Safe Samples: {counts.get('Safe (1)', 0)}")
            st.warning(f"❌ Not Safe Samples: {counts.get('Not Safe (0)', 0)}")

            # Confusion matrix comparison
            st.subheader("Confusion Matrix Comparison (using Test Set)")
            for name, model in models.items():
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title(f"{name} Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction error: {e}")
