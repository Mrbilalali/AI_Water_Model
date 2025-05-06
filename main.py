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
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Required features
required_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                     'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Model Training & Comparison", "Single Sample Prediction", "Batch Prediction"])

@st.cache_resource
def load_and_train_models():
    try:
        data = pd.read_csv("water_potability.csv")
    except FileNotFoundError:
        st.error("Data file 'water_potability.csv' not found.")
        return None, None, None, None, None, None

    data.fillna(data.mean(), inplace=True)
    X = data[required_features]
    y = data['Potability']

    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        metrics[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1}

    return models, metrics, X_test, y_test, X, y

models, metrics, X_test, y_test, X_data, y_data = load_and_train_models()

if page == "Model Training & Comparison":
    st.header("Train and Compare Models")
    if models is None:
        st.stop()

    st.subheader("Model Performance")
    metrics_df = pd.DataFrame(metrics).T
    st.dataframe(metrics_df.style.format("{:.2f}"))

    st.subheader("Confusion Matrix")
    model_choice = st.selectbox("Choose model", list(models.keys()))
    if model_choice:
        model = models[model_choice]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        st.write(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))

        # Plot heatmap
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix Heatmap")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

elif page == "Single Sample Prediction":
    st.header("🔍 Single Sample Water Potability Prediction")
    if models is None:
        st.stop()

    st.info("Provide the chemical parameters of a single water sample below to predict its potability.")

    input_data = {
        'ph': st.slider('pH', 0.0, 14.0, 7.0),
        'Hardness': st.number_input('Hardness (mg/L)', 0.0, value=200.0),
        'Solids': st.number_input('Solids (ppm)', 0.0, value=300.0),
        'Chloramines': st.number_input('Chloramines (ppm)', 0.0, value=3.0),
        'Sulfate': st.number_input('Sulfate (mg/L)', 0.0, value=300.0),
        'Conductivity': st.number_input('Conductivity (μS/cm)', 0.0, value=300.0),
        'Organic_carbon': st.number_input('Organic Carbon (ppm)', 0.0, value=15.0),
        'Trihalomethanes': st.number_input('Trihalomethanes (μg/L)', 0.0, value=3.0),
        'Turbidity': st.number_input('Turbidity (NTU)', 0.0, value=3.0),
    }

    input_df = pd.DataFrame([input_data])[required_features].astype(float)
    model_choice = st.selectbox("Choose a model", list(models.keys()))

    if st.button("Predict"):
        model = models[model_choice]
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]

        st.metric("Safe Probability (%)", f"{prob[1]*100:.2f}%")
        st.progress(prob[1])

        if pred == 1:
            st.success("✅ Prediction: POTABLE (Safe to drink)")
        else:
            st.warning("❌ Prediction: NOT POTABLE (Unsafe)")

elif page == "Batch Prediction":
    st.header("📁 Batch Water Sample Prediction")
    if models is None:
        st.stop()

    st.info("Upload a CSV file containing multiple water samples with all chemical features to predict their potability.")

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
            labels = ["Safe (1)" if p == 1 else "Not Safe (0)" for p in predictions]
            batch_df['Predicted_Potability'] = labels
            st.subheader("Results")
            st.dataframe(batch_df)

            counts = batch_df['Predicted_Potability'].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)

            st.bar_chart(counts)
            st.success(f"✅ Safe Samples: {counts.get('Safe (1)', 0)}")
            st.warning(f"❌ Not Safe Samples: {counts.get('Not Safe (0)', 0)}")

            # Confusion matrix if actual potability is available
            if 'Potability' in batch_df.columns:
                true_labels = batch_df['Potability']
                pred_binary = [1 if label == "Safe (1)" else 0 for label in batch_df['Predicted_Potability']]
                cm = confusion_matrix(true_labels, pred_binary)
                st.subheader("Confusion Matrix")
                st.write(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))

                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax_cm)
                ax_cm.set_title("Batch Prediction Confusion Matrix")
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                st.pyplot(fig_cm)

        except Exception as e:
            st.error(f"Prediction error: {e}")
