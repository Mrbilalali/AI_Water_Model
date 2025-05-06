# Required Libraries: 
# pip install streamlit pandas scikit-learn imbalanced-learn xgboost seaborn altair

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

# -- Streamlit Page Configuration --
st.set_page_config(page_title="Water Quality Classifier", layout="wide")

# -- Data Loading and Preprocessing --
DATA_URL = "https://raw.githubusercontent.com/Sarthak-1408/Water-Potability/main/water_potability.csv"

@st.cache_data
def load_data():
    """Load data from URL and fill missing values."""
    df = pd.read_csv(DATA_URL)
    # Impute missing values with column mean
    df.fillna(df.mean(), inplace=True)
    X = df.drop("Potability", axis=1)
    y = df["Potability"]
    return X, y

@st.cache_data
def split_data(X, y):
    """Split data into training and test sets."""
    # Use stratify to maintain class distribution
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

@st.cache_resource
def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train models with specified resampling techniques and return a dictionary 
    of trained models with their metrics.
    """
    models_metrics = {}
    rus = RandomUnderSampler(random_state=42)
    ros = RandomOverSampler(random_state=42)
    # Scale numeric features for models that benefit from scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1) RUS + RandomForestClassifier
    X_rus, y_rus = rus.fit_resample(X_train, y_train)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_rus, y_rus)
    y_pred = rf.predict(X_test)
    models_metrics["RUS + RandomForest"] = {
        "model": rf,
        "scaled": False,
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report_dict": classification_report(y_test, y_pred, output_dict=True)
    }

    # 2) ROS + LogisticRegression
    X_ros, y_ros = ros.fit_resample(X_train_scaled, y_train)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_ros, y_ros)
    y_pred = lr.predict(X_test_scaled)
    models_metrics["ROS + LogisticRegression"] = {
        "model": lr,
        "scaled": True,
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report_dict": classification_report(y_test, y_pred, output_dict=True)
    }

    # 3) ROS + GaussianNB
    X_ros2, y_ros2 = ros.fit_resample(X_train_scaled, y_train)
    nb = GaussianNB()
    nb.fit(X_ros2, y_ros2)
    y_pred = nb.predict(X_test_scaled)
    models_metrics["ROS + GaussianNB"] = {
        "model": nb,
        "scaled": True,
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report_dict": classification_report(y_test, y_pred, output_dict=True)
    }

    # 4) RUS + KNeighborsClassifier
    X_rus2, y_rus2 = rus.fit_resample(X_train_scaled, y_train)
    knn = KNeighborsClassifier()
    knn.fit(X_rus2, y_rus2)
    y_pred = knn.predict(X_test_scaled)
    models_metrics["RUS + KNeighbors"] = {
        "model": knn,
        "scaled": True,
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report_dict": classification_report(y_test, y_pred, output_dict=True)
    }

    # 5) ROS + XGBoost (XGBClassifier)
    X_ros3, y_ros3 = ros.fit_resample(X_train, y_train)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_ros3, y_ros3)
    y_pred = xgb.predict(X_test)
    models_metrics["ROS + XGBoost"] = {
        "model": xgb,
        "scaled": False,
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report_dict": classification_report(y_test, y_pred, output_dict=True)
    }

    return models_metrics, scaler

# Load and split the data
X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)

# Train all models and get their metrics
model_metrics, scaler = train_all_models(X_train, y_train, X_test, y_test)

# -- Streamlit Interface --

# Sidebar: select which models to display/ use
model_keys = list(model_metrics.keys())
selected_models = st.sidebar.multiselect(
    "Select Model(s) for Analysis:",
    model_keys,
    default=model_keys
)

# Create three tabs: Model Comparison, Single Prediction, Batch Prediction
tab1, tab2, tab3 = st.tabs(["Model Comparison", "Single Prediction", "Batch Prediction"])

with tab1:
    st.header("Model Training and Comparison")
    st.write("The performance of each selected model on the test set is shown below:")
    for name in selected_models:
        res = model_metrics[name]
        st.subheader(name)
        # Display accuracy
        st.write(f"**Accuracy:** {res['accuracy']:.3f}")
        # Display classification report as text
        report_text = classification_report(
            y_test, 
            res["model"].predict(
                scaler.transform(X_test) if res["scaled"] else X_test
            )
        )
        st.text(f"Classification Report:\n{report_text}")
        # Confusion matrix heatmap
        cm = res["confusion_matrix"]
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix for {name}")
        st.pyplot(fig_cm)
        # Bar chart for precision, recall, f1-score (classes 0 and 1)
        report_df = pd.DataFrame(res["report_dict"]).T.iloc[:2, :3]  # take classes 0 and 1 rows
        st.bar_chart(report_df[["precision", "recall", "f1-score"]])

with tab2:
    st.header("Single Sample Prediction")
    st.write("Enter values for a single water sample to predict its potability.")
    with st.form("prediction_form"):
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
        hardness = st.number_input("Hardness", min_value=0.0, value=200.0)
        solids = st.number_input("Solids (ppm)", min_value=0.0, value=20000.0)
        chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)
        sulfate = st.number_input("Sulfate", min_value=0.0, value=250.0)
        conductivity = st.number_input("Conductivity", min_value=0.0, value=300.0)
        organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=10.0)
        trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=4.0)
        turbidity = st.number_input("Turbidity", min_value=0.0, value=3.0)
        submit_single = st.form_submit_button("Predict")
    if submit_single:
        # Create input array
        X_new = np.array([[ph, hardness, solids, chloramines,
                           sulfate, conductivity, organic_carbon,
                           trihalomethanes, turbidity]])
        for name in selected_models:
            model = model_metrics[name]["model"]
            # Scale input if model was trained on scaled data
            X_input = scaler.transform(X_new) if model_metrics[name]["scaled"] else X_new
            pred_proba = model.predict_proba(X_input)[0]
            pred_class = model.predict(X_input)[0]
            st.write(f"**{name}:** Predicted = **{'Potable' if pred_class==1 else 'Not Potable'}**")
            st.write(f" - Probability (Potable=1): {pred_proba[1]*100:.1f}%")
            st.write(f" - Probability (Not Potable=0): {pred_proba[0]*100:.1f}%")

with tab3:
    st.header("Batch Prediction")
    st.write("Upload a CSV file of water samples for batch prediction (columns must match features).")
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)
        st.write("Input Data Preview:")
        st.dataframe(df_batch.head())
        # Make predictions for each selected model
        for name in selected_models:
            model = model_metrics[name]["model"]
            if model_metrics[name]["scaled"]:
                X_batch = scaler.transform(df_batch.values)
            else:
                X_batch = df_batch.values
            df_batch[name] = model.predict(X_batch)
        st.write("Predictions:")
        st.dataframe(df_batch)
        # Display summary bar charts of predicted classes for each model
        for name in selected_models:
            counts = df_batch[name].value_counts(normalize=True).sort_index() * 100
            summary_df = pd.DataFrame({
                'Class': ['Not Potable (0)', 'Potable (1)'],
                'Percentage': [counts.get(0, 0), counts.get(1, 0)]
            })
            st.subheader(f"{name} Prediction Summary")
            chart = alt.Chart(summary_df).mark_bar().encode(
                x='Class:N', y='Percentage:Q', color='Class:N'
            )
            st.altair_chart(chart, use_container_width=True)
