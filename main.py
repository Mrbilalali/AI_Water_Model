import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Streamlit app: Water Quality Classifier and Predictor

# Required input features for water quality prediction
required_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                     'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Model Training & Comparison", "Single Sample Prediction", "Batch Prediction"])

# Load and preprocess data, then train multiple models
@st.cache_resource
def load_and_train_models():
    # Load dataset (ensure 'water_potability.csv' is in the app directory)
    try:
        data = pd.read_csv("water_potability.csv")
    except FileNotFoundError:
        st.error("Data file 'water_potability.csv' not found. Please place the dataset in the app directory.")
        return None, None, None, None, None, None
    # Fill missing values in dataset with column mean
    data.fillna(data.mean(), inplace=True)
    # Separate features and target
    X = data[required_features]
    if 'Potability' in data.columns:
        y = data['Potability']
    else:
        st.error("Training data does not contain 'Potability' target column.")
        return None, None, None, None, None, None
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)
    # Define multiple classifiers
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }
    # Train each model and compute performance metrics
    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        metrics[name] = {"Accuracy": acc, "Precision": prec,
                         "Recall": rec, "F1 Score": f1}
    return models, metrics, X_test, y_test, X, y

# Load models and metrics (cached for efficiency)
models, metrics, X_test, y_test, X_data, y_data = load_and_train_models()

# Page: Model Training & Comparison
if page == "Model Training & Comparison":
    st.header("Train and Compare Multiple Water Quality Classifiers")
    if models is None:
        st.stop()
    # Show performance metrics table
    st.subheader("Model Performance on Test Set")
    metrics_df = pd.DataFrame(metrics).T
    st.dataframe(metrics_df.style.format({"Accuracy": "{:.2f}",
                                          "Precision": "{:.2f}",
                                          "Recall": "{:.2f}",
                                          "F1 Score": "{:.2f}"}))
    # Confusion matrix for a selected model
    st.subheader("Confusion Matrix (Test Data)")
    model_choice = st.selectbox("Choose a model for confusion matrix",
                                list(models.keys()))
    if model_choice:
        model = models[model_choice]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        st.write(f"Confusion matrix for {model_choice}:")
        st.write(pd.DataFrame(cm,
                              index=["Actual 0", "Actual 1"],
                              columns=["Predicted 0", "Predicted 1"]))

# Page: Single Sample Prediction
elif page == "Single Sample Prediction":
    st.header("Predict Water Potability for a Single Sample")
    if models is None:
        st.stop()
    st.subheader("Input Water Quality Parameters")
    # Collect user inputs for each feature
    input_data = {}
    input_data['ph'] = st.slider('pH', 0.0, 14.0, 7.0)
    input_data['Hardness'] = st.number_input('Hardness (mg/L)',
                                             min_value=0.0, step=0.1, value=200.0)
    input_data['Solids'] = st.number_input('Solids (ppm)',
                                          min_value=0.0, step=1.0, value=300.0)
    input_data['Chloramines'] = st.number_input('Chloramines (ppm)',
                                               min_value=0.0, step=0.1, value=3.0)
    input_data['Sulfate'] = st.number_input('Sulfate (mg/L)',
                                           min_value=0.0, step=0.1, value=300.0)
    input_data['Conductivity'] = st.number_input('Conductivity (μS/cm)',
                                                min_value=0.0, step=0.1, value=300.0)
    input_data['Organic_carbon'] = st.number_input('Organic Carbon (ppm)',
                                                  min_value=0.0, step=0.1, value=15.0)
    input_data['Trihalomethanes'] = st.number_input('Trihalomethanes (μg/L)',
                                                   min_value=0.0, step=0.1, value=3.0)
    input_data['Turbidity'] = st.number_input('Turbidity (NTU)',
                                             min_value=0.0, step=0.1, value=3.0)
    input_df = pd.DataFrame([input_data])
    # Model selection for prediction
    model_choice2 = st.selectbox("Choose a model for prediction",
                                 list(models.keys()), index=0)
    # Predict on button press
    if st.button("Predict"):
        model = models[model_choice2]
        try:
            pred = model.predict(input_df)[0]
        except Exception as e:
            st.error(f"Error in prediction: {e}")
        else:
            if pred == 1:
                st.success("Prediction: POTABLE (Safe)")
            else:
                st.warning("Prediction: NOT POTABLE (Unsafe)")

# Page: Batch Prediction
elif page == "Batch Prediction":
    st.header("Batch Prediction from CSV File")
    if models is None:
        st.stop()
    st.subheader("Upload CSV")
    uploaded_file = st.file_uploader("Upload a CSV file with water samples",
                                     type=['csv'])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception:
            st.error("Could not read the uploaded file. Ensure it is a valid CSV.")
            st.stop()
        # Validate required columns presence
        missing_cols = [col for col in required_features if col not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}. Please include all required features: {required_features}")
            st.stop()
        # Drop any extra columns (like 'Potability')
        extra_cols = [col for col in batch_df.columns if col not in required_features]
        if extra_cols:
            batch_df = batch_df.drop(columns=extra_cols)
        # Reorder to match training order
        batch_df = batch_df[required_features]
        # Convert to numeric, coerce errors to NaN
        batch_numeric = batch_df.apply(pd.to_numeric, errors='coerce')
        means = batch_numeric.mean()
        # Check for columns with no numeric data
        invalid_cols = [col for col in required_features if pd.isna(means[col])]
        if invalid_cols:
            st.error(f"Columns {invalid_cols} have invalid (non-numeric) values. Ensure all values are numeric.")
            st.stop()
        # Fill missing values with means
        batch_clean = batch_numeric.fillna(means)
        # Model selection for batch prediction
        model_choice3 = st.selectbox("Choose a model for batch prediction",
                                     list(models.keys()), index=0)
        model = models[model_choice3]
        try:
            predictions = model.predict(batch_clean)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
        # Map predictions to human-readable labels
        label_map = {0: "Not Safe (0)", 1: "Safe (1)"}
        prediction_labels = [label_map.get(pred, str(pred)) for pred in predictions]
        # Display results
        output_df = batch_df.copy()
        output_df['Predicted_Potability'] = prediction_labels
        st.subheader("Prediction Results")
        st.dataframe(output_df)
        # Visualization of prediction distribution
        st.subheader("Prediction Distribution")
        pred_counts = pd.Series(prediction_labels).value_counts().reset_index()
        pred_counts.columns = ['Potability', 'Count']
        st.bar_chart(data=pred_counts.set_index('Potability'))
