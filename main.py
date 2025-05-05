import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Set Streamlit page configuration
st.set_page_config(page_title="Water Quality Potability Prediction", page_icon="💧")

# Title of the app
st.title("Water Quality Potability Prediction")

# Load existing model if available, otherwise train a new one
model_path = "model.pkl"
if os.path.exists(model_path):
    # Load the saved model
    model = joblib.load(model_path)
else:
    # Train a new model
    st.info("Model file not found. Training model...")

    # Load and prepare the dataset
    try:
        data = pd.read_csv("water_potability.csv")
    except FileNotFoundError:
        st.error("Training data 'water_potability.csv' not found. Please ensure the file is available.")
        st.stop()
    data.fillna(data.mean(), inplace=True)  # Fill missing values with mean

    # Define features and target
    feature_cols = [
        'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
    ]
    X = data[feature_cols]
    y = data['Potability']

    # Split into training and test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=101
    )

    # Train XGBoost classifier
    with st.spinner("Training XGBoost model..."):
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=1)
        model.fit(X_train, y_train)
        # Save the trained model to disk
        joblib.dump(model, model_path)
    st.success("Model training completed and saved.")

# Create tabs for batch and single prediction
tab1, tab2 = st.tabs(["Batch Prediction", "Single Sample Prediction"])

with tab1:
    st.header("Batch Prediction (CSV Upload)")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with water samples for prediction",
        type=["csv"]
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
        else:
            # Required feature columns
            feature_cols = [
                'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
            ]
            # Check for missing columns
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                # Prepare data for prediction
                X_new = df[feature_cols].copy()
                X_new.fillna(X_new.mean(), inplace=True)  # Fill missing values
                # Predict classes and probabilities
                preds = model.predict(X_new)
                probs_1 = model.predict_proba(X_new)[:, 1]  # Probability of class 1 (Potable)
                # Calculate probability of predicted class
                pred_probs = []
                for i, pred in enumerate(preds):
                    if pred == 1:
                        pred_probs.append(probs_1[i])
                    else:
                        pred_probs.append(1 - probs_1[i])
                # Prepare results DataFrame
                result_df = df[feature_cols].copy()
                result_df['Predicted Potability'] = preds
                result_df['Prediction Probability'] = pred_probs
                st.subheader("Prediction Results")
                st.dataframe(result_df)
    else:
        st.write("Awaiting CSV file upload for batch prediction.")

with tab2:
    st.header("Single Sample Prediction")
    st.write("Enter feature values for a single water sample:")
    # Create a form for input
    with st.form("single_sample_form"):
        col1, col2 = st.columns(2)
        with col1:
            ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
            hardness = st.number_input("Hardness", min_value=0.0, max_value=500.0, value=100.0, step=1.0)
            solids = st.number_input("Solids", min_value=0.0, max_value=50000.0, value=20000.0, step=100.0)
            chloramines = st.number_input("Chloramines", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
            sulfate = st.number_input("Sulfate", min_value=0.0, max_value=500.0, value=250.0, step=1.0)
        with col2:
            conductivity = st.number_input("Conductivity", min_value=0.0, max_value=1000.0, value=400.0, step=1.0)
            organic_carbon = st.number_input("Organic_carbon", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
            trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
            turbidity = st.number_input("Turbidity", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        submitted = st.form_submit_button("Predict")
    if submitted:
        # Prepare input array for prediction
        input_features = np.array([[
            ph, hardness, solids, chloramines, sulfate,
            conductivity, organic_carbon, trihalomethanes, turbidity
        ]])
        # Make prediction
        pred = model.predict(input_features)[0]
        proba = model.predict_proba(input_features)[0]
        # Determine associated probability
        prob_val = proba[1] if pred == 1 else proba[0]
        # Display result
        if pred == 1:
            st.success(f"Prediction: Potable (1) with probability {prob_val:.2f}")
        else:
            st.success(f"Prediction: Not Potable (0) with probability {prob_val:.2f}")
