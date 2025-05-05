import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# Load the trained model (ensure 'model.pkl' or 'water_model.pkl' is in the same directory)
try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    with open("water_model.pkl", "rb") as f:
        model = pickle.load(f)

st.title("💧 Water Quality Potability Classifier")

# Sidebar for input mode selection
mode = st.sidebar.selectbox("Select Prediction Mode", ["Single Sample", "Batch (CSV)"])

if mode == "Single Sample":
    st.header("Single Water Sample Prediction")
    # Input fields for each feature
    ph = st.number_input("pH", value=7.0, step=0.1)
    hardness = st.number_input("Hardness (mg/L)", value=100.0, step=1.0)
    solids = st.number_input("Solids (ppm)", value=10000.0, step=100.0)
    chloramines = st.number_input("Chloramines (ppm)", value=5.0, step=0.1)
    sulfate = st.number_input("Sulfate (mg/L)", value=250.0, step=1.0)
    conductivity = st.number_input("Conductivity (μS/cm)", value=400.0, step=1.0)
    organic_carbon = st.number_input("Organic Carbon (mg/L)", value=10.0, step=0.1)
    trihalomethanes = st.number_input("Trihalomethanes (ppb)", value=80.0, step=1.0)
    turbidity = st.number_input("Turbidity (NTU)", value=3.0, step=0.1)
    
    if st.button("Predict Single Sample"):
        # Prepare input DataFrame
        input_dict = {
            'ph': [ph],
            'Hardness': [hardness],
            'Solids': [solids],
            'Chloramines': [chloramines],
            'Sulfate': [sulfate],
            'Conductivity': [conductivity],
            'Organic_carbon': [organic_carbon],
            'Trihalomethanes': [trihalomethanes],
            'Turbidity': [turbidity]
        }
        input_df = pd.DataFrame(input_dict)
        
        # Perform prediction
        pred_class = model.predict(input_df)[0]
        pred_prob = model.predict_proba(input_df)[0]  # [prob_not_potable, prob_potable]
        prob_not, prob_pot = pred_prob[0], pred_prob[1]
        
        # Display prediction result
        if pred_class == 1:
            st.success("✅ **Result:** The water sample is predicted **Potable**.")
        else:
            st.error("❌ **Result:** The water sample is predicted **Not Potable**.")
        
        # Display probabilities
        st.subheader("Class Probabilities")
        prob_col1, prob_col2 = st.columns(2)
        prob_col1.write(f"Potable Water: **{prob_pot * 100:.2f}%**")
        prob_col1.progress(int(prob_pot * 100))
        prob_col2.write(f"Not Potable Water: **{prob_not * 100:.2f}%**")
        prob_col2.progress(int(prob_not * 100))
        
        # (Optional) Display a small bar chart for probabilities
        chart_data = pd.DataFrame({
            'Probability (%)': [prob_not * 100, prob_pot * 100]
        }, index=['Not Potable', 'Potable'])
        st.bar_chart(chart_data, x_label="Water Type", y_label="Probability (%)")

elif mode == "Batch (CSV)":
    st.header("Batch Prediction from CSV")
    uploaded_file = st.file_uploader("Upload a CSV file with water samples", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()
        
        # Ensure required feature columns exist
        required_cols = ['ph','Hardness','Solids','Chloramines','Sulfate',
                         'Conductivity','Organic_carbon','Trihalomethanes','Turbidity']
        if not all(col in data.columns for col in required_cols):
            st.error(f"Uploaded CSV must contain columns: {required_cols}")
        else:
            if st.button("Predict Batch"):
                # Predict for each sample
                predictions = model.predict(data[required_cols])
                data['Predicted_Potability'] = predictions
                
                # Calculate counts and percentages
                total_samples = len(data)
                class_counts = pd.Series(predictions).value_counts().sort_index()
                count_not = class_counts.get(0, 0)
                count_pot = class_counts.get(1, 0)
                perc_not = (count_not / total_samples) * 100
                perc_pot = (count_pot / total_samples) * 100
                
                # Display counts and percentages
                st.subheader("Prediction Summary")
                col1, col2 = st.columns(2)
                col1.metric("Potable Samples", f"{count_pot} ({perc_pot:.2f}%)")
                col2.metric("Not Potable Samples", f"{count_not} ({perc_not:.2f}%)")
                
                # Bar chart of percentage distribution
                dist_df = pd.DataFrame({
                    'Percentage (%)': [perc_not, perc_pot]
                }, index=['Not Potable', 'Potable'])
                st.bar_chart(dist_df, x_label="Water Type", y_label="Percentage (%)")
                
                # Show data with predictions
                st.subheader("Predicted Data")
                st.dataframe(data)
