import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Title
st.title("Water Potability Prediction")

# Load and preprocess dataset
df_water = pd.read_csv("water_potability.csv")
df_water_cleaned = df_water.fillna(df_water.mean())

# Class distribution plot
st.subheader("Potability Class Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x='Potability', data=df_water_cleaned, ax=ax1)
st.pyplot(fig1)

# Features and Target
x = df_water_cleaned.iloc[:, :-1]
y = df_water_cleaned['Potability']

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=101)

# Models
models = {
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "MLPClassifier": MLPClassifier(max_iter=500)
}

model_scores = []

# Train and evaluate models
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    model_scores.append((name, acc))

# Show model scores as a graph
st.subheader("Model Accuracy Comparison")
score_df = pd.DataFrame(model_scores, columns=['Algorithm', 'Score'])
fig_score, ax_score = plt.subplots()
sns.barplot(data=score_df, x='Algorithm', y='Score', ax=ax_score)
ax_score.set_ylim(0, 1)
ax_score.set_title("Algorithm Accuracy on Test Set")
st.pyplot(fig_score)

# Single Sample Prediction
st.subheader("🔍 Single Sample Prediction")
st.markdown("Provide values for each feature to predict water potability.")
def user_input_features():
    ph = st.number_input('pH', value=7.0)
    hardness = st.number_input('Hardness', value=200.0)
    solids = st.number_input('Solids', value=15000.0)
    chloramines = st.number_input('Chloramines', value=7.0)
    sulfate = st.number_input('Sulfate', value=330.0)
    conductivity = st.number_input('Conductivity', value=400.0)
    organic_carbon = st.number_input('Organic Carbon', value=10.0)
    trihalomethanes = st.number_input('Trihalomethanes', value=60.0)
    turbidity = st.number_input('Turbidity', value=3.0)
    data = {
        'ph': ph,
        'Hardness': hardness,
        'Solids': solids,
        'Chloramines': chloramines,
        'Sulfate': sulfate,
        'Conductivity': conductivity,
        'Organic_carbon': organic_carbon,
        'Trihalomethanes': trihalomethanes,
        'Turbidity': turbidity
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
st.write("Your Input:", input_df)

# Default model: Random Forest
default_model = RandomForestClassifier(random_state=42)
default_model.fit(x_train, y_train)
pred = default_model.predict(input_df)
proba = default_model.predict_proba(input_df)[0][1] * 100

st.success(f"Prediction: {'Safe' if pred[0] == 1 else 'Unsafe'}")
st.info(f"Confidence: {proba:.2f}%")

# Batch Prediction
st.subheader("📤 Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV File for Batch Prediction", type=['csv'])
if uploaded_file is not None:
    input_batch = pd.read_csv(uploaded_file)
    input_batch = input_batch.fillna(df_water.mean())
    batch_pred = default_model.predict(input_batch)
    batch_proba = default_model.predict_proba(input_batch)[:, 1] * 100
    input_batch['Prediction'] = batch_pred
    input_batch['Probability (%)'] = batch_proba.round(2)
    st.write(input_batch)

    # Bar chart summary
    st.subheader("Batch Prediction Summary")
    batch_summary = input_batch['Prediction'].value_counts(normalize=True) * 100
    fig_batch, ax_batch = plt.subplots()
    batch_summary.plot(kind='bar', color=['red', 'green'], ax=ax_batch)
    ax_batch.set_ylabel("Percentage")
    ax_batch.set_xticklabels(['Unsafe (0)', 'Safe (1)'], rotation=0)
    ax_batch.set_title("Potability Distribution in Batch")
    st.pyplot(fig_batch)

    # Pie chart summary
    fig_pie, ax_pie = plt.subplots()
    input_batch['Prediction'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Unsafe', 'Safe'], colors=['red', 'green'], ax=ax_pie)
    ax_pie.set_title("Potability Percentage (Batch)")
    st.pyplot(fig_pie)

# Footer
st.markdown("---")
st.markdown("Model trained using RandomForestClassifier with accuracy tracking for other models.")
