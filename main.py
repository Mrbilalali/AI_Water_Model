import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

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
X = df_water_cleaned.iloc[:, :-1]
y = df_water_cleaned['Potability']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=101, stratify=y)

# Balance training data via oversampling
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

# Scale features for algorithms that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "RandomForestClassifier": (RandomForestClassifier(random_state=42), False),
    "LogisticRegression": (LogisticRegression(max_iter=2000, solver='lbfgs'), True),
    "KNeighborsClassifier": (KNeighborsClassifier(), True),
    "MLPClassifier": (MLPClassifier(max_iter=1000), True),
    "GaussianNB": (GaussianNB(), False)
}

model_scores = []
# Train and evaluate models
for name, (model, needs_scaling) in models.items():
    if needs_scaling:
        model.fit(X_train_scaled, y_train_bal)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train_bal, y_train_bal)
        preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
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
    data = {col: st.number_input(col, value=float(X_train[col].mean())) for col in X.columns}
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
st.write("Your Input:", input_df)

# Default model: Random Forest
default_model, default_scaled = models['RandomForestClassifier']
default_model.fit(X_train_bal, y_train_bal)

# Prepare input for prediction
input_arr = input_df.values
if default_scaled:
    input_arr = scaler.transform(input_arr)
pred = default_model.predict(input_arr)[0]
proba = default_model.predict_proba(input_arr)[0][1] * 100

st.success(f"Prediction: {'Safe' if pred == 1 else 'Unsafe'}")
st.info(f"Confidence: {proba:.2f}%")

# Batch Prediction
st.subheader("📤 Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV File for Batch Prediction", type=['csv'])
if uploaded_file is not None:
    input_batch = pd.read_csv(uploaded_file)
    input_batch = input_batch.fillna(df_water.mean())
    # Ensure order
    input_batch = input_batch[X.columns]
    batch_arr = input_batch.values
    if default_scaled:
        batch_arr = scaler.transform(batch_arr)
    batch_pred = default_model.predict(batch_arr)
    batch_proba = default_model.predict_proba(batch_arr)[:, 1] * 100
    input_batch['Prediction'] = batch_pred
    input_batch['Probability (%)'] = batch_proba.round(2)
    st.write(input_batch)

    # Bar chart summary
    st.subheader("Batch Prediction Summary")
    batch_summary = pd.Series(batch_pred).value_counts(normalize=True) * 100
    fig_batch, ax_batch = plt.subplots()
    batch_summary.plot(kind='bar', ax=ax_batch)
    ax_batch.set_ylabel("Percentage")
    ax_batch.set_xticklabels(['Unsafe (0)', 'Safe (1)'], rotation=0)
    ax_batch.set_title("Potability Distribution in Batch")
    st.pyplot(fig_batch)

    # Pie chart summary
    fig_pie, ax_pie = plt.subplots()
    pd.Series(batch_pred).map({0:'Unsafe',1:'Safe'}).value_counts().plot.pie(autopct='%1.1f%%', ax=ax_pie)
    ax_pie.set_title("Potability Percentage (Batch)")
    st.pyplot(fig_pie)
