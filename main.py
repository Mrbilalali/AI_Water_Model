import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample

# Title
st.title("Water Potability Prediction - Batch & Individual Analysis")

# Load dataset
df_water = pd.read_csv("water_potability.csv")

# Display missing values
st.subheader("Missing Values per Column")
st.write(df_water.isna().sum())

# Fill missing with column mean
df_water_cleaned = df_water.fillna(df_water.mean())

# Confirm no missing values
st.subheader("After Cleaning")
st.write(df_water_cleaned.isna().sum())

# Class balance chart
st.subheader("Class Distribution (Potability)")
fig1, ax1 = plt.subplots()
sns.countplot(x='Potability', data=df_water_cleaned, ax=ax1)
st.pyplot(fig1)

# Features and Target
x = df_water_cleaned.iloc[:, :-1]
y = df_water_cleaned['Potability']

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=101)

st.write("Training Features Shape:", x_train.shape)
st.write("Training Labels Shape:", y_train.shape)
st.write("Test Features Shape:", x_test.shape)
st.write("Test Labels Shape:", y_test.shape)

# Resampling to address imbalance
df_train = pd.concat([x_train, y_train], axis=1)
majority = df_train[df_train.Potability == 0]
minority = df_train[df_train.Potability == 1]
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
df_balanced = pd.concat([majority, minority_upsampled])

x_train_balanced = df_balanced.iloc[:, :-1]
y_train_balanced = df_balanced['Potability']

# Models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier()
}

results = {}

st.subheader("Model Training and Evaluation")
for name, model in models.items():
    model.fit(x_train_balanced, y_train_balanced)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {
        "accuracy": acc,
        "report": report,
        "conf_matrix": cm
    }
    
    st.markdown(f"### {name}")
    st.write(f"Accuracy: {acc:.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title(f"{name} - Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

# Batch Upload Prediction
st.subheader("📤 Batch Water Sample Prediction")
uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    input_df = input_df.fillna(df_water.mean())

    selected_model_name = st.selectbox("Select Model", list(models.keys()))
    selected_model = models[selected_model_name]
    predictions = selected_model.predict(input_df)
    probabilities = selected_model.predict_proba(input_df)[:, 1]

    input_df['Potability_Prediction'] = predictions
    input_df['Probability (%)'] = (probabilities * 100).round(2)
    
    st.write(input_df)

    # Total Percentage Chart
    st.subheader("🧮 Potability Distribution in Uploaded Batch")
    summary = input_df['Potability_Prediction'].value_counts(normalize=True) * 100
    fig_summary, ax_summary = plt.subplots()
    summary.plot(kind='bar', color=['red', 'green'], ax=ax_summary)
    ax_summary.set_ylabel("Percentage")
    ax_summary.set_xticklabels(['Unsafe (0)', 'Safe (1)'], rotation=0)
    ax_summary.set_title("Potability Distribution in Batch")
    st.pyplot(fig_summary)

    # Pie Chart
    fig_pie, ax_pie = plt.subplots()
    input_df['Potability_Prediction'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Unsafe', 'Safe'], colors=['red', 'green'], ax=ax_pie)
    ax_pie.set_title("Potable vs Non-Potable Distribution")
    st.pyplot(fig_pie)

# Note
st.markdown("---")
st.markdown("*Trained using balanced dataset with RandomOverSampler (minority class).*")
