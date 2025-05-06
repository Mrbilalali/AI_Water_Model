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
from sklearn.utils import resample

st.set_page_config(page_title="Water Potability Predictor", layout="wide")
st.title("💧 Water Potability Predictor")

# Load and clean data
df_water = pd.read_csv("water_potability.csv")
df_water_cleaned = df_water.fillna(df_water.mean())

# Train-test split
x = df_water_cleaned.drop("Potability", axis=1)
y = df_water_cleaned["Potability"]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=101)

# Handle imbalance using RandomOverSampler
df_train = pd.concat([x_train, y_train], axis=1)
majority = df_train[df_train.Potability == 0]
minority = df_train[df_train.Potability == 1]
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
df_balanced = pd.concat([majority, minority_upsampled])

x_train_bal = df_balanced.drop("Potability", axis=1)
y_train_bal = df_balanced["Potability"]

# Define models
models = {
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "MLPClassifier": MLPClassifier(max_iter=1000, random_state=42),
}

st.sidebar.header("Model Selection")
selected_models = st.sidebar.multiselect("Choose models to train:", options=list(models.keys()), default=list(models.keys()))

trained_models = {}
results = []

st.header("🔍 Model Training Results")
for name in selected_models:
    model = models[name]
    model.fit(x_train_bal, y_train_bal)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    results.append({"Algorithm": name, "Score": round(acc, 3)})
    trained_models[name] = model

    st.subheader(f"Model: {name}")
    st.write(f"Accuracy: {acc:.2%}")

    # Confusion matrix
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax_cm)
    ax_cm.set_title(f"Confusion Matrix - {name}")
    st.pyplot(fig_cm)

    # Classification report
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

# Score comparison
score_df = pd.DataFrame(results)
st.subheader("📈 Score Comparison Across Models")
fig_score, ax_score = plt.subplots()
sns.barplot(data=score_df, x="Algorithm", y="Score", palette="mako", ax=ax_score)
ax_score.set_ylim(0.5, 1.0)
ax_score.set_title("Model Accuracy Comparison")
ax_score.bar_label(ax_score.containers[0])
st.pyplot(fig_score)

# --- Batch Prediction ---
st.header("📁 Batch File Prediction")
uploaded_file = st.file_uploader("Upload CSV file with water sample features", type="csv")

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    batch_df = batch_df.fillna(df_water.mean())

    selected_model_name = st.selectbox("Select model for prediction:", options=selected_models)
    selected_model = trained_models[selected_model_name]

    predictions = selected_model.predict(batch_df)
    probabilities = selected_model.predict_proba(batch_df)[:, 1]

    batch_df['Potability_Prediction'] = predictions
    batch_df['Probability (%)'] = (probabilities * 100).round(2)

    st.subheader("🔎 Batch Prediction Results")
    st.dataframe(batch_df)

    potable_percent = (batch_df['Potability_Prediction'].mean()) * 100
    st.metric("Predicted Potable Samples", f"{potable_percent:.2f}%")

    # Pie chart
    fig_pie, ax_pie = plt.subplots()
    batch_df['Potability_Prediction'].value_counts().plot.pie(labels=['Not Safe', 'Safe'], autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#99ff99'], ax=ax_pie)
    ax_pie.set_ylabel("")
    ax_pie.set_title("Potability Prediction Distribution")
    st.pyplot(fig_pie)

# --- Single Prediction ---
st.header("🧪 Single Sample Prediction")
if st.checkbox("Enable", value=True):
    user_input = {}
    for col in x.columns:
        user_input[col] = st.number_input(col, value=float(x[col].mean()), format="%.3f")
    
    single_sample = pd.DataFrame([user_input])
    model_name = st.selectbox("Model for single prediction", options=selected_models, key="single")
    model = trained_models[model_name]
    pred = model.predict(single_sample)[0]
    prob = model.predict_proba(single_sample)[0][1]

    st.markdown(f"### ✅ Prediction: {'Safe' if pred else 'Not Safe'}")
    st.markdown(f"**Probability of being Safe:** {prob*100:.2f}%")

# Footer
st.markdown("---")
st.markdown("*All models are trained using RandomOverSampler for class balancing.*")
