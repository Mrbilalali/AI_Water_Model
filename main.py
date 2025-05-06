import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

st.set_page_config(page_title="Water Potability Classifier", layout="wide")
st.title("🚰 Water Quality Potability Prediction App")

# Load and preprocess dataset
data = pd.read_csv("water_potability.csv")
data.fillna(data.mean(), inplace=True)

# Display pie chart of potability ratio
st.subheader("🔍 Potability Distribution in Dataset")
fig1, ax1 = plt.subplots()
labels = ['Not Potable', 'Potable']
sizes = data['Potability'].value_counts().sort_index()
colors = ['#ff9999','#66b3ff']
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
ax1.axis('equal')
st.pyplot(fig1)

# Split features and target
features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
X = data[features]
y = data['Potability']

# Show toggle to balance the dataset
balance_option = st.selectbox("🔄 Select Sampling Strategy:", ["None", "Random Over Sampling", "Random Under Sampling"])
if balance_option == "Random Over Sampling":
    X, y = RandomOverSampler().fit_resample(X, y)
elif balance_option == "Random Under Sampling":
    X, y = RandomUnderSampler().fit_resample(X, y)

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to evaluate
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier()
}

metrics_df = pd.DataFrame(columns=["Model", "Accuracy"])
st.subheader("📊 Model Evaluation Metrics")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    metrics_df = metrics_df.append({"Model": name, "Accuracy": acc}, ignore_index=True)

    st.markdown(f"### ✅ {name}")
    st.write("Accuracy:", f"{acc*100:.2f}%")
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))
    st.text("Confusion Matrix")
    st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred),
                              columns=["Pred 0", "Pred 1"],
                              index=["True 0", "True 1"]))

# Bar chart for model comparison
st.subheader("📈 Model Accuracy Comparison")
fig2, ax2 = plt.subplots()
sns.barplot(x="Model", y="Accuracy", data=metrics_df, palette="viridis", ax=ax2)
ax2.set_ylim(0, 1)
ax2.set_ylabel("Accuracy")
st.pyplot(fig2)

# Single sample prediction
st.subheader("🔬 Single Sample Prediction")
model_choice = st.selectbox("Select Model for Prediction", list(models.keys()))
model = models[model_choice]

input_data = {}
cols = st.columns(3)
for i, feature in enumerate(features):
    input_data[feature] = cols[i % 3].number_input(f"{feature}", min_value=0.0, step=0.1)

if st.button("Predict Single Sample"):
    input_df = pd.DataFrame([input_data])[features]
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown(f"### Result: {'🟢 Potable' if prediction == 1 else '🔴 Not Potable'}")
    st.progress(probability)
    st.write(f"Prediction Confidence: {probability*100:.2f}% Potable")

# Batch prediction section
st.subheader("📥 Batch Prediction via CSV Upload")
uploaded_file = st.file_uploader("Upload a CSV file with same 9 feature columns", type=["csv"])

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    if not all(col in batch_df.columns for col in features):
        st.error("❌ Missing required columns in uploaded CSV.")
    else:
        predictions = model.predict(batch_df[features])
        probs = model.predict_proba(batch_df[features])[:, 1]
        batch_df['Prediction'] = predictions
        batch_df['Potable Probability (%)'] = (probs * 100).round(2)
        st.success("✅ Prediction complete")
        st.dataframe(batch_df)

        st.markdown("### 📊 Batch Prediction Summary")
        summary = batch_df['Prediction'].value_counts().sort_index()
        fig3, ax3 = plt.subplots()
        ax3.bar(['Not Potable', 'Potable'], summary.values, color=['red', 'green'])
        ax3.set_ylabel("Number of Samples")
        st.pyplot(fig3)

        st.download_button("Download Predictions", data=batch_df.to_csv(index=False),
                           file_name="batch_predictions.csv", mime="text/csv")
