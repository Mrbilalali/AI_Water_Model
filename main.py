import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(layout="wide")
st.title("🚰 Water Potability Classifier")

# Load and clean data
data = pd.read_csv("water_potability.csv")
st.subheader("🔎 Missing Values Before Cleaning")
st.dataframe(data.isna().sum().to_frame("Missing Count"))

data.fillna(data.mean(), inplace=True)

st.subheader("✅ Missing Values After Cleaning")
st.dataframe(data.isna().sum().to_frame("Missing Count"))

# Distribution plot
st.subheader("📊 Potability Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x='Potability', data=data, ax=ax1)
st.pyplot(fig1)

# Features and labels
X = data.drop("Potability", axis=1)
y = data["Potability"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=101)

# Model dictionary
models = {
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "MLPClassifier": MLPClassifier(max_iter=1000, random_state=42)
}

results = []
conf_matrices = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    results.append((name, round(score, 6)))
    conf_matrices[name] = confusion_matrix(y_test, y_pred)

# Display result table
st.subheader("📈 Accuracy Score Comparison")
result_df = pd.DataFrame(results, columns=["Algorithm", "Score"])
st.dataframe(result_df.sort_values("Score", ascending=False), use_container_width=True)

# Plot score bar chart
st.subheader("📉 Score by Algorithm")
fig2, ax2 = plt.subplots()
sns.barplot(x="Score", y="Algorithm", data=result_df.sort_values("Score", ascending=True), palette="viridis", ax=ax2)
for i, row in result_df.iterrows():
    ax2.text(row[1] + 0.005, i, f"{row[1]:.3f}", va='center')
st.pyplot(fig2)

# Confusion matrices
st.subheader("🧮 Confusion Matrices")
model_to_view = st.selectbox("Select model to view confusion matrix:", list(conf_matrices.keys()))
fig3, ax3 = plt.subplots()
sns.heatmap(conf_matrices[model_to_view], annot=True, fmt="d", cmap="Blues", ax=ax3)
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")
ax3.set_title(f"Confusion Matrix - {model_to_view}")
st.pyplot(fig3)

st.markdown("---")
st.info("Upload CSV support & advanced prediction features can be added next. Let me know!")
