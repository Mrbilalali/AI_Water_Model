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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

# --- Constants and Feature List ---
FEATURES = ['ph','Hardness','Solids','Chloramines','Sulfate',
            'Conductivity','Organic_carbon','Trihalomethanes','Turbidity']
TARGET = 'Potability'

st.title("💧 Water Potability Prediction")

# --- Load & Clean Data ---
@st.cache_data
def load_data(path='water_potability.csv'):
    df = pd.read_csv(path)
    df = df.fillna(df.mean())
    return df

df = load_data()

# Show class distribution
st.subheader("Dataset Potability Distribution")
fig, ax = plt.subplots()
sns.countplot(x=TARGET, data=df, ax=ax)
ax.set_xticklabels(['Unsafe (0)','Safe (1)'])
st.pyplot(fig)

# --- Prepare Train/Test ---
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Balance training data with oversampling
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)

# --- Train Models & Record Scores ---
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(),
    'NaiveBayes': GaussianNB(),
    'MLP': MLPClassifier(max_iter=500, random_state=42)
}
scores = []
for name, model in models.items():
    model.fit(X_res, y_res)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    scores.append({'Algorithm': name, 'Score': acc})
score_df = pd.DataFrame(scores).sort_values('Score', ascending=False)

# --- Model Comparison Graph ---
st.subheader("Model Accuracy Comparison on Test Set")
fig2, ax2 = plt.subplots()
sns.barplot(data=score_df, x='Algorithm', y='Score', ax=ax2)
ax2.set_ylim(0,1)
st.pyplot(fig2)

# Select default model (best)
best_name = score_df.iloc[0]['Algorithm']
best_model = models[best_name]

# --- Single Sample Prediction ---
st.subheader("🔍 Single Sample Prediction")
st.info(f"Default model: {best_name}")
input_vals = {}
for feature in FEATURES:
    default = float(df[feature].mean())
    input_vals[feature] = st.number_input(feature, value=default)
input_df = pd.DataFrame([input_vals])[FEATURES]
if st.button("Predict Single Sample"):
    pred = best_model.predict(input_df)[0]
    proba = best_model.predict_proba(input_df)[0][1]
    st.metric("Potable Probability", f"{proba*100:.2f}%")
    if pred==1:
        st.success("Prediction: SAFE to drink")
    else:
        st.error("Prediction: NOT safe to drink")

# --- Batch Prediction ---
st.subheader("📤 Batch Prediction via CSV Upload")
batch_file = st.file_uploader("Upload CSV", type='csv')
if batch_file:
    batch = pd.read_csv(batch_file)
    # Clean and validate
    batch = batch.fillna(df.mean())
    missing = [c for c in FEATURES if c not in batch.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        batch = batch[FEATURES]
        preds = best_model.predict(batch)
        probas = best_model.predict_proba(batch)[:,1]
        batch['Prediction'] = np.where(preds==1,'Safe','Unsafe')
        batch['Probability (%)'] = (probas*100).round(2)
        st.write(batch)
        # summary
        summary = batch['Prediction'].value_counts(normalize=True)*100
        st.subheader("Batch Potability Distribution %")
        fig3, ax3 = plt.subplots()
        summary.plot(kind='bar', ax=ax3)
        ax3.set_ylabel('Percentage')
        st.pyplot(fig3)
        # confusion vs assumed true if provided
        if TARGET in batch.columns:
            cm = confusion_matrix(batch[TARGET], preds)
            fig4, ax4 = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax4)
            ax4.set_title('Confusion Matrix on Batch')
            st.pyplot(fig4)
