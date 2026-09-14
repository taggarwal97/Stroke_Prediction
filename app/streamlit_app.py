# ===================================================================
# Stroke Prediction — Streamlit Dashboard
# Tabs: Risk Prediction | Cluster Insights | EDA Dashboard
# ===================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Stroke Risk Dashboard", layout="wide")

# -------------------------------------------------------------------
# Load saved models and artifacts (paths relative to project root,
# since Streamlit is run via `streamlit run app/streamlit_app.py`
# from the project root)
# -------------------------------------------------------------------
MODELS_DIR = Path("outputs/models")
FIGURES_DIR = Path("outputs/figures")

@st.cache_resource
def load_artifacts():
    knn_model = joblib.load(MODELS_DIR / "knn_model.pkl")
    knn_scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
    kmeans_model = joblib.load(MODELS_DIR / "kmeans_model.pkl")
    cluster_scaler = joblib.load(MODELS_DIR / "cluster_scaler.pkl")
    kmedoids_centers = joblib.load(MODELS_DIR / "kmedoids_centers.pkl")
    return knn_model, knn_scaler, feature_columns, kmeans_model, cluster_scaler, kmedoids_centers

try:
    knn_model, knn_scaler, feature_columns, kmeans_model, cluster_scaler, kmedoids_centers = load_artifacts()
    models_loaded = True
except FileNotFoundError as e:
    models_loaded = False
    st.error(f"Could not load saved models: {e}. Make sure you've run notebooks 02 and 03 first.")

@st.cache_data
def load_clustered_data():
    return pd.read_csv("data/processed/clustered_stroke_data.csv")

# -------------------------------------------------------------------
# Sidebar — patient input form (shared across tabs)
# -------------------------------------------------------------------
st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 0, 100, 45)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
hypertension = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.sidebar.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.sidebar.selectbox("Ever Married", ["No", "Yes"])
work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.sidebar.slider("Average Glucose Level", 50.0, 300.0, 100.0)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
smoking_status = st.sidebar.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

st.title("🧠 Stroke Risk Dashboard")
st.caption("Exploratory analysis, KNN risk prediction, and cluster-based risk segmentation on the Kaggle Stroke Prediction dataset.")

tab1, tab2, tab3 = st.tabs(["🎯 Risk Prediction", "🧩 Cluster Insights", "📊 EDA Dashboard"])

# ===================================================================
# TAB 1 — Risk Prediction (KNN)
# ===================================================================
with tab1:
    st.subheader("Stroke Risk Prediction")

    if not models_loaded:
        st.warning("Models not loaded — prediction unavailable.")
    else:
        # Build a single-row DataFrame matching the training data's raw columns
        input_dict = {
            'gender': gender,
            'age': age,
            'hypertension': 1 if hypertension == "Yes" else 0,
            'heart_disease': 1 if heart_disease == "Yes" else 0,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }
        input_df = pd.DataFrame([input_dict])

        # One-hot encode the same way as training (drop_first=True)
        categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

        # Align columns with what the model was trained on — add any
        # missing dummy columns as 0, and enforce the same column order
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[feature_columns]

        input_scaled = knn_scaler.transform(input_encoded)

        if st.button("Predict Stroke Risk", type="primary"):
            prediction = knn_model.predict(input_scaled)[0]
            probability = knn_model.predict_proba(input_scaled)[0][1]

            col1, col2 = st.columns(2)
            with col1:
                if prediction == 1:
                    st.error(f"⚠️ Elevated Risk — model predicts stroke risk")
                else:
                    st.success(f"✅ Lower Risk — model predicts no stroke risk")
            with col2:
                st.metric("Predicted Stroke Probability", f"{probability*100:.1f}%")

            st.caption(
                "This model prioritizes catching true stroke cases (64% recall) "
                "over minimizing false alarms — appropriate for a screening tool, "
                "not a diagnosis. Always consult a medical professional."
            )

# ===================================================================
# TAB 2 — Cluster Insights
# ===================================================================
with tab2:
    st.subheader("Which Risk Segment Does This Patient Fall Into?")

    if not models_loaded:
        st.warning("Models not loaded — clustering unavailable.")
    else:
        cluster_input = pd.DataFrame([{
            'age': age,
            'hypertension': 1 if hypertension == "Yes" else 0,
            'heart_disease': 1 if heart_disease == "Yes" else 0,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'gender': 0 if gender == "Male" else 1,
            'ever_married': 1 if ever_married == "Yes" else 0
        }])

        cluster_scaled = cluster_scaler.transform(cluster_input)
        assigned_cluster = kmeans_model.predict(cluster_scaled)[0]

        try:
            df_clustered = load_clustered_data()
            cluster_stats = df_clustered.groupby('kmeans_cluster')['stroke'].agg(['mean', 'count'])
            cluster_stroke_rate = cluster_stats.loc[assigned_cluster, 'mean'] * 100
            cluster_size = int(cluster_stats.loc[assigned_cluster, 'count'])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Assigned Cluster (K-Means)", f"Cluster {assigned_cluster}")
            with col2:
                st.metric("Historical Stroke Rate in This Cluster", f"{cluster_stroke_rate:.1f}%",
                          help=f"Based on {cluster_size} patients with similar profiles in the training data")

            baseline_rate = df_clustered['stroke'].mean() * 100
            if cluster_stroke_rate > baseline_rate * 1.5:
                st.warning(f"This cluster's stroke rate ({cluster_stroke_rate:.1f}%) is notably "
                           f"above the overall dataset average ({baseline_rate:.1f}%).")
            else:
                st.info(f"This cluster's stroke rate ({cluster_stroke_rate:.1f}%) is close to or "
                       f"below the overall dataset average ({baseline_rate:.1f}%).")

            st.markdown("**Cluster profile comparison:**")
            profile_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
            profile = df_clustered.groupby('kmeans_cluster')[profile_cols].mean()
            profile['stroke_rate_%'] = df_clustered.groupby('kmeans_cluster')['stroke'].mean() * 100
            st.dataframe(profile.style.highlight_max(axis=0, subset=['stroke_rate_%'], color='#ffcccc'))

        except FileNotFoundError:
            st.info(f"Assigned Cluster: {assigned_cluster} (historical comparison data not found)")

# ===================================================================
# TAB 3 — EDA Dashboard
# ===================================================================
with tab3:
    st.subheader("Exploratory Data Analysis")
    st.caption("Charts generated in the project's EDA and clustering notebooks.")

    figure_files = {
        "Stroke Class Distribution": "stroke_class_distribution.png",
        "Numeric Feature Distributions": "numeric_distributions.png",
        "Stroke Rate by Age Group": "stroke_rate_by_age_group.png",
        "Stroke Rate by Category": "stroke_rate_by_category.png",
        "Correlation Heatmap": "correlation_heatmap.png",
        "Clustering Method Comparison": "clustering_method_comparison.png",
        "Clusters (PCA 2D)": "clusters_pca_2d.png",
    }

    cols = st.columns(2)
    for i, (title, filename) in enumerate(figure_files.items()):
        filepath = FIGURES_DIR / filename
        with cols[i % 2]:
            st.markdown(f"**{title}**")
            if filepath.exists():
                st.image(str(filepath))
            else:
                st.caption(f"⚠️ {filename} not found — run the notebooks to generate it.")