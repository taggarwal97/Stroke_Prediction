
🔗 **Live App:** https://strokeprediction-ahyxktcamqxxcbxbvvvas7.streamlit.app/

# Stroke Prediction — EDA, Classification & Clustering

An end-to-end analysis of the Kaggle Stroke Prediction dataset: exploring 
patient health/demographic data, building a supervised classification 
model (KNN) to predict stroke risk, and comparing five clustering 
techniques to identify natural risk-based patient segments.

## Project Structure
```
├── data/
│ ├── raw/ # Original Kaggle dataset
│ └── processed/ # Cleaned & clustered data
├── notebooks/
│ ├── 01_eda.ipynb # Data cleaning, univariate/bivariate analysis
│ ├── 02_classification_knn.ipynb # KNN classifier with imbalance handling
│ └── 03_clustering.ipynb # K-means, K-medians, K-medoids, Hierarchical, DBSCAN
├── outputs/
│ ├── figures/ # Saved chart images
│ └── models/ # Saved trained models
├── app/
│ └── streamlit_app.py # Interactive risk prediction app
├── requirements.txt
```


## Key Findings

**Classification (KNN):**
- Target is heavily imbalanced (~5% stroke cases); accuracy alone is misleading
- After addressing imbalance via SMOTE, final model achieves **64% recall** 
  and **0.72 ROC-AUC** for the stroke class — an intentional precision/recall 
  tradeoff appropriate for a screening context, where missing a true case 
  is costlier than a false alarm

**Clustering:**
- Four independent methods (K-Means, K-Medians, K-Medoids, Hierarchical) 
  consistently identified patient clusters with **10–18% stroke rates** — 
  2 to 4x the dataset baseline — characterized by higher age and hypertension
- No method produced a purely "stroke" cluster, consistent with stroke 
  remaining a probabilistic outcome even within elevated-risk groups

## Dataset
[Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

## Setup
```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Run the app
```bash
streamlit run app/streamlit_app.py
```