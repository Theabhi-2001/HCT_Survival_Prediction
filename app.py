import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ========================
# Load Model & Preprocessor
# ========================
voting_clf = joblib.load("voting_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
num_cols = joblib.load("num_cols.pkl")
cat_cols = joblib.load("cat_cols.pkl")

st.set_page_config(page_title="HCT EFS Prediction", layout="wide")

st.title("🩺 Hematopoietic Cell Transplantation - Event-Free Survival Prediction")

st.write("""
Upload a CSV file containing **pre-transplant patient features** to predict Event-Free Survival (EFS).
The model combines **Random Forest, XGBoost, and Logistic Regression**  
and provides explainability via **SHAP** for the XGBoost part of the ensemble.
""")

# ========================
# File Upload
# ========================
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file:
    user_data = pd.read_csv(uploaded_file)
    st.write("### 📋 Uploaded Data Preview", user_data.head())

    # Remove leakage columns if present
    for col in ["efs_time", "efs_time_bin"]:
        if col in user_data.columns:
            user_data.drop(columns=[col], inplace=True)

    # Ensure all expected columns exist
    required_cols = num_cols + cat_cols
    missing_cols = set(required_cols) - set(user_data.columns)
    for col in missing_cols:
        user_data[col] = np.nan  # Fill missing columns with NaN

    # Keep only required columns for preprocessing
    user_data_proc = user_data[required_cols]

    # Preprocess
    X_user_p = preprocessor.transform(user_data_proc)

    # ========================
    # Prediction
    # ========================
    if st.button("🔮 Predict EFS"):
        probas = voting_clf.predict_proba(X_user_p)[:, 1]
        preds = (probas >= 0.5).astype(int)

        # Use 'id' column if exists, otherwise use the DataFrame index as ID
        if "id" in user_data.columns:
            id_col = user_data["id"]
        else:
            id_col = user_data.index

        output_df = pd.DataFrame({
            "ID": id_col,
            "EFS_Probability": np.round(probas, 4),
            "EFS_Prediction": preds
        })

        st.write("### 🧾 Prediction Results")
        st.dataframe(output_df)

    # ========================
    # SHAP Summary Plot
    # ========================
    if st.button("📊 Show SHAP Summary Plot"):
        trained_xgb = voting_clf.named_estimators_["xgb"]

        # Feature names after preprocessing
        feature_names = (
            num_cols +
            list(preprocessor.named_transformers_["cat"]
                 .named_steps["ohe"]
                 .get_feature_names_out(cat_cols))
        )

        # Compute SHAP values
        explainer = shap.TreeExplainer(trained_xgb)
        shap_values = explainer.shap_values(preprocessor.transform(user_data_proc))

        # Filter out any efs_time-related features (shouldn't exist but for safety)
        drop_idx = [i for i, f in enumerate(feature_names) if "efs_time" in f]
        filtered_feature_names = [f for i, f in enumerate(feature_names) if i not in drop_idx]
        shap_values_filt = np.delete(shap_values, drop_idx, axis=1)
        X_user_filt = np.delete(preprocessor.transform(user_data_proc), drop_idx, axis=1)

        # Plot SHAP summary with controlled size and font size
        shap.summary_plot(
            shap_values_filt,
            pd.DataFrame(X_user_filt, columns=filtered_feature_names),
            show=False,
            plot_size=(7, 3),    # Adjust size to fit screen nicely
            max_display=12       # Show only top 12 features to keep plot compact
        )

        # Reduce feature names font size on the left
        plt.gca().set_yticklabels(
            plt.gca().get_yticklabels(),
            fontsize=8    # Smaller font size for feature names
        )
        
        plt.title("SHAP Summary Plot")
        st.pyplot(plt.gcf())
