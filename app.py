
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import os


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)

st.title("Breast Cancer Classification Using Machine Learning")

st.write(
    "This application compares five machine learning classification "
    "models trained on the Breast Cancer Wisconsin Diagnostic dataset."
)

st.info(
    "Educational project only. This application is not intended "
    "for clinical diagnosis or medical decision-making."
)


# --------------------------------------------------
# 2. LOAD TRAINED MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    model_paths = {
        "Logistic Regression": "models/logistic_regression.pkl",
        "Decision Tree": "models/decision_tree.pkl",
        "KNN": "models/knn.pkl",
        "Gaussian Naive Bayes": "models/naive_bayes.pkl",
        "Support Vector Machine": "models/svm.pkl"
    }

    loaded_models = {}

    for model_name, path in model_paths.items():

        loaded_models[model_name] = joblib.load(path)

    scaler = joblib.load("models/scaler.pkl")

    return loaded_models, scaler


# --------------------------------------------------
# 3. LOAD EXPECTED FEATURE NAMES
# --------------------------------------------------

@st.cache_data
def load_feature_names():

    reference_data = pd.read_csv("test_data.csv")

    return [
        column for column in reference_data.columns
        if column != "target"
    ]


try:

    models, scaler = load_models()

    feature_columns = load_feature_names()

    st.success("All trained models loaded successfully!")

except Exception as e:

    st.error(f"Error loading project files: {e}")

    st.stop()


# --------------------------------------------------
# 4. MODEL SELECTION
# --------------------------------------------------

st.subheader("Select Machine Learning Model")

selected_model_name = st.selectbox(
    "Choose a classification algorithm:",
    list(models.keys())
)

selected_model = models[selected_model_name]


scaled_models = [
    "Logistic Regression",
    "KNN",
    "Support Vector Machine"
]


# --------------------------------------------------
# 5. UPLOAD TEST DATA
# --------------------------------------------------

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a labelled CSV file containing the 30 features and target column",
    type=["csv"]
)

st.caption(
    "Use test_data.csv from the project for evaluation. "
    "The target column is required to calculate performance metrics."
)


if uploaded_file is not None:

    try:

        uploaded_data = pd.read_csv(uploaded_file)

        st.write("Uploaded Dataset Preview")

        st.dataframe(uploaded_data.head())

        # ------------------------------------------
        # 6. VALIDATE DATASET
        # ------------------------------------------

        missing_columns = [
            column for column in feature_columns
            if column not in uploaded_data.columns
        ]

        if missing_columns:

            st.error(
                f"Missing required feature columns: {missing_columns}"
            )

            st.stop()

        if "target" not in uploaded_data.columns:

            st.error(
                "The uploaded CSV must contain the target column "
                "to calculate evaluation metrics."
            )

            st.stop()

        if uploaded_data.empty:

            st.error("The uploaded CSV contains no records.")

            st.stop()

        # Select features in the original training order
        X_uploaded = uploaded_data[feature_columns].copy()

        # Convert feature values to numeric
        X_uploaded = X_uploaded.apply(
            pd.to_numeric,
            errors="coerce"
        )

        if X_uploaded.isnull().any().any():

            st.error(
                "The uploaded dataset contains missing or "
                "non-numeric feature values."
            )

            st.stop()

        if not np.isfinite(X_uploaded.to_numpy()).all():

            st.error(
                "The uploaded dataset contains infinite values."
            )

            st.stop()

        # Validate target labels
        y_actual = pd.to_numeric(
            uploaded_data["target"],
            errors="coerce"
        )

        if y_actual.isnull().any() or not y_actual.isin([0, 1]).all():

            st.error(
                "Target values must be 0 (Malignant) or 1 (Benign)."
            )

            st.stop()

        y_actual = y_actual.astype(int)

        if y_actual.nunique() < 2:

            st.warning(
                "The dataset contains only one target class. "
                "AUC cannot be calculated."
            )

        # ------------------------------------------
        # 7. APPLY SCALING WHEN REQUIRED
        # ------------------------------------------

        if selected_model_name in scaled_models:

            X_processed = scaler.transform(X_uploaded)

        else:

            X_processed = X_uploaded

        # ------------------------------------------
        # 8. MAKE PREDICTIONS
        # ------------------------------------------

        predictions = selected_model.predict(X_processed)

        # Probability for class 1 (Benign)
        class_one_index = list(selected_model.classes_).index(1)

        probabilities = selected_model.predict_proba(
            X_processed
        )[:, class_one_index]

        # ------------------------------------------
        # 9. DISPLAY PREDICTIONS
        # ------------------------------------------

        st.subheader("Prediction Results")

        prediction_data = pd.DataFrame({
            "Actual Class": y_actual.map({
                0: "Malignant",
                1: "Benign"
            }),
            "Predicted Class": pd.Series(predictions).map({
                0: "Malignant",
                1: "Benign"
            }),
            "Benign Probability": probabilities.round(4)
        })

        st.dataframe(
            prediction_data,
            use_container_width=True
        )

        # ------------------------------------------
        # 10. CALCULATE EVALUATION METRICS
        # ------------------------------------------

        accuracy = accuracy_score(y_actual, predictions)

        precision = precision_score(
            y_actual,
            predictions,
            pos_label=1,
            zero_division=0
        )

        recall = recall_score(
            y_actual,
            predictions,
            pos_label=1,
            zero_division=0
        )

        f1 = f1_score(
            y_actual,
            predictions,
            pos_label=1,
            zero_division=0
        )

        mcc = matthews_corrcoef(y_actual, predictions)

        auc = (
            roc_auc_score(y_actual, probabilities)
            if y_actual.nunique() == 2
            else None
        )

        # ------------------------------------------
        # 11. DISPLAY METRICS
        # ------------------------------------------

        st.subheader("Model Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")

        col2.metric(
            "AUC",
            f"{auc:.4f}" if auc is not None else "N/A"
        )

        col3.metric("MCC", f"{mcc:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Precision", f"{precision:.4f}")

        col5.metric("Recall", f"{recall:.4f}")

        col6.metric("F1 Score", f"{f1:.4f}")

        st.caption(
            "Precision, Recall and F1 use Benign (class 1) "
            "as the positive class."
        )

        # ------------------------------------------
        # 12. CONFUSION MATRIX
        # ------------------------------------------

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_actual,
            predictions,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Malignant", "Benign"]
        ).plot(
            cmap="Blues",
            ax=ax,
            colorbar=False
        )

        ax.set_title(
            f"Confusion Matrix - {selected_model_name}"
        )

        st.pyplot(fig)

        plt.close(fig)

        # ------------------------------------------
        # 13. CLASSIFICATION REPORT
        # ------------------------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y_actual,
            predictions,
            labels=[0, 1],
            target_names=["Malignant", "Benign"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

        # ------------------------------------------
        # 14. MALIGNANT CLASS ANALYSIS
        # ------------------------------------------

        malignant_recall = recall_score(
            y_actual,
            predictions,
            pos_label=0,
            zero_division=0
        )

        malignant_false_negatives = cm[0, 1]

        st.subheader("Malignant Class Analysis")

        col1, col2 = st.columns(2)

        col1.metric(
            "Malignant Recall",
            f"{malignant_recall:.4f}"
        )

        col2.metric(
            "Malignant False Negatives",
            int(malignant_false_negatives)
        )

        # ------------------------------------------
        # 15. DOWNLOAD PREDICTIONS
        # ------------------------------------------

        result_csv = prediction_data.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=result_csv,
            file_name="prediction_results.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(f"An error occurred while processing the dataset: {e}")

else:

    st.warning(
        "Please upload test_data.csv to begin model evaluation."
    )
