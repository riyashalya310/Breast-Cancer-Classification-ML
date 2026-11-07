# Breast Cancer Classification Using Machine Learning

## 1. Project Overview

This project implements and compares five supervised machine learning classification algorithms to classify breast cancer tumours as malignant or benign.

The objective is to evaluate different classification techniques and identify the model that provides the best overall performance using multiple evaluation metrics.

## 2. Problem Statement

Breast cancer classification is a binary classification problem in which tumour characteristics are used to predict whether a tumour is malignant or benign.

The project applies five machine learning algorithms to the Breast Cancer Wisconsin Diagnostic dataset and compares their predictive performance.

This project is developed for academic and educational purposes and is not intended for clinical diagnosis.

## 3. Dataset Description

**Dataset:** Breast Cancer Wisconsin Diagnostic Dataset

**Source:** UCI Machine Learning Repository (accessed through scikit-learn's built-in dataset loader).

- Number of instances: 569
- Number of input features: 30
- Number of target classes: 2
- Missing values: 0
- Duplicate records: 0

**Target classes:**
- 0 – Malignant
- 1 – Benign

The dataset contains numerical measurements describing cell nuclei characteristics.

## 4. Machine Learning Algorithms

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Feature scaling was applied to Logistic Regression, KNN and SVM. Decision Tree and Gaussian Naive Bayes were trained using the original feature values.

## 5. Evaluation Metrics

The models were evaluated using:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)

Precision, Recall and F1-score use Benign (class 1) as the positive class in the primary comparison. Malignant-class recall was also examined separately.

## 6. Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

*The values above are rounded. The final comparison CSV contains the full-precision results.*

## 7. Model Observations

- Logistic Regression achieved excellent classification performance, with an accuracy of 98.25% and an MCC of 0.9623.
- Support Vector Machine achieved similarly strong performance and was very close to Logistic Regression.
- KNN performed well, achieving an accuracy of 95.61%.
- Gaussian Naive Bayes achieved a high AUC but comparatively lower accuracy and MCC.
- Decision Tree recorded the lowest overall performance among the five models.

Logistic Regression achieved a malignant-class recall of 0.9762 and one malignant false negative in the test set.

Based on the preliminary comparison, Logistic Regression is a strong candidate for the best-performing model. The final model selection is based on the full-precision comparison results.

## 8. Project Structure

```text
ML_Assignment_2/
│
├── ML_Assignment_2.ipynb
├── breast_cancer_dataset.csv
├── test_data.csv
├── model_evaluation_results.csv
├── final_model_comparison.csv
├── README.md
├── requirements.txt
├── app.py
│
└── models/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── svm.pkl
    └── scaler.pkl
```

## 9. How to Run the Project

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Upload the labelled test dataset, select a trained model, and view the model's evaluation results.

## 10. Links

- **GitHub Repository:** https://github.com/riyashalya310/Breast-Cancer-Classification-ML
- **Live Streamlit Application:** https://breast-cancer-classification-ml.streamlit.app/
- **BITS Virtual Lab Execution Screenshot:** To be included in the final assignment PDF