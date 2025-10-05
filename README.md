
# 🫀 Heart Disease Prediction using Machine Learning

Access from : https://colab.research.google.com/drive/1xLjoSo8fz43E9Ss9bUmuHdancA9ly5J_?usp=sharing

This repository contains the full pipeline and analysis for predicting heart disease using supervised and unsupervised machine learning techniques. Built as part of the CSE422 Artificial Intelligence course at BRAC University (Summer 2025), this project explores classification models and clustering methods on a real-world medical dataset.

## 📁 Project Structure

- `EDA & Preprocessing`: Data cleaning, encoding, and feature scaling
- `Model Training`: KNN, Logistic Regression, Naive Bayes, Neural Network (MLP)
- `Evaluation`: Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC
- `Unsupervised Learning`: K-Means clustering with PCA visualization
- `Reporting`: Comparative analysis and conclusions

## 📊 Dataset Overview

- **Rows**: 920
- **Features**: 16 (after dropping noisy/missing columns)
- **Target**: `num` (heart disease severity: 0–4)
- **Problem Type**: Multiclass classification (also simplified to binary for analysis)

### 🔍 Feature Types

| Type           | Features                                                                 |
|----------------|--------------------------------------------------------------------------|
| Quantitative   | `age`, `chol`, `oldpeak`, `thalch`, `trestbps`                          |
| Binary Categorical | `sex`, `fbs`, `exang`                                                  |
| Multi-level Categorical | `cp`, `restecg`, `slope`, `thal`                                  |

## 🧼 Data Preprocessing

- Dropped columns with >50% missing values (`ca`, `thal`)
- Imputed missing values using:
  - Mode (for categorical)
  - Mean/Median (for numerical)
- One-Hot Encoding for multi-category features
- StandardScaler applied to numerical features

## 🔀 Dataset Splitting

- **Train/Test Split**: 80/20 stratified
- **Train Shape**: (734, 19)
- **Test Shape**: (184, 19)

## 🤖 Models Used

| Model               | Highlights                                                                 |
|---------------------|---------------------------------------------------------------------------|
| Logistic Regression | Best overall performance (AUC = 0.774)                                    |
| KNN                 | Highest precision (~0.60), good alternative                               |
| Neural Network (MLP)| Competitive but resource-intensive                                        |
| Naive Bayes         | Poor performance, used for baseline comparison                            |

## 📈 Evaluation Metrics

- Accuracy, Precision, Recall, F1-score
- Confusion Matrices
- ROC-AUC (Binary & Multiclass)

### ROC-AUC (Binary)

| Model               | AUC Score |
|---------------------|-----------|
| Logistic Regression | 0.774     |
| Neural Network      | 0.744     |
| KNN                 | 0.684     |
| Naive Bayes         | 0.748     |

## 🔍 Unsupervised Learning

- **K-Means Clustering**: Optimal k = 6 (via Elbow Method)
- **PCA Visualization**: Revealed natural groupings aligned with disease severity

## 🧠 Key Insights

- Multiclass imbalance affects model performance
- Binary simplification improves accuracy but loses clinical granularity
- Logistic Regression offers interpretability and stability
- Neural Networks require more tuning and data to outperform simpler models

## 🏁 Conclusion

This project demonstrates the challenges and potential of applying machine learning to healthcare data. With careful preprocessing and model selection, even simple algorithms can yield meaningful insights. Future work could explore ensemble methods, deeper architectures, or external clinical datasets.

