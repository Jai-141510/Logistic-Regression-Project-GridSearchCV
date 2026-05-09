# 🫀 Heart Disease Prediction
### Logistic Regression with GridSearchCV

> Binary classification model that predicts whether a patient has heart disease based on clinical features like age, cholesterol, resting blood pressure, and chest pain type.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-red?style=flat&logo=jupyter)

---

## 📋 Dataset

| Property | Detail |
|---|---|
| Source | UCI Heart Disease Dataset |
| Patients | 303 |
| Features | 13 (age, cholesterol, chest pain type, etc.) |
| Target | 0 = No Heart Disease, 1 = Heart Disease |

---

## 🔍 What Was Done

**EDA** — checked for nulls (none), confirmed data types were clean, and visualised relationships between key continuous features. The pairplot showed significant class overlap, which hinted the model would have to work harder than usual.

**Pipeline** — wrapped everything in a sklearn Pipeline (StandardScaler → LogisticRegression) rather than scaling upfront. This prevents data leakage during cross validation since the scaler only ever sees training folds.

**Baseline** — ran 5-fold cross validation before any tuning. Got 82% accuracy and 87% recall straight out of the box, which set the benchmark.

**Overfitting check** — train vs test gap was 3-4%. The model was generalising fine.

**GridSearchCV** — tested L1, L2, and ElasticNet penalties across 7 C values (log scale). First run optimised for recall and returned 1.0, which sounds great until I checked the classification report and realised that the model was predicting heart disease for everyone. Switched scoring to F1, which forced a more honest result.

**Final model** — ElasticNet with C=0.01 and l1_ratio=0.1. Mostly Ridge behaviour, which makes sense since most of the 13 features are medically relevant and shouldn't be zeroed out completely.

---

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | 81% |
| Recall (heart disease) | 88% |
| Precision (heart disease) | 78% |
| F1 | 82% |

88% recall means the model catches 88 out of every 100 real heart disease patients. For a first-pass screening tool built on 303 rows, that's a reasonable result. A more powerful model (Random Forest, XGBoost) with more data would be the natural next step.

---

## ⚠️ Key Lesson

Optimising purely for recall caused the model to flag **everyone** as sick — hitting 100% recall but near-zero precision. Always evaluate with a balanced metric like F1 before trusting a result that looks too good.

---

## 🛠️ Stack

`Python` &nbsp;|&nbsp; `scikit-learn` &nbsp;|&nbsp; `pandas` &nbsp;|&nbsp; `seaborn` &nbsp;|&nbsp; `matplotlib`

---

## 📁 Files

```
├── Logistic-Regression-Project-GridSearchCV_.ipynb   # Full notebook
└── heart.csv                                          # Dataset
```
