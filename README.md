# titanic-survival-classification
Predict passenger survival based on demographic, cabin, and ticket information.

**Dataset:** The classic Kaggle Titanic dataset.

**Key Skills:** Data Cleaning, Text Extraction, Classification, Ensemble Modeling.

## 🛠️ Feature Engineering & Preprocessing
This dataset required heavy cleaning and extraction to make the raw text readable for machine learning algorithms:
- **Text Extraction:** Extracted passenger titles (Mr., Mrs., Miss, Master) from the `Name` column to estimate age and social status.
- **Cabin Parsing:** Extracted deck levels (first letter) from the `Cabin` column and created a `cabin_multiple` count feature.
- **Imputation:** Filled missing Age and Fare values using median values strictly from the *training* set to prevent data leakage.
- **Scaling:** Applied Log Transformation to `Fare` to fix extreme skewness, and used `StandardScaler` for all numeric features.

## 🤖 Models Evaluated
- Logistic Regression
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Random Forest
- **Support Vector Machine (Champion Model)**

## 🏆 Results
- Conducted Hyperparameter Tuning via `GridSearchCV` (testing RBF, Linear, and Poly kernels).
- The optimized **Support Vector Machine (SVM)** achieved an accuracy of **~77.7%** on the Kaggle leaderboard, outperforming the gender-based baseline.
