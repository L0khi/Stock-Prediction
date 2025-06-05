# ======================================
# Logistic Regression with SMOTE
# ======================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ========== Load Dataset ==========
data = pd.read_csv(r"C:\Users\kulwant dhillon\Downloads\bank refined.csv")  # Ensure this file is in your script folder

# ========== Define Features and Target ==========
y = data['y']
X = data[['age', 'job', 'marital', 'education', 'housing', 'loan',
          'balance', 'day', 'duration', 'campaign', 'previous']]

# ========== One-Hot Encode Categorical Columns ==========
categorical_cols = ['job', 'marital', 'education', 'housing', 'loan']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# ========== Train-Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)

# ========== Normalize Features ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== Apply SMOTE for Class Balancing ==========
print("Before SMOTE:", np.bincount(y_train.map({'no': 0, 'yes': 1})))
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print("After SMOTE:", np.bincount(y_train_res.map({'no': 0, 'yes': 1})))

# ========== Train Logistic Regression ==========
model = LogisticRegression(max_iter=1000)
model.fit(X_train_res, y_train_res)

# ========== Predict & Evaluate ==========
y_pred = model.predict(X_test_scaled)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
