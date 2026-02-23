# ===============================
# ML Project: Predict Mental Health Risk in Students
# ===============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv(r"C:\ML_Project.py\mentalhealth_prediction.csv")


# Clean column names (VERY IMPORTANT)
df.columns = df.columns.str.replace("\n", " ", regex=False).str.strip()

# Actual column name for Age after cleaning:
age_col = "Q1: What is your age?  ● ___ years"

# -----------------------------
# 2. Data Cleaning / Preprocessing
# -----------------------------

# Fix Age
df['Age'] = (
    df[age_col]
    .astype(str)
    .str.extract(r"(\d+)")
    .astype(float)
)

# Remove unrealistic ages
df.loc[(df['Age'] < 10) | (df['Age'] > 60), 'Age'] = np.nan

# Mappings
sleep_map = {
    'Less than 5': 4,
    '5-6': 5.5,
    '6-7': 6.5,
    '7-8': 7.5,
    'More than 7 hours': 8,
    'More than 8': 8.5,
    'More than 8 hours': 8.5
}

study_map = {
    '0-1 hours': 1,
    '1-2 hours': 1.5,
    '2-3 hours': 2.5,
    '3-4 hours': 3.5,
    '4-5 hours': 4.5,
    '5-6 hours': 5.5,
    '6-7 hours': 6.5,
    'More than 7 hours': 8
}

screen_map = {
    '1-2 hours': 1.5,
    '2-3 hours': 2.5,
    '3-4 hours': 3.5,
    '4-5 hours': 4.5,
    '5-6 hours': 5.5,
    '7-9 hours': 8,
    'More than 9 hours': 10
}

sleep_quality_map = {
    "Very good": 5, "Good": 4, "Fair": 3, "Poor": 2, "Very poor": 1
}

physical_map = {
    "0 hours": 0, "0.5 hour": 0.5, "1 hour": 1, "1.5 hours": 1.5, "2+ hours": 2.5
}

stress_map = {"Yes": 1, "No": 0, "Sometime": 0.5}
support_map = {"Yes": 1, "No": 0, "Sometime": 0.5}

# Apply mappings
df['Q3'] = df['Q3: How many hours do you sleep per day?'].map(sleep_map)
df['Q4'] = df['Q4: On average, how many hours do you study per day?'].map(study_map)
df['Q5'] = df['Q5: How many hours do you spend daily on screens (mobile, laptop, TV)?'].map(screen_map)
df['Q6'] = df['Q6: How would you rate your sleep quality over the past two weeks?'].map(sleep_quality_map)
df['Q7'] = df['Q7: How much time do you spend on physical activity/exercise per day?'].map(physical_map)
df['Q9'] = df['Q9: Do you frequently feel stressed, anxious, or overwhelmed?'].map(stress_map)
df['Q10'] = df['Q10: Do you feel supported by friends/family?'].map(support_map)

# Academic pressure (extract first digit)
df['Academic_pressure'] = (
    df['Q8: How much academic pressure do you feel?  Rate on a scale from 1 to 5:']
    .astype(str)
    .str.extract(r"(\d)")
    .astype(float)
)

# Clean final dataset
df_clean = df[['Age','Q3','Q4','Q5','Q6','Q7','Academic_pressure','Q9','Q10']]

df_clean.dropna(inplace=True)

df_clean.columns = [
    'Age','Sleep_hours','Study_hours','Screen_hours',
    'Sleep_quality','Physical_activity','Academic_pressure',
    'Stress','Family_support'
]

# -----------------------------
# 3. Create Target Variable
# -----------------------------
df_clean['Risk'] = np.where(
    (df_clean['Stress'] >= 0.5) | (df_clean['Academic_pressure'] >= 4),
    1,  # High risk
    0   # Low risk
)

# -----------------------------
# 4. Train/Test Split
# -----------------------------
X = df_clean.drop('Risk', axis=1)
y = df_clean['Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 5. Decision Tree Classifier
# -----------------------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# -----------------------------
# 6. Logistic Regression
# -----------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# -----------------------------
# 7. Evaluate Models
# -----------------------------
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n--- {model_name} Evaluation ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

evaluate_model(y_test, y_pred_dt, "Decision Tree")
evaluate_model(y_test, y_pred_lr, "Logistic Regression")

# -----------------------------
# 8. Visualizations
# -----------------------------
plt.figure(figsize=(20,10))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["Low Risk","High Risk"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
