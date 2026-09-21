import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

conn = sqlite3.connect('churn_project.db')
df = pd.read_sql_query("SELECT * FROM customers", conn)
conn.close()

print("Data loaded:")
print(df.shape)
print(df.head())

df = df.drop(columns=['customer_id'])

categorical_columns = ['gender', 'partner', 'dependents', 'contract_type', 
                        'payment_method', 'internet_service', 'tech_support', 
                        'online_security', 'paperless_billing']

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("After encoding:")
print(df.head())

df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})

X = df.drop(columns=['churn'])
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} customers")
print(f"Test set: {X_test.shape[0]} customers")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model trained successfully")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.1%}")

print("\nDetailed Performance Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'weight': model.coef_[0]
}).sort_values('weight', key=abs, ascending=False)

print("\nWhat matters most for predicting churn:")
print(feature_importance)