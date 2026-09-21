import pandas as pd
import sqlite3
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

conn = sqlite3.connect('churn_project.db')
df = pd.read_sql_query("SELECT * FROM customers", conn)
conn.close()

customer_ids = df['customer_id']
monthly_charges_original = df['monthly_charges']

df_model = df.drop(columns=['customer_id'])

categorical_columns = ['gender', 'partner', 'dependents', 'contract_type', 
                        'payment_method', 'internet_service', 'tech_support', 
                        'online_security', 'paperless_billing']

for col in categorical_columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])

df_model['churn'] = df_model['churn'].map({'Yes': 1, 'No': 0})

X = df_model.drop(columns=['churn'])
y = df_model['churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

churn_probabilities = model.predict_proba(X_scaled)[:, 1]

results = pd.DataFrame({
    'customer_id': customer_ids,
    'monthly_charges': monthly_charges_original,
    'churn_probability': churn_probabilities,
    'actual_churn': df['churn']
})

results['risk_level'] = pd.cut(results['churn_probability'], 
                                  bins=[0, 0.4, 0.6, 1.0], 
                                  labels=['Low Risk', 'Medium Risk', 'High Risk'])

median_charge = results['monthly_charges'].median()
results['value_level'] = results['monthly_charges'].apply(
    lambda x: 'High Value' if x >= median_charge else 'Low Value'
)

results['segment'] = results['risk_level'].astype(str) + ' / ' + results['value_level']

segment_summary = results.groupby('segment').agg(
    customer_count=('customer_id', 'count'),
    avg_churn_probability=('churn_probability', 'mean'),
    avg_monthly_charges=('monthly_charges', 'mean')
).sort_values('avg_churn_probability', ascending=False)

print("Segment Summary:")
print(segment_summary)

results.to_csv('customer_segments.csv', index=False)
print("\nSaved full results to customer_segments.csv")