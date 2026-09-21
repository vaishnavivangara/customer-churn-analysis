import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

conn = sqlite3.connect('churn_project.db')
df = pd.read_sql_query("SELECT * FROM customers", conn)
conn.close()

# Chart 1: Churn rate by contract type
contract_churn = df.groupby('contract_type')['churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
bars = plt.bar(contract_churn.index, contract_churn.values, color=['#d62728', '#ff7f0e', '#2ca02c'])
plt.title('Churn Rate by Contract Type', fontsize=14, fontweight='bold')
plt.ylabel('Churn Rate (%)')
plt.xlabel('Contract Type')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}%', ha='center')
plt.tight_layout()
plt.savefig('chart1_churn_by_contract.png', dpi=150)
plt.close()
print("Saved chart1_churn_by_contract.png")

# Chart 2: Churn rate by tenure group
df['tenure_group'] = pd.cut(df['tenure_months'], bins=[0, 12, 24, 48, 100],
                              labels=['0-12 months', '12-24 months', '24-48 months', '48+ months'])
tenure_churn = df.groupby('tenure_group')['churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
)

plt.figure(figsize=(8, 5))
bars = plt.bar(tenure_churn.index.astype(str), tenure_churn.values, color='#1f77b4')
plt.title('Churn Rate by Customer Tenure', fontsize=14, fontweight='bold')
plt.ylabel('Churn Rate (%)')
plt.xlabel('Tenure Group')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}%', ha='center')
plt.tight_layout()
plt.savefig('chart2_churn_by_tenure.png', dpi=150)
plt.close()
print("Saved chart2_churn_by_tenure.png")

# Chart 3: Churn rate by support services
support_churn = df.groupby(['tech_support', 'online_security'])['churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index()
support_churn['label'] = 'Tech: ' + support_churn['tech_support'] + ', Security: ' + support_churn['online_security']
support_churn = support_churn.sort_values('churn', ascending=False)

plt.figure(figsize=(9, 5))
bars = plt.barh(support_churn['label'], support_churn['churn'], color='#9467bd')
plt.title('Churn Rate by Support Services', fontsize=14, fontweight='bold')
plt.xlabel('Churn Rate (%)')
for bar in bars:
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center')
plt.tight_layout()
plt.savefig('chart3_churn_by_support.png', dpi=150)
plt.close()
print("Saved chart3_churn_by_support.png")

print("\nAll charts created successfully")
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

df_model = df.drop(columns=['customer_id', 'tenure_group'])

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

importance_df = pd.DataFrame({
    'feature': X.columns,
    'weight': model.coef_[0]
}).sort_values('weight', key=abs, ascending=True)

colors = ['#d62728' if w > 0 else '#1f77b4' for w in importance_df['weight']]

plt.figure(figsize=(9, 6))
plt.barh(importance_df['feature'], importance_df['weight'], color=colors)
plt.title('Feature Importance (Logistic Regression Weights)', fontsize=14, fontweight='bold')
plt.xlabel('Weight (negative = reduces churn risk, positive = increases churn risk)')
plt.axvline(x=0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('chart4_feature_importance.png', dpi=150)
plt.close()
print("Saved chart4_feature_importance.png")