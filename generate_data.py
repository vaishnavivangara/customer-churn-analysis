import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

n_customers = 1000

def generate_customer(i):
    tenure = np.random.randint(1, 72)
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], p=[0.55, 0.25, 0.20])
    monthly_charges = round(np.random.uniform(20, 120), 2)
    internet = np.random.choice(['DSL', 'Fiber optic', 'No'], p=[0.35, 0.45, 0.20])
    tech_support = np.random.choice(['Yes', 'No'], p=[0.4, 0.6])
    online_security = np.random.choice(['Yes', 'No'], p=[0.4, 0.6])
    paperless = np.random.choice(['Yes', 'No'], p=[0.6, 0.4])
    payment = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
    senior = np.random.choice([0, 1], p=[0.85, 0.15])
    partner = np.random.choice(['Yes', 'No'])
    dependents = np.random.choice(['Yes', 'No'], p=[0.3, 0.7])
    gender = np.random.choice(['Male', 'Female'])

    churn_score = 0
    if contract == 'Month-to-month':
        churn_score += 0.35
    if tenure < 12:
        churn_score += 0.25
    if tech_support == 'No':
        churn_score += 0.15
    if online_security == 'No':
        churn_score += 0.10
    if internet == 'Fiber optic':
        churn_score += 0.10
    if monthly_charges > 80:
        churn_score += 0.10
    if payment == 'Electronic check':
        churn_score += 0.05

    churn_prob = min(churn_score, 0.9)
    churn = 'Yes' if np.random.random() < churn_prob else 'No'

    total_charges = round(monthly_charges * tenure, 2)

    return {
        'customer_id': f'C{i:04d}',
        'gender': gender,
        'senior_citizen': senior,
        'partner': partner,
        'dependents': dependents,
        'tenure_months': tenure,
        'contract_type': contract,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'payment_method': payment,
        'internet_service': internet,
        'tech_support': tech_support,
        'online_security': online_security,
        'paperless_billing': paperless,
        'churn': churn
    }

data = [generate_customer(i) for i in range(1, n_customers + 1)]
df = pd.DataFrame(data)

df.to_csv('synthetic_churn_data.csv', index=False)
print(f"Generated {len(df)} customers")
print(f"Churn rate: {(df['churn'] == 'Yes').mean():.1%}")
print(df.head())