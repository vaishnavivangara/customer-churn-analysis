import pandas as pd
import sqlite3

# Load the CSV
df = pd.read_csv('synthetic_churn_data.csv')

# Connect to your existing SQLite database
conn = sqlite3.connect('churn_project.db')

# Replace the customers table with this new data
df.to_sql('customers', conn, if_exists='replace', index=False)

conn.commit()
conn.close()

print(f"Loaded {len(df)} customers into churn_project.db")