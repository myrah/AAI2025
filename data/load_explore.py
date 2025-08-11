import pandas as pd
from sqlalchemy import create_engine  # Only for database example

# ---- 1. Load data from CSV ----
csv_data = pd.read_csv('sample_data.csv')
print(" CSV Data Loaded:")
print(csv_data.head())

# ---- 2. Load data from Excel ----
excel_data = pd.read_excel('sample_data.xlsx', engine='openpyxl')
print("\n Excel Data Loaded:")
print(excel_data.head())

# ---- 3. Load data from a SQL database (Optional demo) ----
# Example using SQLite for demonstration
# Create a connection (replace with your DB URI)
engine = create_engine('sqlite:///example.db')  # Assumes example.db exists
# sql_data = pd.read_sql('SELECT * FROM your_table', engine)
# print("\n SQL Data Loaded:")
# print(sql_data.head())

# ---- Basic Data Exploration ----
print("\n First 5 Rows:")
print(csv_data.head())

print("\n Last 5 Rows:")
print(csv_data.tail())

print("\n Info:")
csv_data.info()

print("\n Summary Statistics:")
print(csv_data.describe())
