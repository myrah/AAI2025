import pandas as pd
import numpy as np

# Sample raw dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', None, 'Ethan'],
    'Age': [23, np.nan, 22, 25, 24],
    'Score': ['85', '90', None, '88', '92'],
    'City': ['New York', 'Los Angeles', None, 'Chicago', 'Houston']
}

df = pd.DataFrame(data)

print(" Raw Data:")
print(df)

# ---- 1. Handling Missing Values ----

# Drop rows with any missing values
df_dropped = df.dropna()
print("\n Dropped Rows with Any Missing Values:")
print(df_dropped)

# Fill missing values with a default value
df_filled = df.fillna({'Name': 'Unknown', 'Age': df['Age'].mean(), 'Score': '0', 'City': 'Unknown'})
print("\n Filled Missing Values:")
print(df_filled)

# ---- 2. Data Type Conversion ----

# Convert 'Score' column from string to integer
df_filled['Score'] = df_filled['Score'].astype(int)

print("\n After Converting 'Score' to Integer:")
print(df_filled.dtypes)

# ---- 3. Data Filtering ----

# Filter rows where Age is greater than 23
filtered_df = df_filled[df_filled['Age'] > 23]
print("\n Filtered Rows (Age > 23):")
print(filtered_df)

# ---- 4. Data Transformation ----

# Add a new column with upper-case city names
df_filled['City_Upper'] = df_filled['City'].str.upper()

# Add a calculated column: Age * Score
df_filled['Performance'] = df_filled['Age'] * df_filled['Score']

print("\n Transformed Data:")
print(df_filled)
