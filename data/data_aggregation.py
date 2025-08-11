import pandas as pd

# Sample dataset
data = {
    'Department': ['Sales', 'Sales', 'HR', 'HR', 'IT', 'IT', 'IT'],
    'Employee': ['Alice', 'Bob', 'Charlie', 'Diana', 'Ethan', 'Fiona', 'George'],
    'Salary': [60000, 65000, 58000, 62000, 70000, 72000, 71000],
    'Bonus': [5000, 6000, 4000, 4500, 5500, 6000, 5800]
}

df = pd.DataFrame(data)

print(" Original Data:")
print(df)

# ---- 1. Grouping and Aggregation ----

# Group by Department and calculate mean Salary and Bonus
grouped = df.groupby('Department')[['Salary', 'Bonus']].mean()

print("\n Average Salary and Bonus by Department:")
print(grouped)

# You can also use multiple aggregation functions
grouped_multi = df.groupby('Department').agg({
    'Salary': ['mean', 'max'],
    'Bonus': 'sum'
})

print("\n Grouped with Multiple Aggregations:")
print(grouped_multi)

# ---- 2. Pivot Tables ----

# Create a pivot table to show average Salary by Department
pivot = pd.pivot_table(df, values='Salary', index='Department', aggfunc='mean')

print("\n Pivot Table - Average Salary by Department:")
print(pivot)

# Pivot Table with multiple values and aggregation
pivot_multi = pd.pivot_table(df, values=['Salary', 'Bonus'], index='Department', aggfunc='sum')

print("\n Pivot Table - Total Salary and Bonus by Department:")
print(pivot_multi)
