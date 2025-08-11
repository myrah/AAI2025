import pandas as pd

# ---- Sample DataFrames ----

# First DataFrame: Employees
df_employees = pd.DataFrame({
    'EmployeeID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'DepartmentID': [101, 102, 101, 103]
})

# Second DataFrame: Departments
df_departments = pd.DataFrame({
    'DepartmentID': [101, 102, 104],
    'DepartmentName': ['Sales', 'HR', 'IT']
})

print(" Employees DataFrame:")
print(df_employees)

print("\n Departments DataFrame:")
print(df_departments)

# ---- 1. INNER JOIN (default) ----
inner_join = pd.merge(df_employees, df_departments, on='DepartmentID', how='inner')
print("\n INNER JOIN (only matching rows):")
print(inner_join)

# ---- 2. LEFT JOIN ----
left_join = pd.merge(df_employees, df_departments, on='DepartmentID', how='left')
print("\n LEFT JOIN (all employees, with department info if available):")
print(left_join)

# ---- 3. RIGHT JOIN ----
right_join = pd.merge(df_employees, df_departments, on='DepartmentID', how='right')
print("\n RIGHT JOIN (all departments, with employee info if available):")
print(right_join)

# ---- 4. OUTER JOIN ----
outer_join = pd.merge(df_employees, df_departments, on='DepartmentID', how='outer')
print("\n OUTER JOIN (all rows from both, matched when possible):")
print(outer_join)
