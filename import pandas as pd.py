import pandas as pd

# 1. Read the CSV file
df = pd.read_csv("students.csv")

print("\n--- ORIGINAL DATASET ---")
print(df)

# 2. Display basic information
print("\n--- DATASET INFORMATION ---")
print(df.info())

# 3. Check for missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# 4. Check for duplicate rows
print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

# 5. Remove duplicate rows
df = df.drop_duplicates()

# 6. Calculate average marks
average_marks = df["Marks"].mean()
print("\nAverage Marks:", average_marks)

# 7. Find the student with the highest marks
top_student = df.loc[df["Marks"].idxmax()]
print("\n--- TOPPER ---")
print(top_student)

# 8. Find students who scored above 80
high_scorers = df[df["Marks"] > 80]
print("\n--- STUDENTS SCORING ABOVE 80 ---")
print(high_scorers[["Name", "Marks"]])

# 9. Calculate average marks department-wise
department_average = df.groupby("Department")["Marks"].mean()
print("\n--- DEPARTMENT-WISE AVERAGE ---")
print(department_average)

# 10. Add a new column based on marks
df["Result"] = df["Marks"].apply(
    lambda marks: "Pass" if marks >= 40 else "Fail"
)

# 11. Sort students by marks in descending order
df = df.sort_values(by="Marks", ascending=False)

# 12. Save the cleaned and analysed data
df.to_csv("analysed_students.csv", index=False)

print("\n--- FINAL DATASET ---")
print(df)

print("\nAnalysis completed successfully!")