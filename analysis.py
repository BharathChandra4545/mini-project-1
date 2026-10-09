import pandas as pd
import numpy as np

df = pd.read_csv("data/student-mat.csv", sep=";")

print("FIRST FIVE STUDENTS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

# STEP 4: DATA CLEANING

print("\n--- DATA CLEANING ---")

# Check duplicate records
print("Duplicate rows:", df.duplicated().sum())

# Remove duplicate records
df = df.drop_duplicates().copy()

# Check missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Check dataset after cleaning
print("\nCleaned dataset shape:", df.shape)

# Save cleaned data
df.to_csv("data/cleaned_students.csv", index=False)

print("\nCleaned dataset saved successfully!")

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Distribution of final grades
plt.figure(figsize=(8, 5))
sns.histplot(df["G3"], bins=11, kde=True)
plt.title("Distribution of Final Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# 2. Study time vs final grade
plt.figure(figsize=(8, 5))
sns.boxplot(x="studytime", y="G3", data=df)
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time Category")
plt.ylabel("Final Grade")
plt.tight_layout()
plt.show()

# 3. Earlier grade vs final grade
plt.figure(figsize=(8, 5))
sns.scatterplot(x="G2", y="G3", data=df)
plt.title("G2 Grade vs Final Grade")
plt.tight_layout()
plt.show()

# 4. Correlation heatmap
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Between Numerical Variables")
plt.tight_layout()
plt.show()

print("EDA completed successfully!")

# STEP 6: DEFINE LEARNING RISK

df["learning_risk"] = (df["G3"] < 10).astype(int)

print("\n--- LEARNING RISK ANALYSIS ---")

print("Risk distribution:")
print(df["learning_risk"].value_counts())

print("\nRisk percentage:")
print(
    df["learning_risk"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

df.to_csv("data/cleaned_students.csv", index=False)

print("\nLearning-risk labels created successfully!")
