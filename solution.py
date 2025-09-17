# Import necessary libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pandas.core.dtypes.common import is_numeric_dtype
from scipy.stats import chi2_contingency

# Load the dataset
df = pd.read_csv('diabetes_prediction_dataset.csv')

# Display the first few rows of the dataset
df.head()

df.isnull().sum()

df.columns

df.describe()

# Handle any missing values if necessary (e.g., drop or impute)
df.dropna(inplace=True)



for column in df.columns:
    unik = df[column].unique()
    print(f"Unique before preprocessing '{column}': {unik}\n")

label=LabelEncoder()


for column in df.columns:
    if is_numeric_dtype(df[column]):
        continue
    else:
        df[column]=label.fit_transform(df[column])

for column in df.columns:
    unik = df[column].unique()
    print(f"Unique after preprocessing '{column}': {unik}\n")


print(df.head())

print(df['gender'].value_counts())
print(df['smoking_history'].value_counts())


"""
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
df['smoking_history'] = df['smoking_history'].map({'not current': 0, 'former': 1, 'No Info': 2, 'current': 3, 'never': 4, 'ever': 5})

df = df.apply(pd.to_numeric)

print(df.head())
"""

sns.countplot(x='gender',hue='diabetes',data=df)
plt.xlabel('Gender (0 = Female, 1 = Male)')
plt.legend(["No diabetes", "Diabetes"])
plt.show()

sns.countplot(x='hypertension',hue='diabetes',data=df)
plt.xlabel('Hypertension (0 = No, 1 = Yes)')
plt.legend(["No diabetes", "Diabetes"])
plt.show()

sns.countplot(x='heart_disease',hue='diabetes',data=df)
plt.xlabel('Heart disease (0 = No, 1 = Yes)')
plt.legend(["No diabetes", "Diabetes"])
plt.show()

plt.figure(figsize=(13,4))
sns.countplot(x='smoking_history',hue='diabetes',data=df)
plt.legend(["No diabetes", "Diabetes"])
plt.show()

plt.figure(figsize=(13,4))
sns.countplot(x='HbA1c_level',hue='diabetes',data=df)
plt.legend(["No diabetes", "Diabetes"])
plt.show()

plt.figure(figsize=(13,4))
sns.countplot(x='blood_glucose_level',hue='diabetes',data=df)
plt.legend(["No diabetes", "Diabetes"])
plt.show()



contingency_table_gender = pd.crosstab(df['gender'], df['diabetes'])
print(contingency_table_gender)


# Hodnoty z tabulky
A = contingency_table_gender.loc[1, 1]
B = contingency_table_gender.loc[1, 0]
C = contingency_table_gender.loc[0, 1]
D = contingency_table_gender.loc[0, 0]
E = contingency_table_gender.loc[2, 1]
F = contingency_table_gender.loc[2, 0]

# Výpočet poměru šancí
odds_ratio = (A / B) / (C / D)
print(f"Poměr šancí (Odds Ratio) pro gender: {odds_ratio}")

# Závěr
if odds_ratio > 1:
    print("Pravděpodobnost diabetu je vyšší u males.")
elif odds_ratio < 1:
    print("Pravděpodobnost diabetu je nižší u males.")
else:
    print("Pravděpodobnost diabetu je stejná u males a females")


# Chi-Square test for gender
chi2, p, _, _ = chi2_contingency(contingency_table_gender)
print(f"Chi-Square Test for Gender: chi2 = {chi2}, p = {p}")




contingency_table_hypertension = pd.crosstab(df['hypertension'], df['diabetes'])
print(contingency_table_hypertension)

# Hodnoty z tabulky
A = contingency_table_hypertension.loc[1, 1]
B = contingency_table_hypertension.loc[1, 0]
C = contingency_table_hypertension.loc[0, 1]
D = contingency_table_hypertension.loc[0, 0]

# Výpočet poměru šancí
odds_ratio = (A / B) / (C / D)
print(f"Poměr šancí (Odds Ratio) pro hypertenzi: {odds_ratio}")

# Závěr
if odds_ratio > 1:
    print("Pravděpodobnost diabetu je vyšší u lidí s hypertenzí.")
elif odds_ratio < 1:
    print("Pravděpodobnost diabetu je nižší u lidí s hypertenzí.")
else:
    print("Pravděpodobnost diabetu je stejná u lidí s hypertenzí i bez ní.")

# Chi-Square test for hypertension
chi2, p, _, _ = chi2_contingency(contingency_table_hypertension)
print(f"Chi-Square Test for Hypertension: chi2 = {chi2}, p = {p}")






contingency_table_heart_disease = pd.crosstab(df['heart_disease'], df['diabetes'])
print(contingency_table_heart_disease)

# Hodnoty z tabulky
A = contingency_table_heart_disease.loc[1, 1]
B = contingency_table_heart_disease.loc[1, 0]
C = contingency_table_heart_disease.loc[0, 1]
D = contingency_table_heart_disease.loc[0, 0]

# Výpočet poměru šancí
odds_ratio = (A / B) / (C / D)
print(f"Poměr šancí (Odds Ratio) pro heart diseases: {odds_ratio}")

# Závěr
if odds_ratio > 1:
    print("Pravděpodobnost diabetu je vyšší u lidí s heart diseases.")
elif odds_ratio < 1:
    print("Pravděpodobnost diabetu je nižší u lidí s heart diseases.")
else:
    print("Pravděpodobnost diabetu je stejná u lidí s heart diseases i bez nich.")


# Chi-Square test for heart disease
chi2, p, _, _ = chi2_contingency(contingency_table_heart_disease)
print(f"Chi-Square Test for Heart Disease: chi2 = {chi2}, p = {p}")

"""
['never' 'No Info' 'current' 'former' 'ever' 'not current']
 [4          0         1        3        2        5]
 """