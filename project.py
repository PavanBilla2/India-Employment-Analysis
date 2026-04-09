# ============================================
# INDIA EMPLOYMENT ANALYSIS PROJECT
# ============================================

# STEP 1: IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Set style
sns.set()

# ============================================
# STEP 2: LOAD DATASET
# ============================================

df = pd.read_excel("India_employment.xlsx")

print("\nFirst 5 Rows:\n")
print(df.head())

print("\nDataset Info:\n")
print(df.info())

# ============================================
# STEP 3: DATA CLEANING
# ============================================

print("\nMissing Values:\n")
print(df.isnull().sum())

df = df.rename(columns={
    'obs_value': 'employment_rate',
    'time': 'year',
    'sex.label': 'gender'
})

#to add mean values to the missing values
df['employment_rate'] = df['employment_rate'].fillna(df['employment_rate'].mean())

print(df[['employment_rate', 'year', 'gender']].isnull().sum())

df['gender'] = df['gender'].fillna('Unknown')

# Drop unnecessary column (if exists)
if 'obs_status.label' in df.columns:
    df = df.drop(columns=['obs_status.label'])

# Rename columns
df.columns = ['country', 'source', 'indicator', 'gender', 'category', 'year', 'employment_rate']

# Remove duplicates
df = df.drop_duplicates()

# Convert data types
df['year'] = df['year'].astype(int)
df['employment_rate'] = df['employment_rate'].astype(float)

print("\nCleaned Data:\n")
print(df.head())

# ============================================
# STEP 4: NUMPY OPERATIONS
# ============================================

data = df['employment_rate'].to_numpy()

print("\nNUMPY STATISTICS:")
print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Standard Deviation:", np.std(data))

# ============================================
# STEP 5: DATA VISUALIZATION
# ============================================

# Set style
sns.set(style="whitegrid")

# 1. Line Plot (Trend)
plt.figure()
plt.plot(df['year'], df['employment_rate'], color='blue', marker='o')
plt.xlabel("Year")
plt.ylabel("Employment Rate")
plt.title("Employment Trend in India", fontsize=14, color='darkred')
plt.grid(True)
plt.show()

# 2. Gender Comparison
plt.figure()
sns.barplot(x='gender', y='employment_rate', data=df, palette='Set2')
plt.title("Employment Rate by Gender", fontsize=14, color='purple')
plt.show()

# 3. Distribution Plot
plt.figure()
sns.histplot(df['employment_rate'], kde=True, color='green')
plt.title("Employment Distribution", fontsize=14, color='darkgreen')
plt.show()

# 4. Boxplot (Outliers)
plt.figure()
# sns.boxplot(x=df['employment_rate'], color='orange')
sns.boxplot(x=df['employment_rate'], showfliers=True)
plt.title("Outlier Detection", fontsize=14, color='brown')
plt.show()

# 5. Heatmap
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap", fontsize=14, color='navy')
plt.show()

# ============================================
# STEP 6: EDA (EXPLORATORY DATA ANALYSIS)
# ============================================

print("\nSummary Statistics:\n")
print(df.describe())

print("\nCorrelation:\n")
print(df.corr(numeric_only=True))

print("\nCovariance:\n")
print(df.cov(numeric_only=True))

# Group Analysis
print("\nAverage Employment by Gender:\n")
print(df.groupby('gender')['employment_rate'].mean())

print("\nEmployment Trend by Year:\n")
print(df.groupby('year')['employment_rate'].mean())

# ============================================
# STEP 7: ADVANCED VISUALS
# ============================================

# Seaborn Lineplot
plt.figure()
sns.lineplot(x='year', y='employment_rate', data=df)
plt.title("Employment Trend (Seaborn)")
plt.show()

# Countplot
plt.figure()
sns.countplot(x='gender', data=df)
plt.title("Gender Count")
plt.show()

# Pairplot (optional, may take time)
sns.pairplot(df)
plt.show()

# ============================================
# STEP 8: INSIGHTS (PRINT)
# ============================================

print("\nINSIGHTS:")
print("- Employment rate varies across years.")
print("- Gender-based differences exist in employment.")
print("- Some outliers are present in the dataset.")
print("- Trends show fluctuations over time.")


# ============================================
# STEP 9: MACHINE LEARNING MODEL
# ============================================

# # Copy dataset
# df_ml = df.copy()

# # ============================================
# # STEP 2: CLEAN DATA
# # ============================================

# # Keep only required columns
# df_ml = df_ml[['year', 'gender', 'employment_rate']]

# # Clean gender text
# df_ml['gender'] = df_ml['gender'].astype(str).str.lower()

# # Keep only male & female
# df_ml = df_ml[df_ml['gender'].isin(['male', 'female'])]

# # Encode gender
# df_ml['gender'] = df_ml['gender'].map({'male': 0, 'female': 1})

# # Remove missing values
# df_ml = df_ml.dropna()

# # ============================================
# # STEP 3: REMOVE OUTLIERS
# # ============================================

# upper_limit = df_ml['employment_rate'].quantile(0.95)
# df_ml = df_ml[df_ml['employment_rate'] <= upper_limit]

# # ============================================
# # STEP 4: FEATURES & TARGET
# # ============================================

# X = df_ml[['year', 'gender']]
# y = df_ml['employment_rate']

# # ============================================
# # STEP 5: SCALING
# # ============================================

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # ============================================
# # STEP 6: POLYNOMIAL FEATURES
# # ============================================

# poly = PolynomialFeatures(degree=2)
# X_poly = poly.fit_transform(X_scaled)

# # ============================================
# # STEP 7: TRAIN TEST SPLIT
# # ============================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X_poly, y, test_size=0.2, random_state=42
# )

# # ============================================
# # STEP 8: MODEL TRAINING
# # ============================================

# model = LinearRegression()
# model.fit(X_train, y_train)

# # ============================================
# # STEP 9: PREDICTION
# # ============================================

# y_pred = model.predict(X_test)

# # ============================================
# # STEP 10: EVALUATION
# # ============================================

# print("\nOPTIMIZED LINEAR REGRESSION PERFORMANCE:")
# print("MSE:", mean_squared_error(y_test, y_pred))
# print("R2 Score:", r2_score(y_test, y_pred))

# ============================================
# STEP 1: COPY DATA
# ============================================

df_ml = df.copy()

# ============================================
# STEP 2: SELECT IMPORTANT FEATURES
# ============================================

df_ml = df_ml[['year', 'gender', 'employment_rate']]

# Clean gender
df_ml['gender'] = df_ml['gender'].astype(str).str.lower()

# Keep only male & female
df_ml = df_ml[df_ml['gender'].isin(['male', 'female'])]

# Encode gender
df_ml['gender'] = df_ml['gender'].map({'male': 0, 'female': 1})

# ============================================
# STEP 3: LIGHT CLEANING (NOT TOO STRICT)
# ============================================

# Remove only extreme values
upper = df_ml['employment_rate'].quantile(0.98)
lower = df_ml['employment_rate'].quantile(0.02)

df_ml = df_ml[(df_ml['employment_rate'] <= upper) & 
              (df_ml['employment_rate'] >= lower)]

# Drop missing
df_ml = df_ml.dropna()

# ============================================
# STEP 4: FEATURES & TARGET
# ============================================

X = df_ml[['year', 'gender']]
y = df_ml['employment_rate']

# ============================================
# STEP 5: TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# STEP 6: MODEL
# ============================================

model = LinearRegression()
model.fit(X_train, y_train)

# ============================================
# STEP 7: PREDICTION
# ============================================

y_pred = model.predict(X_test)

# ============================================
# STEP 8: EVALUATION
# ============================================

print("\nFINAL MODEL PERFORMANCE:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# ============================================
# VISUALIZATION 
# ============================================

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Employment Rate")
plt.grid(True)
plt.show()

# ============================================
# SAMPLE PREDICTION
# ============================================

# Save training columns
train_columns = X.columns

# Create sample with same structure
# Step 1: Create raw sample
sample = pd.DataFrame([[2025, 1]], columns=['year', 'gender'])

# Step 2: Apply scaling
sample_scaled = scaler.transform(sample)

# Step 3: Apply polynomial transformation
sample_poly = poly.transform(sample_scaled)

# Step 4: Predict
prediction = model.predict(sample_poly)

print("\nPredicted Employment Rate:", prediction[0])

#Another machine learning model

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Convert to classification problem
df_class = df.copy()

# Create binary target
df_class['high_employment'] = (df_class['employment_rate'] > df_class['employment_rate'].mean()).astype(int)

X = df_class[['year']]
y = df_class['high_employment']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("\nClassification Accuracy:", model.score(X_test, y_test))