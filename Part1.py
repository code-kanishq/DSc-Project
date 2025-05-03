# Upload Kaggle JSON and download dataset
from google.colab import files
files.upload()  # Upload kaggle.json here

import os
import zipfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Setup Kaggle API and download dataset
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d rajyellow46/wine-quality
!unzip -o wine-quality.zip

# Load dataset
df = pd.read_csv('winequalityN.csv')

# Basic info
print("Distinct wine types in the dataset:")
print(df['type'].unique())

print("Dataset Shape (rows, columns):")
print(df.shape)

print("Column names:")
print(df.columns)

print("Data types")
print(df.dtypes)

print("Summary statistics:")
print(df.describe())

print("Missing values per column:")
print(df.isnull().sum())

# Plot distribution of wine types
sns.countplot(x='type', data=df, palette='Set2')
plt.title('Count of Wine Types')
plt.xlabel('Wine Type')
plt.ylabel('Count')
plt.show()

if 'type' in df.columns:
    print("\nUnique wine types:", df['type'].unique())
    print("Counts of each type:\n", df['type'].value_counts())

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Impute missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Remove outliers
z_scores = np.abs(zscore(df[numeric_cols]))
outlier_count = (z_scores > 3).sum(axis=1)
df = df[outlier_count <= 1]
print("After removing outliers:", df.shape)

# Feature engineering
def categorize_quality(score):
    if score <= 4:
        return 'Low'
    elif 5 <= score <= 6:
        return 'Medium'
    else:
        return 'High'
df['Quality Label'] = df['quality'].apply(categorize_quality)
print(df['Quality Label'].value_counts())

sns.countplot(x='quality', data=df, palette='muted', hue='type')
plt.title('Wine Quality Distribution by Type')
plt.xlabel('Quality Score')
plt.ylabel('Count')
plt.legend(title='Wine Type')
plt.show()

# Feature distribution
plt.figure(figsize=(8, 6))
sns.boxplot(x='type', y='alcohol', data=df, palette='pastel')
plt.title('Alcohol Content by Wine Type')
plt.xlabel('Wine Type')
plt.ylabel('Alcohol Content')
plt.show()

sns.histplot(df['alcohol'], kde=True, bins=30)
plt.title('Distribution of Alcohol Content')
plt.show()

# Grouped stats
print(df.groupby('type').mean(numeric_only=True).T)

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

sns.pairplot(df, hue='type', vars=['alcohol', 'sulphates', 'pH', 'density'])
plt.suptitle('Pairplot of Selected Features by Wine Type', y=1.02)
plt.show()

# Feature scaling
X = df.drop(columns=['quality', 'Quality Label', 'type'])
y = df['Quality Label']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Target distribution
sns.countplot(x='quality', data=df, palette='viridis')
plt.title('Wine Quality Distribution')
plt.show()

# PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Explained Variance')
plt.grid(True)
plt.show()

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("Explained variance ratio:\n", explained_variance)
print("PCA Components:\n", pca.components_)
