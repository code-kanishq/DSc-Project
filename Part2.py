# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend to avoid Tkinter issues
import matplotlib.pyplot as plt
from scipy import stats

# Step 2: Load Data
# Load the datasets
red_wine = pd.read_csv('winequality-red.csv', delimiter=';')
white_wine = pd.read_csv('winequality-white.csv', delimiter=';')

# Show basic info about the data
print("Red Wine Dataset Info:")
print(red_wine.info())

print("\nWhite Wine Dataset Info:")
print(white_wine.info())

# Step 3: Descriptive Statistics
def describe_data(df):
    stats = {
        'mean': df.mean(),
        'median': df.median(),
        'variance': df.var(),
        'std_dev': df.std()
    }
    return pd.DataFrame(stats)

# Apply to both datasets
print("\nRed Wine Statistics:")
print(describe_data(red_wine))

print("\nWhite Wine Statistics:")
print(describe_data(white_wine))

# Step 4: Correlation Analysis
# Correlation matrix (Pearson by default)
def correlation_analysis(df):
    correlation_matrix = df.corr()
    return correlation_matrix

# Apply to both datasets
red_corr = correlation_analysis(red_wine)
white_corr = correlation_analysis(white_wine)

print("\nRed Wine Correlation Matrix:")
print(red_corr)

print("\nWhite Wine Correlation Matrix:")
print(white_corr)

# Step 5: Visualizations

# Histograms for Distribution
def plot_histograms(df, title, filename):
    df.hist(bins=20, figsize=(10, 8))
    plt.suptitle(title)
    plt.savefig(filename)
    plt.close()  # Close the plot to avoid memory issues

# Apply to both datasets
plot_histograms(red_wine, "Red Wine - Histograms", 'red_wine_histograms.png')
plot_histograms(white_wine, "White Wine - Histograms", 'white_wine_histograms.png')

# Boxplots for Detecting Outliers
def plot_boxplots(df, title, filename):
    plt.figure(figsize=(10, 8))
    sns.boxplot(data=df)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.savefig(filename)
    plt.close()

# Apply to both datasets
plot_boxplots(red_wine, "Red Wine - Boxplots", 'red_wine_boxplots.png')
plot_boxplots(white_wine, "White Wine - Boxplots", 'white_wine_boxplots.png')

# Pairplots to Observe Relationships Between Pairs of Features
def plot_pairplot(df, title, filename):
    sns.pairplot(df)
    plt.suptitle(title)
    plt.savefig(filename)
    plt.close()

# Apply to both datasets
plot_pairplot(red_wine, "Red Wine - Pairplot", 'red_wine_pairplot.png')
plot_pairplot(white_wine, "White Wine - Pairplot", 'white_wine_pairplot.png')

# Step 6: Validation with Statistical Tests

# T-tests for Comparisons (if applicable)
def t_test(df1, df2, feature):
    stat, p_value = stats.ttest_ind(df1[feature], df2[feature])
    return stat, p_value

# Example: Testing if the feature "fixed acidity" has a significant difference
stat, p_value = t_test(red_wine, white_wine, 'fixed acidity')
print(f"\nT-test for 'fixed acidity' between red and white wines: stat={stat}, p_value={p_value}")

# ANOVA (for comparing more than two groups, if applicable)
def anova_test(df1, df2, feature):
    f_stat, p_value = stats.f_oneway(df1[feature], df2[feature])
    return f_stat, p_value

# Example: Performing ANOVA on "citric acid"
f_stat, p_value = anova_test(red_wine, white_wine, 'citric acid')
print(f"\nANOVA for 'citric acid': f_stat={f_stat}, p_value={p_value}")
