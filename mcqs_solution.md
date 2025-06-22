# Exploratory Data Analysis (EDA) Quiz Study Guide

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quiz Questions with Solutions](#quiz-questions-with-solutions)
3. [Topic-wise Summary](#topic-wise-summary)
4. [Important Concepts](#important-concepts)
5. [Python/Pandas Quick Reference](#pythonpandas-quick-reference)
6. [Study Tips](#study-tips)

---

## Introduction

This comprehensive guide covers 90 questions on Exploratory Data Analysis (EDA), organized by key concepts and topics. Each question includes multiple choice options, the correct answer, and a brief explanation of the underlying concept.

**Note:** Question 91 is a prerequisite check for "Python ka Chilla" course completion.

---

## Quiz Questions with Solutions

### Data Visualization (Questions 1-3, 12, 23, 25, 31-33, 36, 39, 44-45, 57, 63, 65, 79, 83, 85-86)

#### Question 1: Which graph is used to display the relationship between two categorical variables?

- **a)** Scatter plot
- **b)** Histogram
- **c)** Heatmap/Contingency table ✓
- **d)** Box plot

**Concept:** For two categorical variables, we use cross-tabulation visualized as heatmaps or stacked bar charts.

---

#### Question 2: In the Titanic dataset how would you find the number of passengers who survived?

- **a)** `df['Survived'].sum()` ✓
- **b)** `df['Survived'].count()`
- **c)** `df['Survived'].mean()`
- **d)** `df['Survived'].std()`

**Concept:** If 'Survived' is binary (0/1), summing gives the count of survivors.

---

#### Question 3: Which plot is used to visualize the distribution of a categorical variable?

- **a)** Histogram
- **b)** Bar plot/Count plot ✓
- **c)** Scatter plot
- **d)** Line plot

**Concept:** Bar plots show frequency/count for each category.

---

### Statistical Measures (Questions 4, 16-17, 20, 34-35, 41, 61, 80, 82, 87)

#### Question 4: What is the significance of the median in a dataset?

- **a)** It's always equal to the mean
- **b)** It's the middle value and robust to outliers ✓
- **c)** It's the most frequent value
- **d)** It measures spread

**Concept:** Median is the 50th percentile, less affected by extreme values than mean.

---

#### Question 16: What does the term 'Kurtosis' signify in a dataset?

- **a)** Central tendency
- **b)** Spread of data
- **c)** Tailedness/peakedness of distribution ✓
- **d)** Skewness direction

**Concept:** Kurtosis measures whether data has heavy or light tails compared to normal distribution.

---

#### Question 17: What is a Quantile in a dataset?

- **a)** The mean value
- **b)** Values that divide data into equal-sized groups ✓
- **c)** The most frequent value
- **d)** The range of data

**Concept:** Quantiles are cut points (e.g., quartiles divide into 4 parts).

---

### Data Analysis Types

#### Question 5: What is the main objective of clustering in EDA?

- **a)** To predict future values
- **b)** To group similar data points together ✓
- **c)** To reduce dimensions
- **d)** To remove outliers

**Concept:** Clustering identifies natural groupings in data.

---

#### Question 6: What does the term 'Bivariate Analysis' refer to in EDA?

- **a)** Analysis of one variable
- **b)** Analysis of relationship between two variables ✓
- **c)** Analysis of multiple variables
- **d)** Time-based analysis

**Concept:** Bivariate = two variables analyzed together.

---

#### Question 10: What does the term 'Multivariate Analysis' refer to in EDA?

- **a)** Analysis of one variable
- **b)** Analysis of two variables
- **c)** Analysis of multiple variables simultaneously ✓
- **d)** Time series analysis

**Concept:** Multivariate = analyzing relationships among 3+ variables.

---

### Data Preprocessing

#### Question 7: Which method is commonly used to handle missing categorical data?

- **a)** Mean imputation
- **b)** Mode imputation or creating 'Missing' category ✓
- **c)** Median imputation
- **d)** Linear interpolation

**Concept:** For categorical data, use mode or treat missing as a separate category.

---

#### Question 9: Why might you perform a log transformation on a dataset?

- **a)** To remove outliers
- **b)** To handle skewed distributions and make them more normal ✓
- **c)** To increase values
- **d)** To convert to integers

**Concept:** Log transformation reduces right skewness.

---

#### Question 18: What is the main objective of normalization in data processing?

- **a)** To remove outliers
- **b)** To scale features to a common range ✓
- **c)** To increase values
- **d)** To convert to integers

**Concept:** Normalization brings different features to comparable scales.

---

### Statistical Testing

#### Question 8: What is the null hypothesis in hypothesis testing?

- **a)** The hypothesis we want to prove
- **b)** The assumption of no effect or no difference ✓
- **c)** The alternative explanation
- **d)** The rejected hypothesis

**Concept:** H₀ assumes status quo or no significant difference.

---

#### Question 26: What is the primary goal of the Chi-square test in EDA?

- **a)** To compare means
- **b)** To test independence between categorical variables ✓
- **c)** To find correlation
- **d)** To detect outliers

**Concept:** Chi-square tests if two categorical variables are related.

---

#### Question 40: What does the p-value signify in hypothesis testing?

- **a)** Population size
- **b)** Probability of observing results if null hypothesis is true ✓
- **c)** Power of test
- **d)** Prediction accuracy

**Concept:** Low p-value suggests rejecting null hypothesis.

---

### Time Series Analysis

#### Question 11: What is the primary goal of forecasting in time series analysis?

- **a)** To understand past patterns
- **b)** To predict future values based on historical data ✓
- **c)** To remove seasonality
- **d)** To identify outliers

**Concept:** Forecasting extends patterns into the future.

---

#### Question 22: Which technique is commonly used for anomaly detection in time series data?

- **a)** Moving averages
- **b)** STL decomposition or statistical thresholds ✓
- **c)** Linear regression
- **d)** Clustering

**Concept:** Time series anomalies detected using seasonal decomposition or statistical bounds.

---

### Advanced Techniques

#### Question 53: Which method is commonly used for linear dimensionality reduction?

- **a)** K-means clustering
- **b)** Principal Component Analysis (PCA) ✓
- **c)** Decision trees
- **d)** Random sampling

**Concept:** PCA finds principal components explaining maximum variance.

---

#### Question 60: Which method is commonly used for non-linear dimensionality reduction?

- **a)** Linear regression
- **b)** t-SNE or UMAP ✓
- **c)** Mean calculation
- **d)** Sorting

**Concept:** Non-linear methods preserve local structure in lower dimensions.

---

## Topic-wise Summary

### 1. **Data Visualization**

- **Categorical Data**: Bar plots, count plots, pie charts
- **Continuous Data**: Histograms, density plots, box plots
- **Relationships**: Scatter plots, heatmaps, pair plots
- **Special Plots**: Violin plots, dendrograms, 3D plots

### 2. **Statistical Measures**

- **Central Tendency**: Mean, Median, Mode
- **Dispersion**: Variance, Standard Deviation, IQR, Range
- **Distribution Shape**: Skewness, Kurtosis
- **Position**: Quantiles, Percentiles

### 3. **Data Preprocessing**

- **Missing Values**: Deletion, Imputation (mean/median/mode)
- **Outliers**: Detection (IQR, Z-score), Treatment
- **Transformations**: Log, Square root, Normalization, Scaling
- **Encoding**: One-hot encoding, Label encoding

### 4. **Analysis Types**

- **Univariate**: Single variable analysis
- **Bivariate**: Two variable relationships
- **Multivariate**: Multiple variable interactions
- **Time Series**: Temporal patterns and forecasting

### 5. **Statistical Testing**

- **Hypothesis Testing**: Null vs Alternative hypothesis
- **Common Tests**: T-test, Chi-square, Shapiro-Wilk
- **Significance**: P-values, Confidence intervals

### 6. **Advanced Techniques**

- **Clustering**: K-means, Hierarchical, DBSCAN
- **Dimensionality Reduction**: PCA, LDA, t-SNE, UMAP
- **Feature Engineering**: Extraction, Selection, Creation

---

## Important Concepts

### Missing Data Handling

```python
# Check missing values
df.isnull().sum()

# Fill missing values
df['column'].fillna(df['column'].mean(), inplace=True)  # Numerical
df['column'].fillna(df['column'].mode()[0], inplace=True)  # Categorical
```

### Correlation Analysis

```python
# Correlation matrix
correlation_matrix = df.corr()

# Visualize correlations
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
```

### Outlier Detection

```python
# Using IQR
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['column'] < Q1 - 1.5*IQR) | (df['column'] > Q3 + 1.5*IQR)]
```

### Data Transformation

```python
# Log transformation
df['log_column'] = np.log1p(df['column'])

# Normalization (Min-Max scaling)
df['normalized'] = (df['column'] - df['column'].min()) / (df['column'].max() - df['column'].min())

# Standardization (Z-score)
df['standardized'] = (df['column'] - df['column'].mean()) / df['column'].std()
```

---

## Python/Pandas Quick Reference

### Basic Operations

| Command         | Description                   |
| --------------- | ----------------------------- |
| `df.head()`     | View first 5 rows             |
| `df.info()`     | Data types and missing values |
| `df.describe()` | Summary statistics            |
| `df.shape`      | Dimensions (rows, columns)    |
| `df.dtypes`     | Data types of columns         |

### Data Exploration

| Command                    | Description             |
| -------------------------- | ----------------------- |
| `df['col'].value_counts()` | Frequency counts        |
| `df['col'].unique()`       | Unique values           |
| `df['col'].nunique()`      | Number of unique values |
| `df.corr()`                | Correlation matrix      |
| `df.isnull().sum()`        | Count missing values    |

### Data Manipulation

| Command            | Description            |
| ------------------ | ---------------------- |
| `df.fillna()`      | Fill missing values    |
| `df.drop()`        | Remove rows/columns    |
| `df.groupby()`     | Group data             |
| `pd.get_dummies()` | One-hot encoding       |
| `df.apply()`       | Apply function to data |

### Visualization

| Command          | Description      |
| ---------------- | ---------------- |
| `plt.hist()`     | Histogram        |
| `plt.scatter()`  | Scatter plot     |
| `sns.boxplot()`  | Box plot         |
| `sns.heatmap()`  | Heatmap          |
| `sns.pairplot()` | Pair plot matrix |

---

## Study Tips

### 1. **Understand Concepts First**

- Don't memorize, understand WHY each technique is used
- Know the assumptions and limitations of each method

### 2. **Practice with Real Data**

- Use datasets like Titanic, Iris, Boston Housing
- Apply each concept practically

### 3. **Master the Basics**

- Statistical measures (mean, median, standard deviation)
- Basic plots (histogram, scatter, box plot)
- Data types and their appropriate handling

### 4. **Build a Mental Framework**

- **Explore**: Understand your data
- **Clean**: Handle missing values and outliers
- **Transform**: Prepare for analysis
- **Visualize**: Find patterns and insights
- **Test**: Validate assumptions

### 5. **Common Pitfalls to Avoid**

- Using mean for skewed distributions (use median)
- Ignoring missing values
- Not checking data types before analysis
- Using wrong plot types for data types
- Forgetting to scale features

### 6. **Key Differentiators**

- **Parametric vs Non-parametric**: Know when to use each
- **Correlation vs Causation**: Correlation doesn't imply causation
- **Sample vs Population**: Understand statistical inference

---

## Final Notes

- **Question 91** checks completion of "Python ka Chilla" prerequisite course
- Focus on understanding concepts rather than memorization
- Practice coding each concept for better retention
- Use this guide as a reference, not just for the quiz

**Good luck with your EDA journey!**
