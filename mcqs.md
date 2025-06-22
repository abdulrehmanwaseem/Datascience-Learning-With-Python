# Exploratory Data Analysis (EDA) Quiz - Complete Guide

## Questions with Options and Key Concepts

### Question 1: Which graph is used to display the relationship between two categorical variables?

**Options:**
a) Scatter plot
b) Histogram
c) Heatmap/Contingency table
d) Box plot

**Answer: c) Heatmap/Contingency table**
**Concept:** For two categorical variables, we use cross-tabulation visualized as heatmaps or stacked bar charts.

### Question 2: In the Titanic dataset how would you find the number of passengers who survived?

**Options:**
a) `df['Survived'].sum()`
b) `df['Survived'].count()`
c) `df['Survived'].mean()`
d) `df['Survived'].std()`

**Answer: a) `df['Survived'].sum()`
**Concept:\*\* If 'Survived' is binary (0/1), summing gives the count of survivors.

### Question 3: Which plot is used to visualize the distribution of a categorical variable?

**Options:**
a) Histogram
b) Bar plot/Count plot
c) Scatter plot
d) Line plot

**Answer: b) Bar plot/Count plot**
**Concept:** Bar plots show frequency/count for each category.

### Question 4: What is the significance of the median in a dataset?

**Options:**
a) It's always equal to the mean
b) It's the middle value and robust to outliers
c) It's the most frequent value
d) It measures spread

**Answer: b) It's the middle value and robust to outliers**
**Concept:** Median is the 50th percentile, less affected by extreme values than mean.

### Question 5: What is the main objective of clustering in EDA?

**Options:**
a) To predict future values
b) To group similar data points together
c) To reduce dimensions
d) To remove outliers

**Answer: b) To group similar data points together**
**Concept:** Clustering identifies natural groupings in data.

### Question 6: What does the term 'Bivariate Analysis' refer to in EDA?

**Options:**
a) Analysis of one variable
b) Analysis of relationship between two variables
c) Analysis of multiple variables
d) Time-based analysis

**Answer: b) Analysis of relationship between two variables**
**Concept:** Bivariate = two variables analyzed together.

### Question 7: Which method is commonly used to handle missing categorical data?

**Options:**
a) Mean imputation
b) Mode imputation or creating 'Missing' category
c) Median imputation
d) Linear interpolation

**Answer: b) Mode imputation or creating 'Missing' category**
**Concept:** For categorical data, use mode or treat missing as a separate category.

### Question 8: What is the null hypothesis in hypothesis testing?

**Options:**
a) The hypothesis we want to prove
b) The assumption of no effect or no difference
c) The alternative explanation
d) The rejected hypothesis

**Answer: b) The assumption of no effect or no difference**
**Concept:** H₀ assumes status quo or no significant difference.

### Question 9: Why might you perform a log transformation on a dataset?

**Options:**
a) To remove outliers
b) To handle skewed distributions and make them more normal
c) To increase values
d) To convert to integers

**Answer: b) To handle skewed distributions and make them more normal**
**Concept:** Log transformation reduces right skewness.

### Question 10: What does the term 'Multivariate Analysis' refer to in EDA?

**Options:**
a) Analysis of one variable
b) Analysis of two variables
c) Analysis of multiple variables simultaneously
d) Time series analysis

**Answer: c) Analysis of multiple variables simultaneously**
**Concept:** Multivariate = analyzing relationships among 3+ variables.

### Question 11: What is the primary goal of forecasting in time series analysis?

**Options:**
a) To understand past patterns
b) To predict future values based on historical data
c) To remove seasonality
d) To identify outliers

**Answer: b) To predict future values based on historical data**
**Concept:** Forecasting extends patterns into the future.

### Question 12: Which plot is used to visualize the relationship between a numerical and a categorical variable?

**Options:**
a) Scatter plot
b) Box plot or Violin plot
c) Histogram
d) Pie chart

**Answer: b) Box plot or Violin plot**
**Concept:** Box plots show distribution of numerical variable across categories.

### Question 13: What does the term 'Feature Extraction' refer to in EDA?

**Options:**
a) Removing features
b) Creating new features from existing data
c) Selecting best features
d) Normalizing features

**Answer: b) Creating new features from existing data**
**Concept:** Feature extraction derives new meaningful variables.

### Question 14: What is the goal of anomaly detection in EDA?

**Options:**
a) To find average values
b) To identify unusual or outlier data points
c) To group similar items
d) To reduce dimensions

**Answer: b) To identify unusual or outlier data points**
**Concept:** Anomalies are data points that deviate significantly from normal patterns.

### Question 15: What is the primary goal of feature selection in EDA?

**Options:**
a) To create new features
b) To identify and keep most relevant features
c) To transform features
d) To normalize features

**Answer: b) To identify and keep most relevant features**
**Concept:** Feature selection reduces dimensionality by keeping important variables.

### Question 16: What does the term 'Kurtosis' signify in a dataset?

**Options:**
a) Central tendency
b) Spread of data
c) Tailedness/peakedness of distribution
d) Skewness direction

**Answer: c) Tailedness/peakedness of distribution**
**Concept:** Kurtosis measures whether data has heavy or light tails compared to normal distribution.

### Question 17: What is a Quantile in a dataset?

**Options:**
a) The mean value
b) Values that divide data into equal-sized groups
c) The most frequent value
d) The range of data

**Answer: b) Values that divide data into equal-sized groups**
**Concept:** Quantiles are cut points (e.g., quartiles divide into 4 parts).

### Question 18: What is the main objective of normalization in data processing?

**Options:**
a) To remove outliers
b) To scale features to a common range
c) To increase values
d) To convert to integers

**Answer: b) To scale features to a common range**
**Concept:** Normalization brings different features to comparable scales.

### Question 19: How are missing values represented in a Pandas DataFrame?

**Options:**
a) 0
b) NaN (Not a Number)
c) NULL
d) Empty string

**Answer: b) NaN (Not a Number)**
**Concept:** Pandas uses NaN to represent missing values.

### Question 20: Which measure is commonly used to understand the dispersion or spread in a dataset?

**Options:**
a) Mean
b) Median
c) Standard deviation or variance
d) Mode

**Answer: c) Standard deviation or variance**
**Concept:** Standard deviation measures how spread out values are from the mean.

### Question 21: What is the primary goal of correlation analysis?

**Options:**
a) To find causation
b) To measure the strength and direction of relationships between variables
c) To predict values
d) To remove outliers

**Answer: b) To measure the strength and direction of relationships between variables**
**Concept:** Correlation quantifies linear relationships (-1 to +1).

### Question 22: Which technique is commonly used for anomaly detection in time series data?

**Options:**
a) Moving averages
b) STL decomposition or statistical thresholds
c) Linear regression
d) Clustering

**Answer: b) STL decomposition or statistical thresholds**
**Concept:** Time series anomalies detected using seasonal decomposition or statistical bounds.

### Question 23: Which plot can be used to visualize the distribution of categorical data?

**Options:**
a) Histogram
b) Bar plot or Pie chart
c) Box plot
d) Scatter plot

**Answer: b) Bar plot or Pie chart**
**Concept:** Categorical distributions shown with frequency/proportion plots.

### Question 24: What is the purpose of binning data in EDA?

**Options:**
a) To remove data
b) To group continuous data into discrete intervals
c) To calculate mean
d) To find outliers

**Answer: b) To group continuous data into discrete intervals**
**Concept:** Binning converts continuous variables to categorical for analysis.

### Question 25: What does a histogram represent?

**Options:**
a) Relationship between two variables
b) Distribution of a continuous variable
c) Categories and their counts
d) Time series data

**Answer: b) Distribution of a continuous variable**
**Concept:** Histograms show frequency distribution using bins.

### Question 26: What is the primary goal of the Chi-square test in EDA?

**Options:**
a) To compare means
b) To test independence between categorical variables
c) To find correlation
d) To detect outliers

**Answer: b) To test independence between categorical variables**
**Concept:** Chi-square tests if two categorical variables are related.

### Question 27: What is the purpose of binning or bucketing data in EDA?

**Options:**
a) To increase precision
b) To discretize continuous variables
c) To remove outliers
d) To normalize data

**Answer: b) To discretize continuous variables**
**Concept:** Same as Q24 - grouping continuous data into categories.

### Question 28: Which method is commonly used to calculate the covariance between variables?

**Options:**
a) `df.mean()`
b) `df.cov()`
c) `df.std()`
d) `df.sum()`

**Answer: b) `df.cov()`**
**Concept:** Covariance measures joint variability of two variables.

### Question 29: What is the purpose of data transformation in EDA?

**Options:**
a) To delete data
b) To make data more suitable for analysis
c) To increase data size
d) To remove all outliers

**Answer: b) To make data more suitable for analysis**
**Concept:** Transformations improve data properties for analysis.

### Question 30: What is the main challenge in handling categorical data in EDA?

**Options:**
a) They're always missing
b) They can't be directly used in mathematical operations
c) They're always numeric
d) They don't have patterns

**Answer: b) They can't be directly used in mathematical operations**
**Concept:** Categorical data needs encoding for numerical analysis.

### Question 31: What is the purpose of a scatter matrix plot in EDA?

**Options:**
a) To show one variable's distribution
b) To show pairwise relationships between multiple variables
c) To show time series
d) To show categories

**Answer: b) To show pairwise relationships between multiple variables**
**Concept:** Scatter matrix displays all variable pairs in a grid.

### Question 32: Which plot is commonly used to visualize hierarchical clustering?

**Options:**
a) Scatter plot
b) Dendrogram
c) Box plot
d) Histogram

**Answer: b) Dendrogram**
**Concept:** Dendrograms show hierarchical relationships as tree structures.

### Question 33: What is the benefit of using a violin plot in EDA?

**Options:**
a) Shows only median
b) Combines box plot with kernel density estimation
c) Shows only outliers
d) Shows time series

**Answer: b) Combines box plot with kernel density estimation**
**Concept:** Violin plots show distribution shape plus summary statistics.

### Question 34: How is the mean of a dataset calculated?

**Options:**
a) Middle value when sorted
b) Sum of all values divided by count
c) Most frequent value
d) Difference between max and min

**Answer: b) Sum of all values divided by count**
**Concept:** Mean = Σx/n (arithmetic average).

### Question 35: What does the term 'Outlier' signify in a dataset?

**Options:**
a) Average values
b) Data points significantly different from others
c) Missing values
d) Duplicate values

**Answer: b) Data points significantly different from others**
**Concept:** Outliers are extreme values that deviate from the pattern.

### Question 36: Why might you use a 3D plot in EDA?

**Options:**
a) To make it colorful
b) To visualize relationships between three variables
c) To hide outliers
d) To show time

**Answer: b) To visualize relationships between three variables**
**Concept:** 3D plots add a third dimension for multivariate visualization.

### Question 37: Why is outlier detection important in EDA?

**Options:**
a) Outliers are always errors
b) Outliers can significantly affect analysis and models
c) To increase data size
d) To make data normal

**Answer: b) Outliers can significantly affect analysis and models**
**Concept:** Outliers can skew statistics and model performance.

### Question 38: What is the purpose of data cleaning in EDA?

**Options:**
a) To delete all data
b) To handle errors, inconsistencies, and missing values
c) To increase data
d) To make data complex

**Answer: b) To handle errors, inconsistencies, and missing values**
**Concept:** Data cleaning ensures quality and reliability.

### Question 39: Which plot is commonly used to visualize the correlation between variables?

**Options:**
a) Bar plot
b) Heatmap or correlation matrix
c) Pie chart
d) Line plot

**Answer: b) Heatmap or correlation matrix**
**Concept:** Heatmaps use colors to show correlation strength.

### Question 40: What does the p-value signify in hypothesis testing?

**Options:**
a) Population size
b) Probability of observing results if null hypothesis is true
c) Power of test
d) Prediction accuracy

**Answer: b) Probability of observing results if null hypothesis is true**
**Concept:** Low p-value suggests rejecting null hypothesis.

### Question 41: What is the significance of the variance in a dataset?

**Options:**
a) It measures central tendency
b) It measures the average squared deviation from mean
c) It's always zero
d) It measures skewness

**Answer: b) It measures the average squared deviation from mean**
**Concept:** Variance quantifies data spread (σ² = Σ(x-μ)²/n).

### Question 42: What does the term 'Encoding' refer to in handling categorical data?

**Options:**
a) Deleting categories
b) Converting categories to numerical representation
c) Creating categories
d) Sorting categories

**Answer: b) Converting categories to numerical representation**
**Concept:** Encoding transforms categories for numerical processing.

### Question 43: How can you handle imbalanced data in a dataset?

**Options:**
a) Remove all minority class
b) Resampling, SMOTE, or class weights
c) Ignore the problem
d) Use only majority class

**Answer: b) Resampling, SMOTE, or class weights**
**Concept:** Various techniques balance class distributions.

### Question 44: Which plot is used to visualize the relationships between more than two variables?

**Options:**
a) Simple scatter plot
b) Parallel coordinates or scatter matrix
c) Histogram
d) Pie chart

**Answer: b) Parallel coordinates or scatter matrix**
**Concept:** Multiple variables need specialized visualization techniques.

### Question 45: Which plot is commonly used to visualize the distribution of a continuous variable?

**Options:**
a) Bar plot
b) Histogram or density plot
c) Pie chart
d) Scatter plot

**Answer: b) Histogram or density plot**
**Concept:** Continuous distributions shown with bins or smooth curves.

### Question 46: How can you handle missing values in time series data?

**Options:**
a) Always delete them
b) Interpolation or forward/backward fill
c) Replace with zero
d) Ignore them

**Answer: b) Interpolation or forward/backward fill**
**Concept:** Time series needs special methods preserving temporal patterns.

### Question 47: What is the purpose of hypothesis testing in EDA?

**Options:**
a) To visualize data
b) To make statistical inferences about data
c) To clean data
d) To transform data

**Answer: b) To make statistical inferences about data**
**Concept:** Hypothesis testing validates assumptions statistically.

### Question 48: What is the benefit of scaling data in EDA?

**Options:**
a) Makes all values zero
b) Brings features to comparable ranges
c) Removes features
d) Increases dimensionality

**Answer: b) Brings features to comparable ranges**
**Concept:** Scaling prevents features with large values from dominating.

### Question 49: How can you determine the strength of a relationship between two variables?

**Options:**
a) By plotting them
b) Calculate correlation coefficient
c) Calculate mean
d) Count values

**Answer: b) Calculate correlation coefficient**
**Concept:** Correlation coefficient (-1 to +1) quantifies relationship strength.

### Question 50: Why might you use logistic regression in EDA?

**Options:**
a) For continuous prediction
b) To understand factors affecting binary outcomes
c) For clustering
d) For time series

**Answer: b) To understand factors affecting binary outcomes**
**Concept:** Logistic regression explores binary classification relationships.

### Question 51: Which method is commonly used to impute missing values in a dataset?

**Options:**
a) Always delete rows
b) Mean/median/mode imputation or advanced methods
c) Set to infinity
d) Leave as is

**Answer: b) Mean/median/mode imputation or advanced methods**
**Concept:** Various imputation strategies based on data type and pattern.

### Question 52: What is the primary purpose of data transformation in EDA?

**Options:**
a) To lose information
b) To improve data properties for analysis
c) To increase missing values
d) To make data unreadable

**Answer: b) To improve data properties for analysis**
**Concept:** Transformations optimize data for specific analyses.

### Question 53: Which method is commonly used for linear dimensionality reduction?

**Options:**
a) K-means clustering
b) Principal Component Analysis (PCA)
c) Decision trees
d) Random sampling

**Answer: b) Principal Component Analysis (PCA)**
**Concept:** PCA finds principal components explaining maximum variance.

### Question 54: What is the primary goal of dimensionality reduction in EDA?

**Options:**
a) To increase features
b) To reduce features while preserving information
c) To remove all features
d) To duplicate features

**Answer: b) To reduce features while preserving information**
**Concept:** Dimensionality reduction simplifies data maintaining key patterns.

### Question 55: Which method is commonly used to test the normality of a dataset?

**Options:**
a) T-test
b) Shapiro-Wilk test or Q-Q plot
c) Chi-square test
d) F-test

**Answer: b) Shapiro-Wilk test or Q-Q plot**
**Concept:** Statistical tests and visual methods check normal distribution.

### Question 56: What is the primary goal of forecasting models in time series analysis?

**Options:**
a) To understand history
b) To predict future values
c) To remove trends
d) To create seasonality

**Answer: b) To predict future values**
**Concept:** Forecasting extends patterns forward in time.

### Question 57: What does a scatter plot signify?

**Options:**
a) Distribution of one variable
b) Relationship between two continuous variables
c) Categories and counts
d) Time progression

**Answer: b) Relationship between two continuous variables**
**Concept:** Scatter plots show how two variables vary together.

### Question 58: What is the benefit of encoding categorical data into numerical data?

**Options:**
a) To lose information
b) To enable mathematical operations and modeling
c) To remove categories
d) To create errors

**Answer: b) To enable mathematical operations and modeling**
**Concept:** Numerical encoding allows algorithmic processing.

### Question 59: Which method is commonly used to handle missing data?

**Options:**
a) Always ignore
b) Deletion, imputation, or model-based methods
c) Set to maximum value
d) Duplicate existing values

**Answer: b) Deletion, imputation, or model-based methods**
**Concept:** Multiple strategies exist based on missing data patterns.

### Question 60: Which method is commonly used for non-linear dimensionality reduction?

**Options:**
a) Linear regression
b) t-SNE or UMAP
c) Mean calculation
d) Sorting

**Answer: b) t-SNE or UMAP**
**Concept:** Non-linear methods preserve local structure in lower dimensions.

### Question 61: What is the IQR (Interquartile Range) used for in EDA?

**Options:**
a) Finding mean
b) Measuring spread and detecting outliers
c) Finding mode
d) Calculating total

**Answer: b) Measuring spread and detecting outliers**
**Concept:** IQR = Q3 - Q1, robust measure of spread.

### Question 62: What is the primary goal of Linear Discriminant Analysis (LDA) in EDA?

**Options:**
a) Clustering
b) Dimensionality reduction maximizing class separation
c) Regression
d) Time series analysis

**Answer: b) Dimensionality reduction maximizing class separation**
**Concept:** LDA finds projections that best separate classes.

### Question 63: What is the purpose of a Pair Plot in EDA?

**Options:**
a) Show one variable
b) Show all pairwise relationships in dataset
c) Show time series
d) Show categories only

**Answer: b) Show all pairwise relationships in dataset**
**Concept:** Pair plots create scatter plot matrix of all variable pairs.

### Question 64: Which test is commonly used to compare the means of two groups?

**Options:**
a) Chi-square test
b) T-test
c) Correlation test
d) Normality test

**Answer: b) T-test**
**Concept:** T-test compares means between two groups statistically.

### Question 65: What is the purpose of a box plot in EDA?

**Options:**
a) Show only mean
b) Display distribution summary and outliers
c) Show time series
d) Show correlations

**Answer: b) Display distribution summary and outliers**
**Concept:** Box plots show quartiles, median, and outliers.

### Question 66: What is the alternative hypothesis in hypothesis testing?

**Options:**
a) The null hypothesis
b) The hypothesis of an effect or difference
c) Always false
d) The accepted hypothesis

**Answer: b) The hypothesis of an effect or difference**
**Concept:** H₁ proposes there is a significant effect/difference.

### Question 67: Which clustering method does not require the number of clusters to be specified a priori?

**Options:**
a) K-means
b) Hierarchical clustering or DBSCAN
c) K-medoids
d) Linear regression

**Answer: b) Hierarchical clustering or DBSCAN**
**Concept:** Some algorithms determine clusters automatically.

### Question 68: What does multivariate analysis allow you to do?

**Options:**
a) Analyze one variable
b) Analyze multiple variables and their interactions
c) Remove variables
d) Create single variable

**Answer: b) Analyze multiple variables and their interactions**
**Concept:** Multivariate analysis studies complex relationships.

### Question 69: Which method is used to calculate the correlation between variables?

**Options:**
a) `df.sum()`
b) `df.corr()` or Pearson/Spearman correlation
c) `df.mean()`
d) `df.count()`

**Answer: b) `df.corr()` or Pearson/Spearman correlation**
**Concept:** Correlation methods quantify variable relationships.

### Question 70: Which method is used to replace missing values in a dataset?

**Options:**
a) `df.drop()`
b) `df.fillna()` or imputation methods
c) `df.sort()`
d) `df.copy()`

**Answer: b) `df.fillna()` or imputation methods**
**Concept:** Various methods fill missing values appropriately.

### Question 71: Which method is used to calculate the correlation matrix?

**Options:**
a) `df.describe()`
b) `df.corr()`
c) `df.info()`
d) `df.shape`

**Answer: b) `df.corr()`**
**Concept:** Correlation matrix shows all pairwise correlations.

### Question 72: Which method is commonly used for seasonal decomposition of time series?

**Options:**
a) Linear regression
b) STL decomposition or seasonal_decompose
c) K-means
d) PCA

**Answer: b) STL decomposition or seasonal_decompose**
**Concept:** Decomposition separates trend, seasonal, and residual components.

### Question 73: What is the purpose of one-hot encoding in handling categorical data?

**Options:**
a) To remove categories
b) To create binary columns for each category
c) To order categories
d) To reduce categories

**Answer: b) To create binary columns for each category**
**Concept:** One-hot encoding creates dummy variables for categories.

### Question 74: Why is time series analysis important in EDA?

**Options:**
a) It's not important
b) To understand temporal patterns and dependencies
c) To remove time
d) To create static data

**Answer: b) To understand temporal patterns and dependencies**
**Concept:** Time series reveals trends, seasonality, and temporal relationships.

### Question 75: What is the purpose of dimensionality reduction in EDA?

**Options:**
a) To complicate analysis
b) To simplify data while preserving important information
c) To increase dimensions
d) To remove all data

**Answer: b) To simplify data while preserving important information**
**Concept:** Reduces complexity for visualization and analysis.

### Question 76: What is the primary objective of Exploratory Data Analysis (EDA)?

**Options:**
a) To build final models
b) To understand data patterns and characteristics
c) To delete data
d) To finalize results

**Answer: b) To understand data patterns and characteristics**
**Concept:** EDA explores and understands data before modeling.

### Question 77: What is the primary purpose of advanced visualization techniques in EDA?

**Options:**
a) To confuse viewers
b) To reveal complex patterns and relationships
c) To hide information
d) To use colors only

**Answer: b) To reveal complex patterns and relationships**
**Concept:** Advanced visualizations uncover hidden insights.

### Question 78: Which method is used to identify the unique values in a column of a DataFrame?

**Options:**
a) `df.sum()`
b) `df['column'].unique()` or `df['column'].nunique()`
c) `df.mean()`
d) `df.std()`

**Answer: b) `df['column'].unique()` or `df['column'].nunique()`**
**Concept:** These methods find distinct values in data.

### Question 79: What does a Box Plot represent?

**Options:**
a) Only the mean
b) Five-number summary and outliers
c) Only outliers
d) Time series

**Answer: b) Five-number summary and outliers**
**Concept:** Shows min, Q1, median, Q3, max, and outliers.

### Question 80: Which measure is used to understand the spread of a dataset?

**Options:**
a) Mean only
b) Standard deviation, variance, or range
c) Mode only
d) Count only

**Answer: b) Standard deviation, variance, or range**
**Concept:** Multiple measures quantify data dispersion.

### Question 81: Which method is commonly used to identify the optimal number of clusters in a dataset?

**Options:**
a) Random selection
b) Elbow method or silhouette analysis
c) Always use 2
d) Maximum possible

**Answer: b) Elbow method or silhouette analysis**
**Concept:** These methods find natural cluster numbers.

### Question 82: How can you describe the skewness of a dataset?

**Options:**
a) It's always zero
b) Positive (right), negative (left), or zero skew
c) Only positive
d) Only negative

**Answer: b) Positive (right), negative (left), or zero skew**
**Concept:** Skewness measures distribution asymmetry.

### Question 83: Which plot is often used in time series analysis?

**Options:**
a) Pie chart
b) Line plot or time series plot
c) Bar chart only
d) Scatter plot only

**Answer: b) Line plot or time series plot**
**Concept:** Line plots show temporal progression.

### Question 84: Why might you use a box plot in EDA?

**Options:**
a) To show exact values
b) To compare distributions and identify outliers
c) To show time
d) To calculate mean

**Answer: b) To compare distributions and identify outliers**
**Concept:** Box plots efficiently compare multiple distributions.

### Question 85: Which plot is used to visualize the distribution of a dataset?

**Options:**
a) Only scatter plots
b) Histogram, density plot, or box plot
c) Only line plots
d) Only bar plots

**Answer: b) Histogram, density plot, or box plot**
**Concept:** Multiple plots can show distributions differently.

### Question 86: Why are heatmaps used in EDA?

**Options:**
a) To cool data
b) To visualize matrices like correlations or frequencies
c) To heat data
d) To remove patterns

**Answer: b) To visualize matrices like correlations or frequencies**
**Concept:** Heatmaps use color intensity for matrix visualization.

### Question 87: Which measure provides the spread of data around the mean?

**Options:**
a) Median
b) Standard deviation
c) Mode
d) Minimum

**Answer: b) Standard deviation**
**Concept:** Standard deviation measures average distance from mean.

### Question 88: Which method is commonly used for feature extraction in EDA?

**Options:**
a) Deletion
b) PCA, polynomial features, or domain-specific methods
c) Copying
d) Ignoring

**Answer: b) PCA, polynomial features, or domain-specific methods**
**Concept:** Various techniques create informative features.

### Question 89: What is the primary goal of statistical inference in EDA?

**Options:**
a) To visualize data
b) To draw conclusions about populations from samples
c) To delete data
d) To sort data

**Answer: b) To draw conclusions about populations from samples**
**Concept:** Inference generalizes findings beyond the sample.

### Question 90: Why is feature engineering important in EDA?

**Options:**
a) It's not important
b) To create more informative features for analysis
c) To remove all features
d) To complicate data

**Answer: b) To create more informative features for analysis**
**Concept:** Good features improve model performance and insights.

## Study Tips for EDA:

1. **Master the basics**: Understand mean, median, mode, variance, standard deviation
2. **Learn visualization**: Know when to use each plot type
3. **Practice with code**: Use pandas, matplotlib, seaborn regularly
4. **Understand data types**: Numerical vs categorical handling
5. **Know preprocessing**: Missing values, outliers, transformations
6. **Statistical concepts**: Hypothesis testing, correlations, distributions
7. **Advanced techniques**: PCA, clustering, time series decomposition

## Key Python/Pandas Commands to Remember:

- `df.describe()` - Summary statistics
- `df.info()` - Data types and missing values
- `df.corr()` - Correlation matrix
- `df['col'].value_counts()` - Frequency counts
- `df.isnull().sum()` - Count missing values
- `df.fillna()` - Fill missing values
- `df['col'].unique()` - Unique values
- `sns.heatmap()` - Correlation visualization
- `plt.hist()` - Histogram
- `sns.boxplot()` - Box plot
