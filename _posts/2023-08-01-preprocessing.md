---
layout: post
title: "Preprocessing"
date: 2023-08-01 13:33:00 +1000
categories: machinelearning
---
Data preprocessing is an important step in the data mining process. It refers to the cleaning, transforming, and integrating of data in order to make it ready for analysis. The goal of data preprocessing is to improve the quality of the data and to make it more suitable for the specific data mining task.

## Some common steps in data preprocessing include:
### Data Cleaning
This involves indentifying and correcting errors or inconsistenies in the data, such as **missing values, outliers, and duplicates**. Various techniques can be used for data cleaning, such as **imputation**, **removal**, and **transformation**.
### Dat Integration
This involves combining data from multiple sources to create a **unified dataset**. Data integration can be challenging as it requires handling data with **different formats, structures and semantics**. Techniques such as **record linkage*** and **data fusion*** can be used for data intergration.
### Data Transformation
This involves converting the data into a **suitable format** for analysis. Common techniques used in data transformation include **normalization**, **standardization**, and **discretization**. Normalization is used to scale the data to a common range, while standardization is used to transform the data to have zero mean and unit variance. Discretization is used to convert continuous data into discrete categories.
### Data Reduction
This involves **reducing the size** of the dataset while **preserving the important information**. Data reduction can be achieved through techniques such as **feature selection and feature extraction**. Feature selection involves selecting a subset of relevant features from the dataset, while feature extraction involves transforming the data into a lower-dimensional space while preserving the important information.

### Data Discretization
This involves **dividing continuous data** into **discrete** categories or intervals. Discretization is often used in data mining and machine learning algorithms that require categorical data. Discretization can be achieved through techniques such as equal **width binning**, equal **frequency binning**, and **clustering**.

### Data Normalization
This involves scaling the data to a common range, such as between 0 and 1 or -1 and 1. Normalization is often used to handle data with different units and scales. Common normalization techniques include min-max normalization, z-score normalization, and decimal scaling.

![Alt text](image.png)

## Steps Involved in Data Preprocessing
1. **Data Cleaning**:
The data can have many irrelevant and missing parts. To handle this part, data cleaning is done. It involves handling of missing data, noisy data etc.

   * **Missing Data**: 
This situation arises when some data is missing in the data. It can be handled in various ways.


     1. **Ignore the tuples**:
  This approach is suitable only when the dataset we have is quite large and multiple values are missing within a tuple. 
   
     1. **Fill the Missing values**: 
  There are various ways to do this task. You can choose to fill the missing values manually, by attribute mean or the most probable value. 
 
   * **Noisy Data**:
Noisy data is a meaningless data that can’t be interpreted by machines.It can be generated due to faulty data collection, data entry errors etc. It can be handled in following ways : 
     1. **Binning Method**: 
  This method works on sorted data in order to smooth it. The whole data is divided into segments of equal size and then various methods are performed to complete the task. Each segmented is handled separately. One can replace all data in a segment by its mean or boundary values can be used to complete the task. 
   
     1. **Regression**: 
  Here data can be made smooth by fitting it to a regression function.The regression used may be linear (having one independent variable) or multiple (having multiple independent variables). 
   
     1. **Clustering**: 
  This approach groups the similar data in a cluster. The outliers may be undetected or it will fall outside the clusters.
1. **Data Transformation** This step is taken in order to transform the data in appropriate forms suitable for mining process. This involves following ways:
   1. **Normalization**: It is done in order to scale the data values in a specified range (-1.0 to 1.0 or 0.0 to 1.0)
   2. **Attribute Selection**:  New attributes are constructed from the given set of attributes to help the mining process.
   3. **Discretization**: This is done to replace the raw values of numeric attribute by interval levels or conceptual levels.
   4. **Concept Hierarchy Generation**: Here attributes are converted from lower level to higher level in hierarchy. For example: attribute 'city' can be converted to 'country'.

2. **Data Reduction**
Data reduction is a crucial step in the data mining process that involves reducing the size of the dataset while preserving the important information.

   1. **Feature Selection**: This involves selecting a subset of relevant features from the dataset. Feature selection is often performed to remove irrelevant or redundant features from the dataset. It can be done using various techniques such as correlation analysis, mutual information, and principal component analysis (PCA).

    2. **Feature Extraction**: This involves transforming the data into a lower-dimensional space while preserving the important information. Feature extraction is often used when the original features are high-dimensional and complex. It can be done using techniques such as PCA, linear discriminant analysis (LDA), and non-negative matrix factorization (NMF).

    3. **Sampling**: This involves selecting a subset of data points from the dataset. Sampling is often used to reduce the size of the dataset while preserving the important information. It can be done using techniques such as **random sampling, stratified sampling, and systematic sampling**.

   2. **Clustering**: This involves **grouping similar data points together** into clusters. Clustering is often used to reduce the size of the dataset by replacing similar data points with a representative centroid. It can be done using techniques such as k-means, hierarchical clustering, and density-based clustering.

    4. **Compression**: This involves compressing the dataset while preserving the important information. Compression is often used to reduce the size of the dataset for storage and transmission purposes. It can be done using techniques such as wavelet compression, JPEG compression, and gzip compression.