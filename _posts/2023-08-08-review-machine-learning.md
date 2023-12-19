---
layout: post
title: "Basic Concepts of Machine Learning"
date: 2023-08-08 20:35:00 +1000
categories: machinelearning 
---

## Key elements of Machine Learning
Every ML algorithm has 3 components:
- Representation: How to represent knowledge.
- Evaluation: The way to evaluate candidate programs (hypothesis).
- Optimization: They way candidate programs are generated known as the search process.

All machine learning algorithms are combinations of these 3 components. A framework for understanding all algorithms.

## Types of Machine Learning
There are four types of machine learning:

- Supervised learning: (also called inductive learning) Training data includes desired outputs.  This is spam this is not, learning is supervised.
- Unsupervised learning: Training data does not include desired outputs. Example is clustering. It is hard to tell what is good learning and what is not.
- Semi-supervised learning: Training data includes a few desired outputs.
- Reinforcement learning: Rewards from a sequence of actions. AI types like it, it is the most ambitious type of learning.

Supervised learning is the most mature, the most studied and the type of learning used by most machine learning algorithms. Learning with supervision is much easier than learning without supervision.

- Classification: when the function being learned is discrete.
- Regression: when the function being learned is continuous.
- Probability Estimation: when the output of the function is a probability.

## Machine Learning in Practice
Machine learning algorithms are only a **very small part of using machine learning** in practice as a data analyst or data scientist. In practice, the process often looks like:

- Start Loop:
  - **Understand the domain, prior knowledge and goals**. Talk to domain experts. Often the goals are very unclear. You often have more things to try then you can possibly implement.
  - **Data integration, selection, cleaning and pre-processing**. This is often the most time consuming part. It is important to have high quality data. The more data you have, the more it sucks because the data is dirty. Garbage in, garbage out.
  - **Learning models**. The fun part. This part is very mature. The tools are general.
  - **Interpreting results**. Sometimes it does not matter how the model works as long it delivers results. Other domains require that the model is understandable. You will be challenged by human experts.
  - **Consolidating and deploying discovered knowledge**. The majority of projects that are successful in the lab are not used in practice. It is very hard to get something used.
- End Loop.

## Top 10 Common Machine Learning Algorithms
### 1. Linear Regression
It is used to estimate real values (cost of houses, number of calls, total sales, etc.) based on a continuous variable(s). Here, we establish the relationship between independent and dependent variables by fitting the best line. This best-fit line is known as the regression line and is represented by a linear equation Y= a*X + b.

Linear Regression is mainly of two types: Simple Linear Regression and Multiple Linear Regression. Simple Linear Regression is characterized by one independent variable. And, Multiple Linear Regression(as the name suggests) is characterized by multiple (more than 1) independent variables. While finding the best-fit line, you can fit a polynomial or curvilinear regression. And these are known as polynomial or curvilinear regression.

### 2. Logistic Regression
It is a classification algorithm, not a regression algorithm. It is used to estimate discrete values ( Binary values like 0/1, yes/no, true/false ) based on a given set of independent variable(s). In simple words, it predicts the probability of the occurrence of an event by fitting data to a logistic function (sigmoid function). Hence, it is also known as logit regression. Since it predicts the probability, its output values lie between 0 and 1 (as expected).

There are many different steps that could be tried in order to improve the model:

- including interaction terms
- removing features
- regularization techniques
- using a non-linear model

### 3. Decision Tree
It is a type of supervised learning algorithm that is mostly used for classification problems. Surprisingly, it works for both categorical and continuous dependent variables. In this algorithm, we split the population into two or more homogeneous sets. This is done based on the most significant attributes/ independent variables to make as distinct groups as possible.

### 4. SVM (Support Vector Machine)
In SVM algorithm, we plot each data item as a point in n-dimensional space (where n is the number of features you have), with the value of each feature being the value of a particular coordinate.

SVM models is suitable with small to medium-sized datasets and high-demensional data.


### 5. Naive Bayesian
It is a classification technique based on Bayes’ theorem with an assumption of independence between predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. For example, a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter. Even if these features depend on each other or upon the existence of the other features, a naive Bayes classifier would consider all of these properties to independently contribute to the probability that this fruit is an apple.

The Naive Bayesian model is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods.

### 6. kNN (k-Nearest Neighbors)
It can be used for both classification and regression problems. However, it is more widely used in classification problems in the industry. K nearest neighbors is a simple algorithm that stores all available cases and classifies new cases by a majority vote of its k neighbors. The case assigned to the class is most common amongst its K nearest neighbors measured by a distance function.

These distance functions can be Euclidean, Manhattan, Minkowski, and Hamming distances. The first three functions are used for continuous functions, and the fourth one (Hamming) for categorical variables. If K = 1, then the case is simply assigned to the class of its nearest neighbor. At times, choosing K turns out to be a challenge while performing kNN modeling.

Things to consider before selecting kNN:

- KNN is computationally expensive
- Variables should be normalized else higher range variables can bias it
- Works on pre-processing stage more before going for kNN like an outlier, noise removal

### 7. k-Means
It is a type of unsupervised algorithm which solves the clustering problem. Its procedure follows a simple and easy way to classify a given data set through a certain number of clusters (assume k clusters). Data points inside a cluster are homogeneous and heterogeneous to peer groups.

How K-means forms cluster:
1. K-means picks k number of points for each cluster known as centroids.
2. Each data point forms a cluster with the closest centroids, i.e., k clusters.
3. Finds the centroid of each cluster based on existing cluster members. Here we have new centroids.
4. As we have new centroids, repeat steps 2 and 3. Find the closest distance for each data point from new centroids and get associated with new k-clusters. Repeat this process until convergence occurs, i.e., centroids do not change.

### 8. Random Forest
Random Forest is a trademarked term for an ensemble learning of decision trees. In Random Forest, we’ve got a collection of decision trees (also known as “Forest”). To classify a new object based on attributes, each tree gives a classification, and we say the tree “votes” for that class. The forest chooses the classification having the most votes (over all the trees in the forest).

Each tree is planted and grown as follows:
1. If the number of cases in the training set is N, then a sample of N cases is taken at random but with replacement. This sample will be the training set for growing the tree.
2. If there are M input variables, a number m << M is specified such that at each node, m variables are selected at random out of the M, and the best split on this m is used to split the node. The value of m is held constant during the forest growth.
3. Each tree is grown to the largest extent possible. There is no pruning.

### 9. Dimensionality Reduction Algorithms
To know more about these algorithms, you can read [Beginners Guide To Learn Dimension Reduction Techniques](https://www.analyticsvidhya.com/blog/2015/07/dimension-reduction-methods/).

### 10. Gradient Boosting Algorithms
Gradient Boosting is a machine learning ensemble technique used for regression and classification tasks. It builds a strong predictive model by combining the predictions of multiple individual models, typically decision trees. Gradient Boosting is known for its high predictive accuracy and ability to handle complex relationships within data.

Now, let's look at 4 most commonly used gradient boosting algorithms.

1. GBM
2. XGBoost
3. LightGBM
4. CatBoost

  

## [Reference](https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/#Who_Can_Benefit_the_Most_From_This_Guide?)