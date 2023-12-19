---
layout: post
title: "Performance Metrics in Machine Learning"
date: 2023-08-01 22:33:00 +1000
categories: machinelearning 
---

Performance metrics in machine learning are essential for assessing the effectiveness and reliability of models. They’re a key element of every machine learning pipeline, allowing developers to fine-tune their algorithms and drive improvements.

The metrics can be broadly categorized into two main types: regression and classification metrics.

## Top regression metrics
### Mean Absolute Error (MAE)
It measures the average magnitude of errors between predicted and actual values without considering their direction. MAE is especially useful in applications that aim to minimize the average error and is less sensitive to outliers than other metrics like Mean Squared Error (MSE).

$$MAE = \frac{1}{n}\sum_i^n|y_i - \hat{y_i}|$$

- **What it shows**:
MAE measures the average magnitude of errors in the predictions made by the model (without considering their direction).

- **When to use**:
Use MAE when you want a simple, interpretable metric to evaluate the performance of your regression model.

- **When to avoid**:
Avoid using MAE to emphasize the impact of larger errors, as it does not penalize them heavily.

```python
import torch

act_vals = torch.tensor([2,4,6,8])
pred_vals = torch.tensor([2.5, 3.5, 6.5, 7.5])

def mean_absolute_error(y_true, y_pred):
    abs_diff = torch.abs(y_true - y_pred)
    mae = torch.mean(abs_diff)
    return mae

mae = mean_absolute_error(act_vals, pred_vals)
```

### Mean Squared Error (MSE)
It measures the average squared difference between the predicted and actual values, thus emphasizing larger errors. MSE is particularly useful in applications where the goal is to minimize the impact of outliers or when the error distribution is assumed to be Gaussian.

$$MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y_i})^2$$

- What it shows:
MSE measures the average squared difference between the actual and predicted values, penalizing larger errors more heavily than smaller ones.

- When to use:
Use MSE when you want to place a higher emphasis on larger errors.

- When not to use:
Avoid using MSE if you need an easily interpretable metric or if **your dataset has a lot of outliers, as it can be sensitive to them.**
```python
import torch

act_vals = torch.tensor([2, 4, 6, 8])
pred_vals = torch.tensor([2.5, 4.5, 6.5, 8.5])

def mean_squared_error(y_true, y_pred):
    diff = (y_true - y_pred) ** 2
    mse = torch.mean(diff)
    return diff

mse = mean_squared_error(act_vals, pred_vals)
print(f"mse: {mse:.2f}")
```
### Root Mean Squared Error (RMSE)
Root Mean Squared Error (RMSE) has the same unit as the target variable, making it more interpretable and easier to relate to the problem context than MSE.

$$RMSE = \sqrt{MSE}$$

- **When to use**:
Use RMSE to penalize larger errors and obtain a metric with the same unit as the target variable.

- **When not to use**:
Avoid using RMSE if you need an interpretable metric or if your dataset has a lot of outliers.

### R-Squared
R Squared (R^2), also known as the coefficient of determination, measures the proportion of the total variation in the target variable explained by the model's predictions.

R^2 ranges from 0 to 1, with higher values indicating a better model fit.

- **What it shows**:
R-squared measures the proportion of the variance in the dependent variable that the model's independent variables can explain.

- **When to use**:
Use R-squared when you want to understand how well your model is explaining the variation in the target variable compared to a simple average.

- **When not to use**:
Avoid using it if your model has a large number of independent variables or if it is sensitive to outliers.

```python
import torch

# Create tensors for actual and predicted values
actual_values = torch.tensor([2.0, 4.0, 6.0, 8.0])
predicted_values = torch.tensor([2.5, 3.5, 6.5, 7.5])

def r_squared_error(y_true, y_pred):
    # Calculate the mean of the actual values
    y_mean = torch.mean(y_true)
    
    # Calculate the sum of squares (numerator)
    residual_sum_of_squares = torch.sum((y_true - y_pred) ** 2)
    
    # Calculate the total sum of squares (denominator)
    total_sum_of_squares = torch.sum((y_true - y_mean) ** 2)
    
    # Calculate R² using the formula
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    
    return r_squared

# Calculate R²
r_squared = r_squared_error(actual_values, predicted_values)
print(f"R Squared Error: {r_squared:.2f}")
```
## Top classification metrics
### Accuraccy
Accuracy is a fundamental evaluation metric for assessing the overall performance of a classification model.

$$Accuraccy = \frac{TP + TN}{TP + FP +  TN + FN}$$

**What it shows**:
Accuracy measures the proportion of correct predictions made by the model out of all predictions.

**When to use**:
Accuracy is useful when the class distribution is balanced, and false positives and negatives have equal importance.

**When not to use**:
If the dataset is imbalanced or the cost of false positives and negatives differs, accuracy may not be an appropriate metric.

### Confusion Matrix
A confusion matrix, also known as an error matrix, is a tool used to evaluate the performance of classification models in machine learning and statistics. It presents a summary of the predictions made by a classifier compared to the actual class labels, allowing for a detailed analysis of the classifier's performance across different classes.

It helps identify misclassification patterns and calculate various evaluation metrics such as precision, recall, F1-score, and accuracy. By analyzing the confusion matrix, you can diagnose the model's strengths and weaknesses and improve its performance.

TP: True Positives - The number of patients with the disease correctly predicted as "yes."

TN: True Negatives - The number of patients without the disease was correctly predicted as "no."

FP: False Positives - The number of patients who don't have the disease but were incorrectly predicted as "yes."

FN: False Negatives - The number of patients who have the disease but were incorrectly predicted as "no."

### Precision and Recall

Precision and recall are essential evaluation metrics in machine learning for understanding the trade-off between false positives and false negatives. 

$$ Precision = \frac{TP}{TP + FP}$$
$$ Recall = \frac{TP}{TP+FN}$$

Precision (P) is the proportion of true positive predictions among all positive predictions. It is a measure of how accurate the positive predictions are.

Recall (R), also known as sensitivity or true positive rate (TPR), is the proportion of true positive predictions among all actual positive instances. It measures the classifier's ability to identify positive instances correctly. 

A high precision means the model has fewer false positives, while a high recall means fewer false negatives. Depending on the specific problem you're trying to solve, you might prioritize one of these metrics over the other.

**What they show**
Precision measures the proportion of true positive predictions among all positive predictions, while recall measures the proportion of true positive predictions among all actual positive instances.

**When to use**
Precision and recall are useful when the class distribution is imbalanced or when the cost of false positives and false negatives is different.

**When not to use**
Accuracy might be more appropriate if the dataset is balanced and the costs of false positives and negatives are equal.

```python
import torch

def precision_recall(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Input tensors must have the same shape"
    
    # Convert predictions to binary (0 or 1) by applying a threshold (0.5 in this case)
    y_pred_binary = (y_pred >= 0.5).float()
    
    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = torch.sum(y_true * y_pred_binary)
    FP = torch.sum((1 - y_true) * y_pred_binary)
    FN = torch.sum(y_true * (1 - y_pred_binary))
    
    # Calculate Precision and Recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    return precision, recall

# Example usage
y_true = torch.tensor([1, 0, 1, 1, 0, 1])
y_pred = torch.tensor([0.9, 0.3, 0.7, 0.1, 0.2, 0.8])

precision, recall = precision_recall(y_true, y_pred)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
```

### F1 Score

The F1-score is the harmonic mean of precision and recall, providing a metric that balances both measures. It is beneficial when dealing with imbalanced datasets, where one class is significantly more frequent than the other. The formula for the F1 score is:

$$F1 Score = \frac{2 \times Precision \times Recall}{Precision + Recall}$$

**What it shows**
The F1-score is the harmonic mean of precision and recall, providing a metric that considers false positives and false negatives.

**When to use**
The F1-score is useful when the class distribution is imbalanced or when the cost of false positives and false negatives is different.

**When not to use**
Accuracy might be more appropriate if the dataset is balanced and the costs of false positives and negatives are equal.


### Area Under the Receiver Operating Characteristic Curve (AU-ROC)
The AU-ROC is a popular evaluation metric for binary classification problems. It measures the model's ability to distinguish between positive and negative classes. The ROC curve plots the true positive rate (recall) against the false positive rate (1 - specificity) at various classification thresholds. The AU-ROC represents the area under the ROC curve, and a higher value indicates better model performance.

**What it shows**
AU-ROC represents the model's ability to discriminate between positive and negative classes. A higher AU-ROC value indicates better classification performance.

**When to use**
Use AU-ROC to compare the performance of different classification models, especially when the class distribution is imbalanced.

**When not to use**
Accuracy might be more appropriate if the dataset is balanced and the costs of false positives and negatives are equal.

```python
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

# Assuming you have the following PyTorch tensors:
# - `y_true`: a 1D tensor containing the true binary labels (0 or 1) for each sample
# - `y_pred`: a 1D tensor containing the predicted probabilities for the positive class

# Convert tensors to NumPy arrays
y_true_np = y_true.detach().cpu().numpy()
y_pred_np = y_pred.detach().cpu().numpy()

# Calculate AUROC score using scikit-learn
auroc = roc_auc_score(y_true_np, y_pred_np)

print(f"AUROC score: {auroc}")
```
## Other important metrics
### Intersection over Union (IoU)
Intersection over Union (IoU) is a popular evaluation metric in object detection and segmentation tasks. It measures the overlap between the predicted bounding box and the ground truth bounding box, providing an understanding of how well the model detects objects in images.

A higher IoU value indicates a better model performance, with 1.0 being the perfect score.

### Mean Average Precision (MAP)
Mean Average Precision (mAP) is another widely used performance metric in object detection and segmentation tasks. It is the average of the precision values calculated at different recall levels, providing a single value that captures the overall effectiveness of the model.

**What it shows**
Mean Average Precision (mAP) is a metric that computes the average precision (AP) for multiple object classes. It combines precision and recall, considering the presence of false positives and false negatives and their distribution across different confidence thresholds. The mAP score ranges from 0 (worst performance) to 1 (best performance).

**When to use**
Use mAP in object detection and segmentation tasks to evaluate the model's overall performance across all object classes—when there are multiple object classes, and you want a single metric to assess the model's performance across all classes.

**When not to use**
Avoid using mAP when you need a detailed analysis of the model's performance in specific classes, as it averages the performance across all classes. In such cases, analyze class-wise AP instead.
## Final words
- Different machine learning tasks require specific evaluation metrics. Regression tasks commonly use metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² (R-Squared). In contrast, classification tasks use metrics like Accuracy, Confusion Matrix, Precision and Recall, F1-score, and AU-ROC. Object detection and segmentation tasks rely on metrics like Intersection over Union (IoU) and Mean Average Precision (mAP).
- Choosing the right metric for a given project requires a clear understanding of the project goals and business objectives. Different metrics prioritize different aspects of model performance, and selecting the most relevant metric ensures that the model is optimized to meet the project's specific needs.
- Be aware of the strengths and weaknesses of each metric. For example, accuracy is a simple and intuitive metric for classification tasks but can be misleading for imbalanced datasets. Metrics like Precision, Recall, and F1-score may be more appropriate.
- Consistently use the chosen metric across various models and algorithms to effectively compare their performance. Doing so lets you identify the best-performing model that aligns with your project goals and business objectives.