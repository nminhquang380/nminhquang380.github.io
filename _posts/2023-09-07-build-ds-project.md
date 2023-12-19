---
layout: post
title: "Get lost when trying to build a Data Science Project"
date: 2023-09-07 20:35:00 +1000
categories: datascience 
---
### Disclaimer
Currently I am lost in Data Science where there are a lot of concepts, technique and skills I should learn. Additionally, next year I will graduate so I should prepare something to get a job, especially in Australia which have a roughly competitive work market. Thus I try to build some DS projects, however, I got some issues, and this article helps me a lot, let me share it!

----
## How to build a data science project from scratch

There are many online courses about data science and machine learning that will guide you through a theory and provide you with some code examples and an analysis of very clean data.

However, in order to start practising data science, it is better if you challenge a real-life problem. Digging into the data in order to find deeper insights. Carrying out feature engineering using additional sources of data and building stand-alone machine learning pipelines.

This blog post will guide you through the main steps of building a data science project from scratch. It is based on a real-life problem — what are the main drivers of rental prices in Berlin? It will provide an analysis of this situation. It will also highlight the common mistake beginners tend to make when it comes to machine learning.

These are the steps that will be discussed in detail:

- finding a topic
- extracting data from the web and cleaning it
- gaining deeper insights
- engineering of features using external APIs
- common mistakes while carrying out machine learning
- feature importance: finding the drivers of rental prices
- building machine learning models.

### Finding a topic
There are many problems that can be solved by analyzing data, but it is always better to find a problem that you are interested in and that will motivate you. While searching for a topic, you should definitely concentrate on your preferences and interests.

For instance, if you are interested in healthcare systems, there are many angles from which you could challenge the data provided on that topic. “Exploring the ChestXray14 dataset: problems” is an example of how to question the quality of medical data. Another example — if you are interested in music, you could try to predict the genre of the song from its audio.

However, I suggest not only to concentrate on your interests but also to listen to what people around you are talking about. What bothers them? What are they complaining about? This can be another good source of ideas for a data science project. In those cases where people are still complaining about it, this may mean that the problem wasn’t solved properly the first time around. Thus, if you challenge it with data, you could provide an even better solution and have an impact in how this topic is perceived.

This may all sound a bit too abstract, so lets find out how I came up with the idea to analyze Berlin rental prices.

> “If I had known that the rental prices were so high here, I would have negotiated for a higher salary.” 

This is just one of the things I heard from people who had recently moved to Berlin for work. Most newcomers complained that they hadn’t imagined Berlin to be so expensive, and that there were no statistics about possible price ranges of the apartment. If they had known this it beforehand, they could have asked for a higher salary during the job application process or could have considered other options.

I googled, checked several rental apartment websites, and asked several people, but could not find any plausible statistics or visualizations of the current market prices. And this was how I came up with the idea of this analysis.

I wanted to gather the data, build an interactive dashboard where you could select different options such as a 40m2 apartment situated in Berlin Mitte with a balcony and equipped kitchen, and it would show you the price ranges. This, alone, would help people understand apartment prices in Berlin. Also, by applying machine learning, I would be able to identify the drivers of the rental prices and practice with different machine learning algorithms.

### Extracting data from the web and cleaning it
#### Getting the data
Now that you have an idea about your data science project, you can start looking for the data. There are tons of amazing data repositories, such as Kaggle, UCI ML Repository or dataset search engines, and websites containing academic papers with datasets. Alternatively, you could use web scraping.

But be cautious — old data is everywhere. When I was searching for the information about the rental prices in Berlin, I found many visualizations but they were old, or without any year specified.

For some statistics, they even had a note saying that this price would only be for a 2 room apartment of 50 m2 without furniture. But what if I am searching for a smaller apartment with a furnished kitchen?

As I could find only old data, I decided to [web scrape](https://hackernoon.com/web-scraping-tutorial-with-python-tips-and-tricks-db070e70e071) the websites that offered rental apartments. Web scraping is a technique used to extract data from websites through an automated process.

Here are the main findings:

- Before scraping, check if there is a public API available
- Be kind! Don’t overload the website by sending hundreds of requests per second
- Save the date when the extraction took place. It will be explained why this is important.

#### Data cleaning
Once you starting getting the data, it is very important to have a look at it as early as possible in order to find any possible issues.

While web scraping rental data, I included some small checks such as the number of missing values for all features. Web-masters could change the HTML of the website, which would result in my program not getting the data anymore.

Once I had ensured that all technical aspects of web scraping were covered, I thought the data would almost be ideal. However, I ended up cleaning the data for around a week because of not so obvious duplicates.

Once you starting getting the data, it is very important to have a look at it as early as possible in order to find any possible issues. For instance, if you web scrape, you could have missed some important fields. If you use a comma separator while saving data into a file, and one of the fields also contains commas, you can end up having files which are not separated very well.

There were several sources of duplicates:

- Duplicated apartments because they had been online for a while
- Agencies had input errors, for example the rental price or the storey of the apartment. They would correct them after a while, or would publish a completely new ad with corrected values and additional description modifications
- Some prices were changed (increased and decreased) after a month for the same apartment

While the duplicates from the first case were easy to identify by their ID, the duplicates from the second case were very complicated. The reason is that an agency could slightly change a description, modify the wrong price, and publish it as a new ad so that the ID would also be new.

I had to come up with many logic-based rules to filter out the old versions of the ads. Once I was able to identify that these apartments would be the actual duplicates but with slight modifications, I could sort them by the extraction date, taking the latest one as the most recent.

Additionally, some agencies would increase or decrease the price for the same apartment after a month. I was told that if nobody wanted this apartment, the price would decrease. Conversely, I was told that, if there were so many requests for it, that the agencies increased the price. These sounds like good explanations.

#### Gaining deeper insights
Now that we have everything ready, we can start analyzing the data. I know data scientists love `seaborn` and `ggplot2`, as well as many static visualizations from which they can derive some insights.

However, interactive dashboards can help you and other stakeholders to find useful insights. There are many amazing easy-to-use tools for that, such as `Tableau` and `Microstrategy`.

It took me less than 30 minutes to create an interactive dashboard where one can select all the important components and see how the price would change.

A fairly simple dashboard could already provide insights into the prices in Berlin for newcomers and could be a good user driver for a rental apartment website.

Already from this data visualization you can see that the price distribution of 2.5 rooms falls into the distribution of 2 room apartment. The reason for this is that most of the 2.5 room apartments aren’t situated in the center of the city which, of course, reduces the price.

This data was gathered in winter 2017/18 and it will also get outdated. However, my point is that the rental websites could frequently update their statistics and visualizations to provide more transparency to this question.

#### Engineering of features using external APIs
Visualization helps you to identify important attributes, or “features,” that could be used by these machine learning algorithms. If the features you use are very uninformative, any algorithm will produce bad predictions. With very strong features, even a very simple algorithm can produce pretty decent results.

In the rental price project, price is a continuous variable, so it is a typical regression problem. Taking all extracted information, I collected the following features in order to be able to predict a rental price.

However, there was one feature that was problematic, namely the address. There were 6.6K apartments and around 4.4K unique addresses of different granularity. There were around 200 unique postcodes which could be converted into the dummy variables but then very precious information of a particular location would be lost.

**What do you do when you are given a new address?** </br>
You either google where it is or how to get there.

By using an external API following the four additional features given, the apartment’s address could be calculated:
1. duration of a train trip to the S-Bahn Friedrichstrasse (central station)
2. distance to U-Bahn Stadtmitte (city center) by car

3. duration of a walking trip to the nearest metro station

4. number of metro stations within one kilometer from the apartment

These four features boosted the performance significantly.

**Common mistakes when carrying out machine learning and data science**
After scraping or getting the data, there are many steps to accomplish before applying a machine learning model.

You need to visualize each of the variables to see distributions, find the outliers, and understand why there are such outliers.

What can you do with missing values in certain features?

What would be the best way to convert categorical features into numerical ones?

There are many such questions, but I will give some details on the ones where the majority of beginners encounter mistakes.

#### 1. Visualization
   
Firstly, you should visualize the distribution of the continuous features to get a feeling if there are many outliers, what the distribution would be, and if it makes sense.

There are many ways to visualize it, for example box plots, histograms, cumulative distribution functions, and violin plots. However, one should pick the plot that will give the most information about the data.

To see the distribution (if it is normal, or bimodal), the histograms will be the most helpful. Although histograms are a good starting point, the box plots might be superior in identifying the number of outliers and seeing where the median quartiles lie.

Based on the plots, the most interesting question would be: do you see what you expected to see? Answering this question will help you either in finding insights or finding bugs in the data.

To get inspired and understand what plot will give the most value, I frequently referred to the Python’s `seaborn` gallery. Another good source of inspiration for the visualization and finding insights are kernels on `Kaggle`. Here is my `kaggle` kernel of the in-depth visualization of the titanic dataset.

In the context of rental prices, I plotted the histograms of each continuous feature and expected to see a long right tail in the distribution of the rent without bills and total area.

Box plots helped me see the number of outliers for each of the features. In fact, most of the outliers apartments based on the rent without bills were either the ateliers for the small shops with more than 200m2 or the student dormitories with very low rent.

#### 2. Do I impute the values based on the whole dataset?

Sometimes there will be missing values, due to various reasons. If we exclude every observation with at least one missing value, we can end up with a very reduced dataset.

There are many ways of imputing the values, mean, or median. It is up to you how to do it but make sure to calculate the imputation statistics only on the training data to avoid data leakage of your test set.

In the rental data, I also extracted a description of the apartment. Whenever the quality, condition, or type of apartment was missing, I would impute it from the description if the description contained this information.

#### 3. How do I transform categorical variables?
Some algorithms, depending on the implementation, wouldn’t work directly with the categorical data, so one would need to somehow transform them into numerical values.

There are many ways of transforming categorical variables into numerical features, such as Label Encoder, One Hot Encoding, bin encoding, and hashing encoding. However, most people use the Label Encoding incorrectly when the One Hot Encoding should have been used instead.

Assume, in our rental data, that we have an apartment-type column with the following values: [ground floor, loft, maisonette, loft, loft, ground floor]. LabelEncoder can turn this into [3,2,1,2,2,1], introducing ordinality, which means that ground_floor >loft > maisonette. For some algorithms like decision trees, and its deviations, this type of encoding for this feature would be fine, but applying regressions and SVM might not make that much sense.

#### 4. Do I need to standardize variables?
Standardization brings all continuous variables to the same scale, meaning if one variable has values from 1K to 1M and another from 0.1 to 1, after standardization they will have the same range.

L1 or L2 regularizations are the common way of reducing overfitting and can be used within many regression algorithms. However, it is important to apply feature standardization before L1 or L2.

The rental price is in Euros so the fitted coefficient would be approximately 100 times larger than the fitted coefficient if the price was in cents. L1 and L2 penalize the larger coefficients more, meaning it will penalize the features in smaller scales more. To prevent this, the features should be standardized before applying L1 or L2.

Another reason to standardize is that if you or the your algorithm use gradient descent, gradient descent converges much faster with feature scaling.

#### 5. Do I need to derive the logarithm of the target variable?
It took me a while to understand that there is no universal answer.

It depends on many factors:

whether you want fractional or absolute error
which algorithm you use
what residual plots and changes in the metrics tell you
In regression, firstly pay attention to the residual plots and the metric. Sometimes the logarithmization of the target variable leads to a better model and the results of the model would still be easy to understand. However, there are still other transformations that could be of interest, such as to taking the square root.

There are many answers on Stack Overflow regarding this question, and I think Residual Plots and RMSE on raw and log target variable explains it very well.

For the rental data, I derived the logarithm of the price as the residual plots looked a bit better.

#### 6. Some more important stuff
Some algorithms, such as regressions, will suffer from collinearities in the data because the coefficients become very unstable (more math). SVM might or might not suffer from collinearity due to the choice of kernel.

Decision-based algorithms will not suffer from multicollinearity as they could use features interchangeably in different trees without it affecting the performance. However, the interpretation of feature importance then gets more difficult as the correlated variable may not appear to be as important as it is.

### Machine Learning
After you have familiarized yourself with data and cleaned out the outliers, it is the perfect time to get the hang of machine learning. There are many algorithms you could use for this supervised machine learning.

There were three different algorithms I wanted to explore, comparing characterstics such as performance differences and speed. These three were gradient boosted trees with different implementations (XGBoost and LightGMB), Random Forest (FR, scikit-learn) and 3-layer Neuronal Networks (NN, Tensorflow). I selected RMSLE (root mean squared logarithm error) to be the metric for the optimization of the process. I used RMSLE because I derived the logarithm of the target variable.

XGBoost and LigthGBM performed comparably, RF slightly worse, whereas NN was the worst.

Decision tree-based algorithms are very good at interpreting features. For example, they produce a feature importance score.

#### Feature importance: finding the drivers of the rental price

After fitting a decision tree-based model, you can see what features are the most valuable for the price prediction.

Feature importance provides a score that indicates how informative each feature was in the construction of the decision trees within the model. One of the ways to calculate this score is to count how many times a feature is used to split the data across all trees. This score can be computed in different ways.

Feature importance can reveal other insights about the main price drivers.

For the rental price prediction, it isn’t surprising that total area is the most important driver of the price. Interestingly, some features that were engineered with external API are also in the top most important features.

However, as mentioned in “Interpretable Machine Learning with XGBoost”, there can be inconsistencies in feature importance depending on the attribution option. The author of the linked blogpost, and SHAP NIPS paper, proposes a new way of calculating feature importance that will be both accurate and consistent. This uses the shap Python library. SHAP values represent the responsibility of a feature for a change in the model output.

#### Ensemble averaging
After playing around with different models and comparing performance, you could just combine the results of each of the model and build an ensemble!

Bagging is the machine learning ensemble model that utilizes the predictions of several algorithms to calculate the final aggregated predictions. It is designed to prevent overfitting and reduces the variance of the algorithms.

As I already had predictions from the above mentioned algorithms, I combined all four models in all possible ways and picked the seven best single and ensemble models based on the RMSLE of the validation set.

The ensemble of three decision-tree based algorithms performed the best compared to each single model.

You could also produce a weighted ensemble, assigning more weight to a better single model. The reasoning behind it is that other models could overrule the best model only if they collectively agree on an alternative.

In reality, one would never know if an averaged ensemble would be better than the single model without just trying it out.

#### Stacked models
An averaged or weighted ensemble is not the only way to combine the predictions of different models. You could also stack the models in very different ways!

The idea behind stacked models is to create several base models and a meta model on top of the results from the base models in order to produce final predictions. However, it is not so obvious how to train the meta model because it can be biased towards the best of the base models. A very good explanation of how to do it correctly can be found in the post “Stacking models for improved predictions”.

For the rental price case, stacked models didn’t improve the RMSLE at all — they even increased the metrics. There might be several reasons for this — either I coded it incorrectly ;) or there was just too much noise introduced by stacking.

If you want to explore more of the ensemble and stacked model articles, the Kaggle Ensemble Guide explains many different kinds of ensembling with the performance comparison and referrals on how such stacked models got to the top of Kaggle’s competitions.

### Final thoughts
- listen to what people talk about around you; their complaining can serve as a good starting point for solving something big
- let people find their own insights by providing interactive dashboards
don’t restrict yourself to common feature engineering as multiplying two variables
- Try to find additional sources of data or explanations
- try out ensembles and stacked models as those methods could improve the performance
  
And please, provide the date of the data you display!

### Reference

Kokatjuhha, J. (2020, March 19). How to build a data science project from scratch. We’ve Moved to FreeCodeCamp.org/News. https://medium.com/free-code-camp/how-to-build-a-data-science-project-from-scratch-dc4f096a62a1

‌