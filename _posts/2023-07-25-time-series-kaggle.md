---
layout: post
title: "Time Series with Kaggle"
date: 2023-07-25 12:33:00 +1000
categories: deeplearning
---

## Linear Regression with Time Series
### What is a Time Series?
The basic object of forecasting is the time series, which is a set of observations recorded over time.
### Time-step features

There are two kinds of features unique to time series: **time-step features and lag features.**

**Time-step features** are features we can **derive directly from the time index**. The most basic time-step feature is the time dummy, which counts off time steps in the series from beginning to end.

To make a **lag feature** we **shift the observations of the target series** so that they appear to have **occured later in time.** Here we've created a 1-step lag feature, though shifting by multiple steps is possible too.

## Trend
### What is Trend?

The trend component of a time series represents a persistent, long-term change in the mean of the series. The trend is the slowest-moving part of a series, the part representing the largest time scale of importance. In a time series of product sales, an increasing trend might be the effect of a market expansion as more people become aware of the product year by year.

### Moving Average Plots
To see what kind of trend a time series might have, we can use a **moving average plot**. To compute a moving average of a time series, we compute the average of the values within a sliding window of some defined width. Each point on the graph represents the average of all the values in the series that fall within the window on either side. The idea is to smooth out any short-term fluctuations in the series so that only long-term changes remain.

### Engineering Trend
Once we've identified the shape of the trend, we can attempt to model it using a time-step feature.

In case, we can use `DeterministicProcess` to create a feature set for a cubic trend model and also forecast.

For example:
```python
y = data[['time', 'target']]
# If the trend is complicated, you can increase the order to get better fit.
dp = DeterministicProcess(index=df.index, order=3)

X = dp.in_sample()
X_fore = dp.out_of_sample(steps=90)

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax = y.plot(**plot_params, alpha=0.5, title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, linewidth=3, label="Trend", color='C0')
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color='C3')
ax.legend()
```
## Seasonality
### What is Seasonality?
We say that a time series exhibits seasonality whenever there is a regular, periodic change in the mean of the series. Seasonal changes generally follow the clock and calendar -- repetitions over a day, a week, or a year are common. Seasonality is often driven by the cycles of the natural world over days and years or by conventions of social behavior surrounding dates and times.

There are two kinds of features that model seasonality. The first kind, indicators, is best for a season with few observations, like a weekly season of daily observations. The second kind, Fourier features, is best for a season with many observations, like an annual season of daily observations.

### Seasonal Plots and Seasonal Indicators
Just like we used a moving average plot to discover the trend in a series, we can use a seasonal plot to discover seasonal patterns.

**Seasonal indicators** are **binary** features that represent seasonal differences in the level of a time series. Seasonal indicators are what you get if you treat a **seasonal period** as a **categorical feature** and apply one-hot encoding.

**Fourier features** try to capture the overall shape of the seasonal curve with **just a few features.**

**Fourier features** are **pairs** of **sine and cosine curves**, one pair for each **potential frequency** in the season starting with the longest. Fourier pairs modeling annual seasonality would have frequencies: **once per year, twice per year, three times per year, and so on.**

How many **Fourier pairs** should we actually include in our feature set? We can answer this question with the **periodogram**. The **periodogram** tells you the strength of the frequencies in a time series. Specifically, the value on the y-axis of the graph is (a ** 2 + b ** 2) / 2, where a and b are the coefficients of the sine and cosine at that frequency

