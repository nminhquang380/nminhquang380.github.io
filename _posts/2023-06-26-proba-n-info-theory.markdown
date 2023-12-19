---
layout: post
title: "Probability and Information Theory"
date: 2023-06-26 19:00:00 +1000
categories: deeplearning
---
## 1. Random Variables

A random variable is a variable that can take on different values randomly.

Random variables may be discrete or continuous. A discrete random variable
is one that has a finite or countably infinite number of states. Note that these
states are not necessarily the integers; they can also just be named states that
are not considered to have any numerical value. A continuous random variable is
associated with a real value.

## 2. Probability Distributions

A probability distribution is a description of **how likely a random variable or
set of random variables is to take on each of its possible states**. The way we
describe probability distributions depends on whether the variables are discrete or
continuous.

### 2.1. Discrete Variables and Probability Mass Functions

A probability distribution over discrete variables may be described using a probability mass function (PMF). We typically denote probability mass functions with
a capital P. Often we associate each random variable with a different probability
mass function and the reader must infer which PMF to use based on the identity
of the random variable, rather than on the name of the function; P(x) is usually
not the same as P(y).

The probability mass function maps from a state of a random variable to
the probability of that random variable taking on that state. The probability
that x = x is denoted as P(x), with a probability of 1 indicating that x = x is
certain and a probability of 0 indicating that x = x is impossible. Sometimes
to disambiguate which PMF to use, we write the name of the random variable
explicitly: P (x = x).

Probability mass functions can act on many variables at the same time. Such
a probability distribution over many variables is known as a joint probability
distribution. P(x = x, y = y ) denotes the probability that x = x and y = y
simultaneously. We may also write P(x, y) for brevity. 

### 2.2. Continuous Variables and Probability Density Functions

When working with continuous random variables, we describe probability distributions using a probability density function (PDF) rather than a probability
mass function. To be a probability density function, a function p must satisfy the
following properties:

- The domain of p must be the set of all possible states of x. 
- ∀x ∈ x, p(x) ≥ 0. Note that we do not require p(x) ≤ 1.
- $\int p(x)dx = 1.$ 

A probability density function p(x) does not give the probability of a specific
state directly; instead the probability of landing inside an infinitesimal region with
volume δx is given by p(x)δx.

## 3. Marginal Probability

Sometimes we know the probability distribution over a set of variables and we want
to know the probability distribution over just a subset of them. The probability
distribution over the subset is known as the marginal probability distribution. For example, suppose we have discrete random variables x and y, and we know
P(x, y). We can find P(x) with the sum rule:

$$\forall x \in X, P(X= x) = \sum_yP(X= x, Y = y).$$

For continuous variables, we need to use integration instead of summation:

$$p(x) = \int p(x,y)dy.$$

## 4. Conditional Probability
In many cases, we are interested in the probability of some event, given that some
other event has happened. This is called a conditional probability. We denote
the conditional probability that y = y given x = x as P(y = y | x = x). This
conditional probability can be computed with the formula

$$P(Y = y | X= x) = \frac{P(Y=y, X= x)}{P(X=x)}$$
The conditional probability is only defined when P(x = x) > 0. We cannot compute
the conditional probability conditioned on an event that never happens.

## 5. The Chain Rule of Conditional Probabilities
Any joint probability distribution over many random variables may be decomposed
into conditional distributions over only one variable.

This observation is known as the chain rule, or product rule, of probability.

For example, applying the definition twice, we get

$$P(a, b, c) = P(a | b, c)P(b, c)$$

$$P(b, c) = P(b | c)P(c)$$

$$P(a, b, c) = P(a | b, c)P(b | c)P(c).$$

## 6. Independence and Conditional Independence
 Two random variables x and y are independent if their probability distribution
can be expressed as a product of two factors, one involving only x and one involving
only y:

∀x ∈ x, y ∈ y, p(x = x, y = y) = p(x = x)p(y = y).

Two random variables x and y are conditionally independent given a random
variable z if the conditional probability distribution over x and y factorizes in this
way for every value of z:

∀x ∈ x, y ∈ y, z ∈ z, p(x = x, y = y | z = z) = p(x = x | z = z)p(y = y | z = z).

## 7. Expectation, Variance and Covariance
The **expectation**, or **expected value**, of some function f(x) with respect to a
probability distribution P(x) is the average, or mean value, that f takes on when
x is drawn from P. For discrete variables this can be computed with a summation:
$$E_{x~P}[f(x)] = \sum P(x)f(x)$$
while for continuous variables, it is computed with an integral:
$$E_{x~P}[f(x)] = \int p(x)f(x)dx$$

The **variance** gives a measure of how much the values of a function of a random
variable x vary as we sample different values of x from its probability distribution:
$$Var(f(x)) = E[(f(x)-E[f(x)])^2]$$
When the variance is low, the values of f (x) cluster near their expected value. The
square root of the variance is known as the **standard deviation**. 
## 8. Common Probability Distributions
### 8.1. Bernoulli Distribution
The Bernoulli distribution is a distribution over a single binary random variable. It is controlled by a single parameter φ ∈ [0, 1], which gives the probability of the
random variable being equal to 1.
### 8.2. Multinoulli Distribution
### 8.3. Gaussian Distribution
The most commonly used distribution over real numbers is the normal distribution, also known as the Gaussian distribution
### 8.4. Exponential and Laplace Distribution
In the context of deep learning, we often want to have a probability distribution
with a sharp point at x = 0. To accomplish this, we can use the exponential
distribution:

p(x; λ) = λ1x≥0 exp (−λx). 

## 9. Useful Properties of Common Functions
Certain functions arise often while working with probability distributions,
especially the probability distributions used in deep learning models.

One of these functions is the logistic sigmoid
## 10. Bayes' Rule
We often find ourselves in a situation where we know P(y | x) and need to know
P (x | y). Fortunately, if we also know P(x), we can compute the desired quantity
using Bayes’ rule:
$$P(x|y) = \frac{P(x)P(y|x)}{P(y)}$$
## 11. Information Theory
