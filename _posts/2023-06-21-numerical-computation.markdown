---
layout: post
title: "Numerical Computation"
date: 2023-06-21 19:13:00 +1000
categories: deeplearning
---

## 1. Overflow and Underflow
The fundamental difficulty in performing continuous math on a digital computer
is that we need to represent infinitely many real numbers with a finite number
of bit patterns. This means that for almost all real numbers, we incur some
approximation error when we represent the number in the computer. In many
cases, this is just rounding error. Rounding error is problematic, especially when
it compounds across many operations, and can cause algorithms that work in
theory to fail in practice if they are not designed to minimize the accumulation of
rounding error.

One form of rounding error that is particularly devastating is underflow. Underflow occurs when numbers near zero are rounded to zero. Many functions
behave qualitatively differently when their argument is zero rather than a small
positive number.

Another highly damaging form of numerical error is overflow. Overflow occurs
when numbers with large magnitude are approximated as ∞ or −∞. Further
arithmetic will usually change these infinite values into not-a-number values.

One example of a function that must be stabilized against underflow and
overflow is the **softmax function**.

## 2. Poor Conditioning
Conditioning refers to how rapidly a function changes with respect to small changes
in its inputs. Functions that change rapidly when their inputs are perturbed slightly
can be problematic for scientific computation because rounding errors in the inputs
can result in large changes in the output.

## 3. Gradient-Based Optimization
Most deep learning algorithms involve optimization of some sort. Optimization
refers to the task of either minimizing or maximizing some function f(x) by altering
x. We usually phrase most optimization problems in terms of minimizing f (x). Maximization may be accomplished via a minimization algorithm by minimizing −f(x).

## 4. Constrained Optimization
Sometimes we wish not only to maximize or minimize a function f(x) over all
possible values of x. Instead we may wish to find the maximal or minimal
value of f (x) for values of x in some set S. This is known as constrained
optimization. Points x that lie within the set S are called feasible points in
constrained optimization terminology. We often wish to find a solution that is small in some sense. A common
approach in such situations is to impose a norm constraint, such as ||x|| ≤ 1. One simple approach to constrained optimization is simply to modify gradient
descent taking the constraint into account. If we use a small constant step size  , we can make gradient descent steps, then project the result back into S. If we use
a line search, we can search only over step sizes  that yield new x points that are
feasible, or we can project each point on the line back into the constraint region. When possible, this method can be made more efficient by projecting the gradient
into the tangent space of the feasible region before taking the step or beginning
the line search.
