---
layout: post
title: "Autoencoders"
date: 2023-06-20 3:35:00 +1000
categories: deeplearning 
---

An autoencoder is a neural network that is trained to attempt to copy its input
to its output. Internally, it has a hidden layer h that describes a code used to
represent the input. The network may be viewed as consisting of two parts: an
encoder function h = f (x) and a decoder that produces a reconstruction r = g(h). If an autoencoder succeeds in simply
learning to set g(f (x)) = x everywhere, then it is not especially useful. Instead, autoencoders are designed to be unable to learn to copy perfectly. Usually they are
restricted in ways that allow them to copy only approximately, and to copy only
input that resembles the training data. Because the model is forced to prioritize
which aspects of the input should be copied, it often learns useful properties of the
data.

## 1. Undercomplete Autoencoders


Copying the input to the output may sound useless, but we are typically not
interested in the output of the decoder. Instead, we hope that training the
autoencoder to perform the input copying task will result in h taking on useful
properties. One way to obtain useful features from the autoencoder is to constrain h to
have a smaller dimension than x. An autoencoder whose code dimension is less
than the input dimension is called undercomplete. Learning an undercomplete
representation forces the autoencoder to capture the most salient features of the
training data.
The learning process is described simply as minimizing a loss function:
$$L(x, g(f(x)))$$
where L is a loss function penalizing g(f (x)) for being dissimilar from x, such as
the mean squared error.

Unfortunately, if the encoder and decoder are allowed too much capacity, the autoencoder
can learn to perform the copying task without extracting useful information about
the distribution of the data. Theoretically, one could imagine that an autoencoder
with a one-dimensional code but a very powerful nonlinear encoder could learn to
represent each training example x
(i) with the code i. The decoder could learn to
map these integer indices back to the values of specific training examples. This
specific scenario does not occur in practice, but it illustrates clearly that an autoen- coder trained to perform the copying task can fail to learn anything useful about
the dataset if the capacity of the autoencoder is allowed to become too great.

## 2. Regularized Autoencoders

Ideally, one could train any architecture of autoencoder successfully, choosing
the code dimension and the capacity of the encoder and decoder based on the
complexity of distribution to be modeled. Regularized autoencoders provide the
ability to do so. Rather than limiting the model capacity by keeping the encoder
and decoder shallow and the code size small, regularized autoencoders use a loss
function that encourages the model to have other properties besides the ability
to copy its input to its output. These other properties include sparsity of the
representation, smallness of the derivative of the representation, and robustness
to noise or to missing inputs. A regularized autoencoder can be nonlinear and
overcomplete but still learn something useful about the data distribution, even if
the model capacity is great enough to learn a trivial identity function.

