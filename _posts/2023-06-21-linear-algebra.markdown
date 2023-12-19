---
layout: post
title: "Linear Algebra for AI"
date: 2023-06-21 17:35:00 +1000
categories: deeplearning
---

## 1. Scalars, Vectors, Matrices and Tensors

The study of linear algebra involves several types of mathematical objects:
- **Scalars**: A scalar is just a single number, in contrast to most of the other
objects studied in linear algebra, which are usually arrays of multiple numbers.
- **Vectors**: A vector is an array of numbers. The numbers are arranged in
order. We can identify each individual number by its index in that ordering.
- **Matrices**: : A matrix is a 2-D array of numbers, so each element is identified
by two indices instead of just one.
- **Tensors**: In some cases we will need an array with more than two axes. In the general case, an array of numbers arranged on a regular grid with a
variable number of axes is known as a tensor

One important operation on matrices is the transpose. The transpose of a
matrix is the mirror image of the matrix across a diagonal line, called the main
diagonal, running down and to the right, starting from its upper left corner.
We denote the transpose of a
matrix A as $A^T$, and it is defined such that:
$$(A^T)_{i,j}=A_{j,i}$$

In the context of deep learning, we also use some less conventional notation. We
allow the addition of a matrix and a vector, yielding another matrix: $C = A + b$, where $C_{i,j} = A_{i,j} + b_j$ . In other words, the vector b is added to each row of the
matrix. This shorthand eliminates the need to define a matrix with b copied into
each row before doing the addition. This implicit copying of b to many locations
is called **broadcasting**.
## 2. Multiplying Matrices and Vectors
One of the most important operations involving matrices is multiplication of two
matrices. The matrix product of matrices A and B is a third matrix C. In
order for this product to be defined, A must have the same number of columns as
B has rows. If A is of shape m × n and B is of shape n × p, then C is of shape
m × p. We can write the matrix product just by placing two or more matrices
together, for example, 
$$C = AB$$ 
Note that the standard product of two matrices is not just a matrix containing
the product of the individual elements. Such an operation exists and is called the
element-wise product, or Hadamard product, and is denoted as $A \odot B$.

## 3. Identity and Inverse Matrices
To describe matrix inversion, we first need to define the concept of an identity
matrix. An identity matrix is a matrix that does not change any vector when we
multiply that vector by that matrix. We denote the identity matrix that preserves
n-dimensional vectors as $I_n$.
$$\forall x \in \R^n, I_nx = x$$
The **matrix inverse** of A is denoted as $A^{-1}$, and it is defined as the matrix such that
$$A^{-1}A = I_n$$
## 4. Norms
Sometimes we need to measure the size of a vector. In machine learning, we usually
measure the size of vectors using a function called a norm. Formally, the $L^p$ norm
is given by
$$||x||_p = (\sum_i {|x_i|^p})^-$$
The L2 norm, with p = 2, is known as the **Euclidean norm**, which is simply
the Euclidean distance from the origin to the point identified by x.

The L1 norm is commonly used in machine learning when the difference between
zero and nonzero elements is very important.

Sometimes we may also wish to measure the size of a **matrix**. In the context
of deep learning, the most common way to do this is with the otherwise obscure
**Frobenius norm**:
$$||A||_F = \sqrt{\sum_{i,j}A_{i,j}^2}$$
## 5. Special Kinds of Matrices and Vectors
1. **Diagonal matrices** consist mostly of zeros and have nonzero entries only along
the main diagonal.
2. A **symmetric matrix** is any matrix that is equal to its own transpose:
$A = A^T$.
3. A **unit vector** is a vector with unit norm: $||x||_2=1$.
4. An **orthogonal matrix** is a square matrix whose rows are mutually orthonormal and whose columns are mutually orthonormal:
$$A^T A = A A^T = I.$$
This implies that
$$A^{-1} = A^T$$
## 6. Eigendecomposition
One of the most widely used kinds of matrix decomposition is called eigendecomposition, in which we decompose a matrix into a set of eigenvectors and
eigenvalues.
An eigenvector of a square matrix A is a nonzero vector v such that multi- plication by A alters only the scale of v:
$$Av = \lambda v$$
The scalar λ is known as the eigenvalue corresponding to this eigenvector.
Suppose that a matrix A has n linearly independent eigenvectors {v
(1)
, . . . , v
(n)} with corresponding eigenvalues {λ1, . . . , λn}. We may concatenate all the
eigenvectors to form a matrix V with one eigenvector per column: V = [v
(1)
, . . . , v
(n)
]. Likewise, we can concatenate the eigenvalues to form a vector λ = [λ1 , . . . ,λn]T. The eigendecomposition of A is then given by
$$A = Vdial(\lambda)V^{-1}$$
We have seen that constructing matrices with specific eigenvalues and eigen- vectors enables us to stretch space in desired directions. Yet we often want to
decompose matrices into their eigenvalues and eigenvectors. Doing so can help
us analyze certain properties of the matrix, much as decomposing an integer into
its prime factors can help us understand the behavior of that integer.

Specifically, every real symmetric
matrix can be decomposed into an expression using only real-valued eigenvectors
and eigenvalues:

$$A = Q Q^T$$

where Q is an orthogonal matrix composed of eigenvectors of A, and is a
diagonal matrix. The eigenvalue Λi,i is associated with the eigenvector in column i
of Q, denoted as Q:,i. Because Q is an orthogonal matrix, we can think of A as
scaling space by λi in direction v(i).
## 7. Singular Value Decomposition
The SVD enables us to
discover some of the same kind of information as the eigendecomposition reveals;
however, the SVD is more generally applicable. Every real matrix has a singular
value decomposition, but the same is not true of the eigenvalue decomposition.
For example, if a matrix is not square, the eigendecomposition is not defined, and
we must use a singular value decomposition instead.

The singular value decomposition is similar, except this time we will write A
as a product of three matrices:

$$A = UDV^T$$

Suppose that A is an m ×n matrix. Then U is defined to be an m ×m matrix, D to be an m × n matrix, and V to be an n × n matrix.
## 8. The Moore-Penrose Pseudoinverse
Practical algorithms for computing the pseudoinverse are based not on this defini- tion, but rather on the formula

$$A^+ = V D^+U^T$$

where U, D and V are the singular value decomposition ofA, and the pseudoinverse
D+ of a diagonal matrix D is obtained by taking the reciprocal of its nonzero
elements then taking the transpose of the resulting matrix.
## 9. The Trace Operator and Determinant
The trace operator gives the sum of all the diagonal entries of a matrix:

$$Tr(A) = \sum_{i}A_{i,j}$$

The trace operator is useful for a variety of reasons. Some operations that are
difficult to specify without resorting to summation notation can be specified using
matrix products and the trace operator. For example, the trace operator provides
an alternative way of writing the Frobenius norm of a matrix:
$$||A||_F = \sqrt{Tr(AA^T)}$$

The determinant of a square matrix, denoted det(A), is a function that maps
matrices to real scalars. The determinant is equal to the product of all the
eigenvalues of the matrix.