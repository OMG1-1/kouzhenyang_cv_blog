# 矩阵

## 矩阵是什么？

矩阵是一种数学对象，可以看作是一个由数排成行和列的矩形阵列。我们通常使用大写字母表示矩阵，例如$\mathbf{A}$ 、$\mathbf{B}$、$\mathbf{C}$等。元素可以是任何数值、变量、表达式或函数。

## 矩阵的类型

按照矩阵的行列情况，我们可以将矩阵分为行矩阵、列矩阵、零矩阵以及$n$阶方阵等类型。具体来说，$m \times n$阶矩阵中，如果$m=1$，就被称为行矩阵，或者$n$维行向量；同样地，如果$n=1$，就被称为列矩阵，或者$m$维列向量。而所有元素都为 0 的$m\times n$阶矩阵被称为零矩阵。

### 单位矩阵

单位矩阵是一个$n \times m$矩阵，从左到右的对角线上的元素是 1，其余元素都为 0。

如：$\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$, $\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$

!!! note "单位矩阵在矩阵乘法中的作用相当于数字`1`"

    如果 $\mathbf{A}$ 是 $m \times n$ 矩阵，$\mathbf{I}$ 是单位矩阵，则 $\mathbf{AI} = \mathbf{A}, \mathbf{IA} = \mathbf{A}$,

    $$
    \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \times
    \begin{bmatrix} a & b \\ c & d \end{bmatrix}
    =
    \begin{bmatrix}
    1 \times a + 0 \times c & 1 \times b + 0 \times d
    \\
    0 \times a + 1 \times c & 0 \times b + 1 \times d
    \end{bmatrix}
    =
    \begin{bmatrix} a & b \\ c & d \end{bmatrix}
    $$

    $$
    \begin{bmatrix} a & b \\ c & d \end{bmatrix} \times
    \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
    =
    \begin{bmatrix}
    a \times 1 + b \times 0 & a \times 0 + b \times 1
    \\
    c \times 1 + d \times 0 & c \times 0 + d \times 1
    \end{bmatrix}
    =
    \begin{bmatrix} a & b \\ c & d \end{bmatrix}
    $$

### 逆矩阵 $\mathbf{A^{-1}}$

逆矩阵是矩阵理论的重要概念，如果一个 $n$ 阶方阵 $A$ 在相同的数域上存在另一个 $n$ 阶矩阵 $B$，使得：$AB=BA=E$ ，则我们称 $B$ 是 $$ 的逆矩阵，而 $A$ 被称为可逆矩阵。其中，$E$ 为单位矩阵。

然而，并非所有矩阵都存在逆矩阵。首先，矩阵必须是一个方阵。其次，矩阵的行列式不能为 0。

!!! note "逆矩阵运算"

    $\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$ 的逆矩阵：

    $$
    \mathbf{A^{-1}}=
    \cfrac{1}{\lvert \mathbf{A} \rvert}
    \begin{bmatrix}
    d & -b \\
    -c & a
    \end{bmatrix}
    $$

    其中，$\lvert \mathbf{A} \rvert$ 是二阶行列式[^1]

!!! question "求下面 $B$ 的逆矩阵"

    $\mathbf{B} =
    \begin{bmatrix}
    3 & -4 \\
    2 & -5
    \end{bmatrix}$ 的逆矩阵。

    $$
    \mathbf{B^{-1}}=
    \cfrac{1}{\lvert \mathbf{B} \rvert}
    \begin{bmatrix}
    -5 & 4 \\
    -2 & 3
    \end{bmatrix}
    =
    \cfrac{1}{(-5) \times 3 - (-4) \times 2}
    \begin{bmatrix}
    -5 & 4 \\
    -2 & 3
    \end{bmatrix}
    =
    -\cfrac{1}{7}
    \begin{bmatrix}
    -5 & 4 \\
    -2 & 3
    \end{bmatrix}
    =
    \begin{bmatrix}
    \frac{5}{7} & -\frac{4}{7} \\
    \frac{2}{7} & -\frac{3}{7}
    \end{bmatrix}
    $$

计算逆矩阵的方法有很多，比如初等行运算（高斯－若尔当）。
在我们介绍逆矩阵的计算方法之前，需要先明确一点，逆矩阵不等于矩阵转置。矩阵转置的操作是将一个矩阵行和列互换，在线性代数当中，矩阵 A 的转置记作$\mathbf{A^T}$，而 A 的逆矩阵记作$\mathbf{A^{-1}}$，虽然看起来比较相似，但二者是有本质区别的。

### 奇异矩阵

**当一个矩阵没有逆矩阵的时候，称该矩阵为奇异矩阵。**这是因为在数学中，如果一个矩阵的行列式为零，那么这个矩阵就被称为奇异矩阵或者非可逆矩阵。
具体来说，对于一个$n$阶方阵$A$，如果它的行列式等于零，那么根据线性代数的基本定理，能够知道 A 没有逆矩阵。换句话说，对于任何向量$x$，我们都有$Ax=0$，这意味着$x$是$A$的一个解，但$A$并没有唯一确定的解。因此，$A$被认为是奇异的，或者说是不可逆的。

$$
\begin{align*}
A \in \mathbb{R}^{n \times n} \\
n\text{det}(A) = 0 \\
n\Rightarrow A^{-1} \notin \mathbb{R}^{n \times n} \\
n\Rightarrow A \text{ is singular or non-invertible.}
\end{align*}
$$

其中，$\mathbb{R}^{n \times n}$表示实数空间中的 n 阶方阵，$\text{det}(A)$表示矩阵 A 的行列式，$A^{-1}$表示矩阵 A 的逆矩阵，$\notin$表示不等于。

## 矩阵的表达

在矩阵的表达方式上，我们需要遵循以下规则：矩阵元素必须在"[]"内；同行元素之间用空格（或","）隔开；行与行之间用";"（或回车符）隔开。

## 矩阵基本运算

矩阵的基本运算包括加法、减法、乘法和标量乘法。

### 矩阵加减法

两个同型矩阵相加或相减，结果仍为同型矩阵。设$A$和$B$是同型矩阵，则它们的和$C=A+B$和差$D=A-B$分别满足以下条件：
$C_{ij} = A_{ij} + B_{ij} \qquad D_{ij} = A_{ij} - B_{ij}$
其中，$C_{ij}$和$D_{ij}$分别表示$C$和$D$中第$i$行第$j$列的元素。

运算方式：

- 矩阵加法：$\mathbf{C} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} + \begin{bmatrix} e & f \\ g & h \end{bmatrix} = \begin{bmatrix} a+e & b+f \\ c+g & d+h \end{bmatrix}$
- 矩阵减法：$\mathbf{D} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} - \begin{bmatrix} e & f \\ g & h \end{bmatrix} = \begin{bmatrix} a-e & b-f \\ c-g & d-h \end{bmatrix}$

### 标量乘法

一个矩阵和一个标量相乘，结果是将该标量乘以该矩阵的所有元素。设$\mathbf{A}$是一个矩阵，$\lambda$是一个标量，则它们的乘积$\mathbf{A}\lambda$满足以下条件：
$(\mathbf{A}\lambda)_{ij} = \lambda A_{ij}$
其中，$\mathbf{A}\lambda$表示将$\mathbf{A}$的所有元素都乘以$\lambda$,而$\lambda A_{ij}$表示将$\lambda$乘以$\mathbf{A}$的第$i$行第$j$列的元素。

运算方式：

- 标量乘法：$\mathbf{A}\lambda = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \lambda = \begin{bmatrix} a\lambda & b\lambda \\ c\lambda & d\lambda \end{bmatrix}$

### 矩阵乘法

两个矩阵相乘，结果是一个积矩阵。设$A$和$B$是两个$m \times n$矩阵和$n \times p$矩阵，则它们的积$C=AB$满足以下条件：
$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$
其中，$C_{ij}$表示$C$中第$i$行第$j$列的元素，而$\sum_{k=1}^{n} A_{ik} B_{kj}$表示对$B$的第$j$列元素求和，并对每个元素与对应的$A$中第$i$行元素相乘再求和。

运算方式：

- 矩阵乘法：$\mathbf{C} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} e & f \\ g & h \end{bmatrix} = \begin{bmatrix} ae + bg & af + bh \\ ce + dg & cf + dh \end{bmatrix}$

## 特征矩阵计算 \*

## 协方差矩阵计算 \*

### 协方差矩阵 \*

### 特征值与特征向量计算

A 为 n 阶矩阵，若数 λ 和 n 维非 0 列向量 x 满足 Ax=λx，那么数 λ 称为 A 的特征值，x 称为 A 的对应于特征值 λ 的特征向量。

式 Ax=λx 也可写成(A-λE)x=0，E 是单位矩阵，并且|A-λE|叫做 A 的特征多项式。当特征多项式等于 0 的时候，称为 A 的特征方程，特征方程是一个齐次线性方程组，求解特征值的过程其实就是求解特征方程的解。

对于协方差矩阵 A，其特征值$\lambda$（可能有多个）计算方法为：

$$
\left| A-\lambda E \right| = 0
$$

行列式 $\left| A \right| = ad-bc$是 A 的二阶行列式

$$
E =
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
\end{bmatrix}
$$

对数字图像矩阵做特征值分解，其实是在提取这个图像中的特征，这些提取出来的特征是一个个的向量，即对应着特征向量。而这些特征在图像中到底有多重要，这个重要性则通过特征值来表示。

比如一个 100x100 的图像矩阵 A 分解之后，会得到一个 100x100 的特征向量组成的矩阵 Q，以及一个 100x100 的只有对角线上的元素不为 0 的矩阵 E，这个矩阵 E 对角线上的元素就是特征值，而且还是按照从大到小排列的(取模，对于单个数来说，其实就是取绝对值)，也就是说这个图像 A 提取出来了 100 个特征，这 100 个特征的重要性由 100 个数字来表示，这 100 个数字存放在对角矩阵 E 中。

所以归根结底，特征向量其实反应的是矩阵 A 本身固有的一些特征，本来一个矩阵就是一个线性变换，当把这个矩阵作用于一个向量的时候，通常情况绝大部分向量都会被这个矩阵 A 变换得“面目全非”，但是偏偏刚好存在这么一些向量，被矩阵 A 变换之后居然还能保持原来的样子，于是这些向量就可以作为矩阵的核心代表了。

于是我们可以说:一个变换(即一个矩阵)可以由其特征值和特征向量完全表述，这是因为从数学上看，这个矩阵所有的特征向量组成了这个向量空间的一组基底。而矩阵作为变换的本质其实就是把一个基底下的东西变换到另一个基底表示的空间中。

[^1]: 二阶行列式

    二阶行列式：$\lvert \mathbf{A} \rvert = ad - bc$

    其中，$a$ 和 $d$ 是矩阵 $\mathbf{A}$ 的行和列的线性组合，$b$ 和 $c$ 是矩阵 $\mathbf{A}$ 的行和列的线性组合。

    二阶行列式是矩阵的行列式，而矩阵的行列式是矩阵的逆矩阵。
