# 滤波

滤波是图像处理中的一个重要的技术，它可以用来消除图像中的噪声，提高图像的清晰度，增强图像的对比度，提高图像的分辨率等。

## 卷积和滤波的区别

滤波和卷积都是图像处理中常用的操作，它们在原理上具有一定的相似性。然而在实现细节上，二者存在一些区别。滤波操作是图像对应像素与掩膜（mask）的对应元素相乘相加。而卷积操作则是将一个卷积核（也称为掩膜）在图像上滑动，计算卷积核与滑动窗口内的图像像素的乘积之和。

虽然滤波和卷积操作有所区别，但两者也有紧密的联系。具体来说，图像的线性滤波就是通过卷积操作来完成的。此外，卷积核可以看作是一种特殊的滤波器，只不过它是用来对图像进行特征提取的，也就是说，卷积核是由学习得到的权重参数构成的。因此，滤波和卷积常常被同时应用在图像处理和深度学习等领域中。

## 图像滤波技术概述

图像滤波技术分为以下几种：

1. 均值滤波
2. 中值滤波
3. 一阶($\alpha \beta$)滤波
4. 卡尔曼滤波
5. 高斯滤波
6. 双边滤波
7. 锐化滤波
8. 边缘检测

简单的滤波算法效果有限，只能处理线性数据；且每个滤波算法都有自己的局限，针对于不通的问题需要选择合适的方法：

> 测试数据

```python
import random
import math
import numpy as np
import matplotlib.pyplot as plt

n = 500
real = [] # 真值
mear = [] # 观测值
pred = [] # 滤波值

# 建立真值和观测值
for i in range(n):
  num = 0.003 * i
  real.append(num)
  num += 0.1 * np.random.standard_normal()  # 本身的不确定性
  num += 0.5 * np.random.standard_normal()  # 观测的不确定性
  mear.append(num)

plt.plot(range(n), mear)
plt.plot(range(n), real)
plt.show()
```

![test_data](/docs/chapter04_number_image/filtering.assets/test_data.png)

## 滤波分类

### 均值滤波

均值滤波是典型的线性滤波算法，在图像中应用比较多，原理是以该像素点周围的八个像素点取平均操作，然后替代该像素点，也就是卷积操作。

对于处理简单的线性数据 y=ax+b，原理也是类似的，取该点周围的 n 个点取平均即可，n 可以看为是一个滑窗。因此，可以取该点的前后 n 个数据的平均值，也可以取前 n 个数据的平均值，根据不同场景数据设计即可。

如下代码比较适合离线数据处理，是对原始观测的数据中取某点的前后滑窗大小的均值，好比图像中应用中就是对原始图片滤波。如果对于在线数据，一个不断增加数据的数组，建议使用一阶滤波器或者 kalman 滤波器。

```python
# window滑窗越大，滤波效果越明显，结果越滞后
# 设置了该点的左右滑窗大小，可根据实际情况选取参数
def average_filter(window_left, window_right, arr):
  size = len(arr)
  result = []
  for i in range(window_left, size-window_right):
    sum = 0
    # 滑窗
    for j in range(-window_left, window_right+1):
      sum += arr[i+j]
    sum /= (window_left + window_right + 1)
    result.append(sum)
  return result

pred = [] # 滤波值
# 前后5个，总共11个点求平均值
pred = average_filter(5, 5, mear)

# 前5个数，总共6个点求平均值
# pred = average_filter(5, 0, mear)


plt.plot(range(n), mear)
plt.plot(range(n), real)
# 会牺牲掉前后window大小的数据，可以作相应改进
plt.plot(range(len(pred)), pred)
print(len(pred))
```

![average](/docs/chapter04_number_image/filtering.assets/average.png)

### 中值滤波

和均值滤波相似，同样是选取固定大小滑窗，然后选取滑窗内的中位数作为滤波结果。或者选取中位数平均数，类似比赛中去掉最高最低分，对其余比分求平均，这种可以叫做中位值平均滤波法。思路都是差不多的，都是需要做一遍排序。

中值滤波能有效克服偶然因素引起的波动噪声。

中值滤波对椒盐噪声（椒盐噪声是指图像中出现了白点或者黑点，可能是光亮的亮点或者暗的暗点）非常有效。

```python
# window滑窗越大，滤波效果越明显，结果越滞后
# 设置了该点的左右滑窗大小，可根据实际情况选取参数
def Median_Filter(window_left, window_right, arr):
  size = len(arr)
  result = []
  for i in range(window_left, size-window_right):
    # 滑窗
    temp = []
    for j in range(-window_left, window_right+1):
      temp.append(arr[i+j])
    temp.sort()
    point = temp[(int)(len(temp)/2)]
    result.append(point)
  return result

# 中值平均值滤波
def MedianAvg_Filter(window_left, window_right, arr):
  size = len(arr)
  result = []
  for i in range(window_left, size-window_right):
    # 滑窗
    temp = []
    for j in range(-window_left, window_right+1):
      temp.append(arr[i+j])
    temp.sort()
    # 可以去掉最大值后，取中位数的平均值
    median_mean = []
    for m in range(1, len(temp)-1):
      median_mean.append(temp[m])

    result.append(np.mean(median_mean))
  return result

pred = [] # 滤波值
# 前后5个，总共11个点求中值
pred = Median_Filter(5, 5, mear)
# pred = MedianAvg_Filter(5, 5, mear)

# 前5个数，总共6个点求中值
# pred = Median_filter(5, 0, mear)


plt.plot(range(n), mear)
plt.plot(range(n), real)
# 会牺牲掉前后window大小的数据，可以作相应改进
plt.plot(range(len(pred)), pred)
```

![middle](/docs/chapter04_number_image/filtering.assets/middle.png)

### 一阶($\alpha \beta$)滤波

一阶滤波是比较常用简单的滤波方法，就是当前采样结果和上一个滤波结果加权求和，权重和为 1。对周期干扰噪声有良好的抑制作用，但同样会产生相位滞后，权重是固定值也是其缺点之一。

```python
# a值越小，越不相信观测，滤波效果越明显，结果越滞后
def ab_filter(a, now):
  global last
  return a * now + (1 - a) * last

pred = []
last = mear[0]
pred.append(last)

for i in range(1, n):
  last = ab_filter(0.4, mear[i])
  pred.append(last)

plt.plot(range(n), mear)
plt.plot(range(n), real)
plt.plot(range(n), pred)
```

![alpha](/docs/chapter04_number_image/filtering.assets/alpha.png)

### 卡尔曼滤波

#### 卡尔曼滤波算法原理(KF,EKF,AKF,UKF)

[卡尔曼滤波算法原理(KF,EKF,AKF,UKF)](https://blog.csdn.net/weixin_43152152/article/details/115753921)

```python
# 滤波效果主要调整参数：
# 过程噪声方差q(越小越相信预测，反之亦然)， 观测噪声方差r(越小越相信观测，反之亦然)
q, r = 0.1, 2
# 状态均值x， 过程噪声均值w，方差p
x, w, p = 0, 0, 0
def kalman_filter(z):
  global x, p
  # 预测
  x_ = x + w
  p_ = p + q
  k = p_ / (p_ + r)
  # 更新
  x = x_ + k * (z - x_)
  p = (1-k) * p_
  return x

pred = [] # 滤波值
for i in range(n):
  pred.append(kalman_filter(mear[i]))

plt.plot(range(n), mear)
plt.plot(range(n), real)
plt.plot(range(n), pred)
```

![klman](/docs/chapter04_number_image/filtering.assets/klman.png)

### 高斯滤波

#### 前置了解

> 卷积：分析数学中的重要运算
>
> 设:$f(x)$,$g(x)$是 R1 上的两个可积函数，作积分：$\int_{-\infty}^{\infty}f(\tau)g(x-\tau)d \tau$
>
> 可以证明，关于几乎所有的实数$x$，上述积分存在。
> 这样，随着$x$的不通取值，这个积分就定义了一个新函数$h(x)$，称为$f$与$g$的卷积，记作$h(x)=(f*g)(x)$。
>
> 卷积是一个单纯的定义，本身无意义，但在各个领域的应用是十分广泛的，在滤波中可以理解为是一个加权平均的过程，每个像素点的值都由其本身和邻域内的其他像素值经过加权平均后得到，而如何加权则是依据核函数高斯函数。

#### 概述

高斯滤波是一种线性平滑滤波，适用于消除高斯噪声，广泛应用于图像处理的减噪过程。

通俗的讲，高斯滤波(Gaussian filter)就是对整幅图像进行**加权平均**的过程，每一个像素点的值，都由其本身和邻域内的其他像素值经过加权平均后得到。

高斯滤波包含许多种，包括低通、带通和高通等。

> 高斯高通滤波：
> 高斯高通滤波器会锐化图像，原理是：保留图像高频成分（图像细节部分），过滤图像低频成分（图像平滑区域）

> 高斯低通滤波：
> 高斯低通滤波器会模糊图像，原理是：过滤图像高频成分（图像细节部分），保留图像低频成分（图像平滑区域）

高斯滤波的具体操作是：用一个模板（或称卷积、掩模）扫描图像中的每一个像素，用模板确定的邻域内像素的加权平均灰度值去替代模板中心像素点的值用。

#### 高斯模糊

我们通常图像上说的高斯滤波，指的是`高斯模糊(Gaussian Blur)` ，是一种高斯低通滤波，所以对图像进行‘高斯模糊’后，图像会变得模糊。

> 高斯模糊对于抑制**高斯噪声 (服从正态分布的噪声) **非常有效。

#### 算法

在图像处理中，高斯滤波一般有两种实现方式，一是用**离散化窗口滑窗卷积**，另一种通过**傅里叶变换**。

最常见的就是第一种滑窗实现，只有当离散化的窗口非常大，用滑窗计算量非常大（即使用可分离滤波器的实现）的情况下，可能会考虑基于傅里叶变化的实现方法。

由于高斯函数可以写成可分离的形式，因此可以采用可分离滤波器实现来加速。

> 可分离滤波器：是可以把多维的卷积化成多个一维卷积。具体到二维的高斯滤波，就是指先对行做一维卷积，再对列做一维卷积。这样就可以将计算复杂度从 $O(M^2*N^2)$降到 $O(2M^2N)$，M，N 分别是图像和滤波器的窗口大小。

高斯模糊是一个非常典型的图像卷积的例子，本质就是将（灰度）图像 I 核一个高斯核进行卷积操作：

$$
I_{\sigma}=I \otimes G_{\sigma}
$$

其中：$\otimes$表示卷积操作，$G_{\sigma}$是标准差为$\sigma$的二维高斯核，定义为：

$$
G_{\sigma}=
\frac{1}{2\pi\sigma^2}
e^{-\frac{x^2+y^2}{2\sigma^2}}
$$

### 双边滤波

<!-- todo -->

### 锐化滤波

<!-- todo -->

### 边缘检测

<!-- todo -->

## 总结

对于简单的线性数据处理完之后就可使用最小二乘法来拟合出一个比较好的结果；

因为各个滤波器取的参数不一，结果对比起来是没有意义的，而且采样点比较多，没有具体分析细节，建议应用时测试充分选取合适的方法。
