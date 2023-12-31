# 图像滤波

图像滤波是图像处理中的一种基本操作，用于改善图像质量或提取图像特征。它主要通过对图像中的像素值进行加权平均来达到消除噪声、增强图像边缘等目的。(1)
{.annotate}

1. 即在尽量保留图像细节特征的条件下对目标图像的噪声进行抑制，是图像预处理中不可缺少的操作，其处理效果的好坏将直接影响到后续图像处理和分析的有效性和可靠性。

## 图像滤波的目的

1. 消除图像中混入的噪声
2. 为图像识别抽取出图像特征

## 图像滤波的应用

图像滤波的应用非常广泛，例如在医学影像、深度相机、闪光图像、深度图像等领域都有重要应用。因此，对图像滤波的理解和掌握对于图像处理和分析具有重要的意义。

## 图像滤波的要求

1. 不能损坏图像轮廓及边缘
2. 图像视觉效果应当比处理前更好

## 前置知识

!!! quote "滤波器"

    一种形象的比喻法是：我们可以把滤波器想象成一个包含加权系数的窗口，当使用这个滤波器平滑处理图像时，就把这个窗口放到图像之上，透过这个窗口来看我们得到的图像。

## 滤波的方式

常见的图像滤波方式有线性滤波和非线性滤波两种。线性滤波包括方框滤波、均值滤波和高斯滤波等，这些方法通常用于平滑图像或减少噪声；非线性滤波主要有中值滤波和双边滤波，常用于去除椒盐噪声或其他异常值。在 OpenCV 库中，我们可以使用 medianBlur()函数来实现中值滤波。

此外，有些复杂的滤波技术如导向滤波、联合双边滤波等，它们在保留边缘信息等方面有更好的表现。

### 线性滤波

???+ note "方框滤波"

    方框滤波是一种线性平滑滤波器，使用像素点邻域的加权平均灰度来替代像素点的灰度值，主要用来消除图像中的噪声。

???+ note "平滑滤波"

    平滑滤波是低频增强的空间域滤波技术。它的目的有两类:一类是模糊;另一类是消除噪音。空间域的平滑滤波一般采用简单平均法进行，就是求邻近像元点的平均亮度值。邻域的大小与平滑的效果直接相关，邻域越大平滑的效果越好，但邻域过大，平滑会使边缘信息损失的越大，从而使输出的图像变得模糊，因此需合理选择邻域的大小。

???+ note "均值滤波"

    图像处理中最常用的手段，从频率域观点来看均值滤波是一种低通滤波器，高频信号将会去掉，因此可以帮助消除图像尖锐噪声，实现图像平滑，模糊等功能。理想的均值滤波是用每个像素和它周围像素计算出来的平均值替换图像中每个像素。

    公式: $g(x,y)=\frac{1}{M}\sum_{f \in s}f(x,y)$

    > 特性

    - 从左到右再从上到下计算图像中的每个像素，最终得到处理后的图像。
    - 均值滤波可以加上两个参数，即迭代次数，Kernel 数据大小
        - 一个相同的 Kernel，多次迭代，效果越好
        - 迭代次数相同，Kernel 矩阵越大，效果越好

    !!! tip

        这个 kernel 加权求和后再除 9 才是均值，用均值替换中心像素

    > 优缺点
    >
    > 优点：算法简单，计算速度快
    > 缺点：降低噪声的同时使图像产生模糊，特别是景物的边缘和细节部分

### 非线性滤波

???+ note "中值滤波"

    > 基于排序统计理论的一种能有效抑制噪声的非线性信号处理技术。

    中值滤波是一种非线性平滑技术，它将每一像素点的灰度值设置为该点某邻域窗口内的所有像素点灰度值的中值。

    中值滤波的基本原理是把数字图像或数字序列中一点的值用该点的一个邻域中各点值的中值代替，让周围的像素值接近的真实值，从而消除孤立的噪声点。

    > 优缺点
    >
    > 优点：抑制效果很好，画面的清晰度基本保持
    > 缺点：对高斯噪声的一直效果不是很好

???+ note "最大最小值滤波"

    最大最小值滤波是一种比较保守的图像处理手段，与中值滤波类似，首先要排序周围像素和中心像素值，然后将中心像素值与最小和最大像素值比较，如果比最小值小，则替换中心像素为最小值，如果中心像素比最大值大，则替换中心像素为最大值。

### 更复杂的滤波（了解）

???+ note "引导滤波"

    ??? quote "局部线性模型"

        在引导滤波的定义中，会用到局部线性模型。

        该模型认为，某函数上一点与其临近部分的点成线性关系，一个复杂的函数就可以用很多局部的线性函数来表示，当需要求该函数上某一点的值时，只需计算所有包含该点的线性函数的值并做平均即可。这种模型在表示非解析函数上非常有用。

        局部线性模型在图像处理中也有很好的应用，如图像增强、图像去噪、图像复原、图像分割等。

    ??? quote "矩阵归一化"

        矩阵归一化(Matrix Normalization)是归一化矩阵的行和列，使得矩阵的每一行和每一列的平方和为 1。

        归一化矩阵的行和列的平方和为 1 的矩阵称为正交矩阵，正交矩阵的逆矩阵等于其转置矩阵。

        归一化的目标主要是使得预处理的数据被限定在一定范围内，例如 [0,1]或者 [-1,1]，从而消除奇异样本数据导致的不良影响。所谓的奇异样本数据，是指相对于其他输入样本特别大或特别小的样本矢量（即特征向量）。这类数据的存在可能会引发训练时间增加和无法收敛等问题。因此，在进行训练之前，需要对预处理数据进行归一化。

        至于归一化的原理则涉及到数据的处理。数据处理的归一化就是将矩阵的数据以列为单元，按照一定比例映射到某一区间。具体的方法有截断饱和归一化（设置最大最小值，或者用饱和函数截断），中心化（$x-\mu$）等。

        在计算机视觉中，归一化主要用于图像处理。通过将图像的像素值缩放到一个特定的范围（如[0,1]或者[-1,1]），可以消除光照变化对图像处理的影响，提高图像处理的效果。

    引导滤波，也被称为导向滤波，是一种非线性的图像滤波技术。这种技术利用一张引导图对初始图像进行滤波处理，使得最后的输出图像在大体上与初始图像相似，但在纹理部分与引导图相似。

    根据一种基本假设，滤波输出可以看作是引导图像的局部线性变换。此外，它还被誉为保边滤波器之一，其在去除噪声的同时能保留边缘信息。相对于常见的均值滤波、高斯滤波等各向同性滤波器，导向滤波最大的特点是在去除噪声的同时，能最大限度保持边缘不被平滑。

    导向滤波的原理是利用初始图像和引导图像的局部相似度作为权值，将初始图像与引导图像进行加权求和，从而实现对图像的滤波处理。

    公式：

    $g(x,y) = \frac{I(x,y) * \alpha + \beta}{1 + \alpha}$ 其中，$I(x,y)$为初始图像，$\beta$为常数，$\alpha$为相似度系数。

    步骤：

    1. 计算初始图像和引导图像的相似度，得到相似度矩阵
    2. 相似度矩阵归一化
    3. 相似度矩阵与初始图像做加权求和
    4. 得到导向滤波后的图像

    另外，值得一提的是，与双边滤波器相比，导向滤波在边界附近效果较好；同时它还具有O(N)的线性时间的速度优势。因此，除了图像平滑之外，导向滤波还可以应用在图像增强、HDR压缩、图像抠图及图像去雾等场景。

    引导滤波的实现步骤如下：

    1. 确定窗口大小$\omega$，窗口中心为$i$。
    2. 计算窗口中像素的均值$\mu_k$。
    3. 计算窗口中像素的方差$a_k^2$。
    4. 计算窗口中像素的线性系数$a_k$。
    5. 计算窗口中像素的线性系数$b_k$。
    6. 计算滤波后的像素值。

    引导滤波的实现代码如下：
    ```python
    def guide_filter(self, I, p, epsilon=0.0001):
        '''
        :param I: 输入图像
        :param p: 引导图像
        :param epsilon: 引导滤波器的参数
        :return: 引导滤波后的图像
        '''
        # 计算窗口中像素的均值
        mu = self.window_mean(I, p)
        # 计算窗口中像素的方差
        a2 = self.window_variance(I, p)
        # 计算窗口中像素的线性系数
        a = np.sqrt(a2 + epsilon)
        # 计算窗口中像素的线性系数
        b = a * mu
        # 计算滤波后的像素值
        return self.window_mean(I, p) + self.window_mean(p, p) * self.window_mean(I, I) - self.window_mean(p, I) * self.window_mean(p, p)
    ```

    引导滤波最大的优势在于，可以写出时间复杂度与窗口大小无关的算法，因此在使用大窗口处理图片时，其效率更高。

???+ note "双边滤波"

    双边滤波(Bilateral Filter)是结合了高斯滤波器和空间域核[^1]和值域核[^2]的双滤波器。

    双边滤波是一种非线性的滤波方法，它同时考虑了空间邻近度和像素值相似度，以达到保边去噪的目的。这是一种结合图像的空间邻近度与像素值相似度的处理办法，在滤波时，该滤波方法同时考虑空间临近信息与颜色相似信息。双边滤波采用两个高斯分布，分别对应空间分布和像素值分布。

    双边滤波的基本思路是同时考虑将要被滤波的像素点的空域信息（domain）和值域信息（range）。这种 combined 滤波方式考虑到了空域信息和灰度相似性。

    双边滤波器的优点是可以做边缘保存（edge preserving），一般用高斯滤波去降噪，会较明显地模糊边缘，对于高频细节的保护效果并不明显。而双边滤波器基于空间分布的高斯滤波函数，所以在边缘附近，离的较远的像素不会太多影响到边缘上的像素值，这样就保证了边缘附近像素值的保存。

???+ note "联合双边滤波"

    联合双边滤波(Joint Bilateral Filter)是双边滤波和引导滤波的结合，在保持边缘的同时，也保持了图像的细节。

    公式：$q_i=\dfrac{1}{\left| \omega \right|} \sum_{k:i \in \omega_k} \dfrac{1}{\left| \omega \right|} \sum_{j \in \omega_k}w(i,j)
    (a_kIi+b_k)p_j$ 其中，$w(i,j)$是双边滤波的权重，$\omega_k$是所有包含像素$i$的窗口，$k$是其中心位置。

[^1]: 空间域核(Spatial Domain Kernel)是高斯核，用于保持边缘，对噪声具有自适应能力。
[^2]: 值域核(Value Domain Kernel)是高斯核，用于保持细节，对噪声具有自适应能力。
