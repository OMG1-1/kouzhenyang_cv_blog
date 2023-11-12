## 梯度

### 梯度的介绍

梯度是微积分中的一个重要概念，它描述了一个向量场在某一点的方向导数沿着该方向取得最大值，即函数在该点处沿着此方向（梯度的方向）变化最快，变化率最大（为梯度的模）。

简而言之，对多元函数的各个自变量求偏导数，并把求得的这些偏导数写成向量形式，就是梯度。
梯度的基本表达式是：
$\nabla f(x, y, z, ...) = 
\left(\frac{\partial f}{\partial x}, 
\frac{\partial f}{\partial y}, 
\frac{\partial f}{\partial z}, 
...
\right)$
其中，$\frac{\partial f}{\partial x}$、$\frac{\partial f}{\partial y}$、$\frac{\partial f}{\partial z}$ 等表示函数 $f$ 关于变量$x$、$y$、$z$的偏导数。

梯度通常用于机器学习和深度学习中的优化算法，如梯度下降法。

### 基本思想

梯度的基本思想是通过计算函数在给定点处的梯度来找到函数在该点处最陡峭上升的方向，从而确定函数在该点处的最佳搜索方向。**梯度是一个向量**，它的**方向是函数增长最快的方向**，而它的**模是函数在该方向上的增长速率**。

### 原理

梯度的原理是基于微积分中的偏导数。对于一个多变量函数 (f(x, y, z, ...))，其梯度是一个向量，包含所有偏导数。梯度的方向是函数增长最快的方向，而它的模是函数在该方向上的增长速率。

### 作用

梯度的作用主要有以下几点：

1. 梯度可以用于**优化问题**。例如，在机器学习中，我们经常需要最小化损失函数来训练模型。通过计算损失函数关于参数的梯度，我们可以使用梯度下降法等优化算法来更新参数，从而最小化损失函数。
2. 梯度可以帮助我们理解复杂函数的性质。例如，通过计算一个复杂函数的梯度，我们可以了解该函数在不同位置的变化情况。

### 梯度下降算法

#### 基本思路

...

#### 主要原理

1. 确定小目标（预测函数）

   _机器学习常见的任务是通过学习算法，自动发现数据背后的规律不断改进模型并做出预测。_

   基本方法：

   1. 找一个过原点的直线$y = wx$，即预测函数
   2. 然后计算所有样本点与直线的偏离程度
   3. 根据误差大小来调整斜率$w$
   4. 找到差距（代价函数）

   ![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697969570364-b8e41ae6-05af-42be-b9a9-8caa4749051c.png#averageHue=%23030302&clientId=u94a33e38-13fa-4&from=paste&height=389&id=u24146f77&originHeight=778&originWidth=1294&originalType=binary&ratio=2&rotation=0&showTitle=false&size=187382&status=done&style=none&taskId=u42749781-a5da-499e-be4f-a80a0356f05&title=&width=647)![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697969545172-2df16213-541e-4404-8666-3a0166f11dfc.png#averageHue=%23050505&clientId=u94a33e38-13fa-4&from=paste&height=400&id=ub7fdb3e6&originHeight=800&originWidth=1254&originalType=binary&ratio=2&rotation=0&showTitle=false&size=135004&status=done&style=none&taskId=u0f0c3da8-e1e1-4974-b8a1-c37b23501f9&title=&width=627)
   量化数据的偏离程度，即误差（常见的量化方式有：均方误差【误差平方和的平均值】）

- 得到误差函数，它代表了学习所需要付出的代价，故常被称为代价函数

通过定义预测函数，然后根据误差公式推导代价函数，成功将样本点拟合过程映射到了函数

3. 明确搜索方向（梯度计算）

   ![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697969453959-7b1189bf-274c-4506-b91b-0db7264d1c88.png#averageHue=%23040303&clientId=u94a33e38-13fa-4&from=paste&height=327&id=u48085d58&originHeight=654&originWidth=1442&originalType=binary&ratio=2&rotation=0&showTitle=false&size=192301&status=done&style=none&taskId=u2cfc53af-ca3a-405e-a5ab-0a0f8202387&title=&width=721)

4. 大胆的往前走吗？（学习率）

   ![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697969453959-7b1189bf-274c-4506-b91b-0db7264d1c88.png#averageHue=%23040303&clientId=u94a33e38-13fa-4&from=paste&height=327&id=RbNGK&originHeight=654&originWidth=1442&originalType=binary&ratio=2&rotation=0&showTitle=false&size=192301&status=done&style=none&taskId=u2cfc53af-ca3a-405e-a5ab-0a0f8202387&title=&width=721)
   ![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697969669972-58d1e191-9643-4b8b-9126-c9a8d73fab2a.png#averageHue=%23030202&clientId=u94a33e38-13fa-4&from=paste&height=336&id=ua807bbb2&originHeight=672&originWidth=1462&originalType=binary&ratio=2&rotation=0&showTitle=false&size=160295&status=done&style=none&taskId=u7d05ffe0-8cbf-4c4a-9dc3-2be549c8f10&title=&width=731)

5. 不达目的不罢休（循环迭代）

   循环计算梯度和按学习率前进这两步，直到找到最低点。
   ![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697969453959-7b1189bf-274c-4506-b91b-0db7264d1c88.png#averageHue=%23040303&clientId=u94a33e38-13fa-4&from=paste&height=327&id=BgXXy&originHeight=654&originWidth=1442&originalType=binary&ratio=2&rotation=0&showTitle=false&size=192301&status=done&style=none&taskId=u2cfc53af-ca3a-405e-a5ab-0a0f8202387&title=&width=721)
   ![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697969669972-58d1e191-9643-4b8b-9126-c9a8d73fab2a.png#averageHue=%23030202&clientId=u94a33e38-13fa-4&from=paste&height=336&id=pf1Wj&originHeight=672&originWidth=1462&originalType=binary&ratio=2&rotation=0&showTitle=false&size=160295&status=done&style=none&taskId=u7d05ffe0-8cbf-4c4a-9dc3-2be549c8f10&title=&width=731)

#### 没这么简单

在实际工作中，训练样本千奇百怪，代价函数千变万化，不太可能是简单的抛物线
如：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697973161844-2cdb3d23-6c25-45bd-a8f9-66a970f8b237.png#averageHue=%23838481&clientId=u94a33e38-13fa-4&from=paste&height=381&id=ue3f2ce61&originHeight=762&originWidth=1296&originalType=binary&ratio=2&rotation=0&showTitle=false&size=589134&status=done&style=none&taskId=ubfb1b29c-df8a-4c31-a9d6-0ef73433807&title=&width=648)
![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697973143029-beb32951-853e-4d5a-b74a-f7365d43ea7b.png#averageHue=%23020201&clientId=u94a33e38-13fa-4&from=paste&height=343&id=u313f96c2&originHeight=686&originWidth=1372&originalType=binary&ratio=2&rotation=0&showTitle=false&size=220602&status=done&style=none&taskId=ua2cc258d-8340-4551-b5b6-6075ffd2229&title=&width=686)
![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697973207237-e4d8b9cd-1778-4f30-b301-9c807209b43e.png#averageHue=%2326211a&clientId=u94a33e38-13fa-4&from=paste&height=427&id=ud01b3600&originHeight=854&originWidth=1424&originalType=binary&ratio=2&rotation=0&showTitle=false&size=680364&status=done&style=none&taskId=u0d8e2cfc-4ee3-482a-a41a-f98c86e448c&title=&width=712)

#### 梯度下降法的各种变体

##### Batch Gradient Descent(BGD)

![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697973288402-2d533501-3993-4cb4-8594-63dd581078c3.png#averageHue=%23060604&clientId=u94a33e38-13fa-4&from=paste&height=375&id=ud6236378&originHeight=750&originWidth=1374&originalType=binary&ratio=2&rotation=0&showTitle=false&size=438964&status=done&style=none&taskId=u9efc2bf4-94db-4cdc-85ed-ca1ea032dd2&title=&width=687)
特点：全部训练样本参与训练
优点：保证算法精确度，找到全局最优点
缺点：计算速度慢

##### Stochastic Gradient Descent(SGD)

![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697973502319-67f4fa09-aee9-4c09-8bc1-eaa438aacb1a.png#averageHue=%23080707&clientId=u94a33e38-13fa-4&from=paste&height=408&id=ued8b5ed9&originHeight=816&originWidth=1350&originalType=binary&ratio=2&rotation=0&showTitle=false&size=423279&status=done&style=none&taskId=ubb1adbdb-3c39-44c5-9c14-1eda4c00981&title=&width=675)
特点：每下降一步只需一个样本参与计算
优点：速度快
缺点：精准度较差

##### MBGD

![image.png](https://cdn.nlark.com/yuque/0/2023/png/28755494/1697974423051-ebd3cf51-c799-49b9-88c8-0e79f60ec355.png#averageHue=%23050504&clientId=u94a33e38-13fa-4&from=paste&height=368&id=ub941dd46&originHeight=736&originWidth=1416&originalType=binary&ratio=2&rotation=0&showTitle=false&size=434380&status=done&style=none&taskId=u91c81492-1f5b-4c9a-8389-ce1ad7a360a&title=&width=708)
特点：每下降一步只需一批次样本参与计算
优点：速度较快的同时保证了一定的精准度

##### 其他更优的算法：

- Adagrad 动态调节学习率：不常更新的学习率增大，频繁更新的学习率降低
  - 问题是：频繁跟新的学习率过小，以致逐渐消失
- RMSProp 优化动态调节学习率：解决 Adagrad 不足之处
- AdaDelta：无需设置学习率
- Adam：融合 AdaGrad 和 RMSProp
- Momentum：模拟动量
  - 下降过程中，充分考虑前一阶段下降的惯性
- FTRL……

#### 梯度下降算法并非完美无缺

- 对于学习率的设定十分敏感；太大会反复横跳，太小会浪费计算力
- 除 BGD 外，无法保证找到全局最低点，可能会陷入局部低点难以自拔
