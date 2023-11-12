# 常用第三方库

OpenCV、matplotlib 和 skimage 都是非常强大的图像处理库，下面是它们的一些基础入门案例：

## OpenCV 基础入门

```python
import cv2

# 读取图片
img = cv2.imread('example.jpg')

# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 注意是BGR而不是RGB

# 显示图片
cv2.imshow('Gray image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## matplotlib 基础入门

```python
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 绘制图形
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine wave')
plt.show()
```

## skimage 基础入门

```python
from skimage import data, io, color, exposure, filters
import matplotlib.pyplot as plt

# 加载图片
img = data.chelsea()

# 转换为灰度图
gray = color.rgb2gray(img)

# 应用高斯滤波器
blurred = filters.gaussian(gray, sigma=1)

# 显示原始图片和处理后的图片
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].imshow(gray, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(blurred, cmap='gray')
ax[1].set_title('Blurred Image')
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.show()
```