# 中心对称证明

原图像 $M \times M$ --> 目标图像 $N \times N$

目标图像在原图像坐标系的位置为 $(x, y)$

原图坐标$(x_m,y_m), m = 0,1,2,...,M-1$, 几何中心 $(x_c,y_c), c=\frac{M-1}{2}$
目标图坐标 $(x_n,y_n), n = 0,1,2,...,N-1$, 几何中心 $(x_d,y_d), d=\frac{N-1}{2}$

此时有

$$
m = n \times \frac{M}{N}
$$

要使几何中心相同，那么必存在一个值$Z$，使得$\frac{M-1}{2}+Z  = (\frac{N-1}{2}+Z)\frac{M}{N}$;

解括号得：

$$
Z + \frac{M-1}{2} = \frac{(N-1)M}{2N} + \frac{ZM}{N}
$$

移项得：

$$
Z - \frac{ZM}{N} = \frac{(N-1)M}{2N} - \frac{M-1}{2}
$$

化简得：

$$
Z(1-\frac{M}{N}) = \frac{-M+N}{2N}
$$

来来来，嚼烂了吃……

$$
\begin{align}
Z(\frac{N-M}{N}) &= \frac{N-M}{2N} \\
&=\frac{1}{2}(\frac{N-M}{N})
\end{align}
$$

来来来，嚼烂了吃……

$$
Z =\frac{1}{2}
$$
