

## Autoregressive Model

原来以为自回归模型就是指GPT这种，今天查了一下发现不是这样的，于是做一个总结。

自回归其实就是将自己的1至p阶当作自变量进行回归，其分析了过去多个时刻数据对当前时刻结果的影响。

对于p阶自回归模型AR(p)，有
$$
y_t = \alpha_0 + \alpha_1 y_{t-1} + \alpha_2 y_{t-2} + \cdots + \alpha_p y_{t-p} + \varepsilon_t\\
其中\varepsilon_t是方差为\sigma^{2}的白噪声序列
$$

#### 1.ARMA模型

这个模型将AR算法与MA算法相结合，是数学建模中比较常用的一种模型。

> ##### MA模型(移动平均模型)：
>
> 这个模型分析了当前时刻结果与过去多个时刻噪声之间的关系
>
> 对于p阶移动平均模型MA(p)，有：
> $$
> y_t = \varepsilon_t + \beta_1 \varepsilon_{t-1} + \beta_2 \varepsilon_{t-2} + \cdots + \beta_q \varepsilon_{t-q}
> $$

ARMA模型的AR部分通过线性组合过去的值来预测当前值，MA部分通过线性组合过去的误差来预测当前值。

对于ARMA(p,q)模型，有：
$$
\begin{align*}
y_t &= \alpha_0 + \sum_{i=1}^{p} \alpha_i y_{t-i} + \varepsilon_t + \sum_{i=1}^{q} \beta_i \varepsilon_{t-i} \\
&\Rightarrow y_t = \alpha_0 + \sum_{i=1}^{p} \alpha_i L^i y_t + \varepsilon_t + \sum_{i=1}^{q} \beta_i L^i \varepsilon_t \\
&\Rightarrow (1 - \sum_{i=1}^{p} \alpha_i L^i) y_t = \alpha_0 + (1 + \sum_{i=1}^{q} \beta_i L^i) \varepsilon_t\\
\end{align*}
$$

$$
公式中的L代表滞后因子,Ly_t=y_{t-1}\\
其中 (1 - \sum_{i=1}^{p} \alpha_i L^i) y_t 为AR(p)格式,(1 + \sum_{i=1}^{q} \beta_i L^i) 是MA(p)格式
$$



#### 2.ARIMA模型

在ARMA模型基础上加入差分，

对于ARIMA(p,d,q)模型，有：
$$
(1 - \sum_{i=1}^{p} \alpha_i L^i)(1 - L)^d y_t = \alpha_0 + (1 + \sum_{i=1}^{q} \beta_i L^i) \varepsilon_t\\
其中(1 - \sum_{i=1}^{p} \alpha_i L^i)是AR部分，(1 - L)^d是差分部分，(1 + \sum_{i=1}^{q} \beta_i L^i)是MA部分
$$

#### 3.CRF（条件随机场）

一种用于标记和分割序列数据的概率模型，通过条件概率建模序列中的标记序列
$$
P(Y \mid X) = \frac{1}{Z(X)} \exp \left( \sum_{k=1}^{K} \lambda_k f_k(Y, X) \right)
$$

#### 4.RNN

来到了深度学习领域，RNN通过多个
$$
h_t = \sigma (W_h h_{t-1} + W_x x_t + b)\\
P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}),
$$

> ![../_images/rnn.svg](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407182109000.svg)
>
> ​                                                                                                          *来自d2l*

第n个时刻的结果取决于前n-1个时刻，

#### 5.LSTM和GRU

#### 6.Casually Language Model

这就是我们都知道的GPT等文章的目标函数，根据前文的词语来推测下个词语
$$
P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, w_2, \ldots, w_{t-1})
$$










