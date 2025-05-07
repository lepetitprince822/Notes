## LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS

LoRA通过将低秩分解矩阵注入transformer架构，实现参数量的减小。在LLM中相比传统微调有不少优势

#### 1 INTRODUCTION

自然语言处理的一些应用都需要将预训练过的模型进行微调适应特定任务，但是这种方法随着模型参数的增大变得越来越困难。过去针对这个问题，人们都说使用**1.冻结部分参数、2.prefix tuning**等方法来降低计算，但是这样操作以后性能会变差，不如baseline

Li et al. (2018a) 和 Aghajanyan et al. (2020) 的研究发现尽管模型参数很多，但是权重的变化只会变化小部分参数。基于此，这篇文章就假设LLM权重的变化可以用低秩矩阵来有效表示。

#### 2 PROBLEM STATEMENT

文章首先针对文本建模讲述了预训练和微调过程：
$$
\max_{\Phi} \sum_{(x, y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log \left( P_{\Phi} (y_t \mid x, y_{<t}) \right)\\
很经典的语言建模目标函数，通过\sum_{t=1}^{|y|} \log \left( P_{\Phi} (y_t \mid x, y_{<t}) \right)计算出这个序列\textit{成功生成目标词}的对数似然\\
而 \sum_{(x, y) \in \mathcal{Z}}是针对整个数据集，最大化该训练数据中成功生成当前目标词的对数似然概率的总和\\
$$

$$
\max_{\Theta} \sum_{(x, y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log \left( p_{\Phi_0 + \Delta\Phi(\Theta)} (y_t \mid x, y_{<t}) \right)\\
假设预训练后的参数为\phi_0，那么针对于某种任务的微调就是将参数变为\phi_0+\Delta\phi，因为反向传播导致的参数更新会很大，所以全部微调成本很高。\\
而lora就是希望用低秩来表示\Delta\phi，增加计算与存储效率
$$

#### 3 AREN’T EXISTING SOLUTIONS GOOD ENOUGH?

首先先了解一下过去的几种微调手段

- Adapter（[ Parameter-Efficient Transfer Learning for NLP ](https://arxiv.org/abs/1902.00751):这个策略是在transformer的attention层与FFN层后加入Adapter层。

  <img src="https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407271629160.png" alt="image-20240727040708795" style="zoom: 67%;" />

> 右图展示了Adapter Layer内部结构，首先对输入进行降维操作，接着进入一个非线性激活函数，最后再进行升维操作，还有一个残差操作
> $$
> h = \text{ReLU}(x W_{\text{down}}) W_{\text{up}} + x
> $$
> 所有的 adapter 初始化参数都从均值为 0 标准差为 0.01 的正态分布中采样。这样可以保证刚开始训练时，adapter 主干的全连接网络的输出很小，主要由残差连接传递信息。

在微调的时候，冻结transformer的参数，只能更改Adapter层的参数，这样参数量会减少很多。而且不同的任务对应不同的adapter

**缺点：**虽然提高了训练速度，但是预测速度却降低了，精度往往还有所损失。

另外针对adapter的改进也有很多，这里我没有时间去读，不过查到了几种：**1.AdapterFusion 2.AdapterDrop 3.Compacter**

- prompt tuning：将整个输入作为可学习的参数，微调时只更改这一部分的参数，

- prefix tuning：在输入前面加上一个前缀，通过更改前缀的参数来适应不同任务。不过事实上只在K和V前面加了prefix，Q并没有增加

  <img src="https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407271629176.png" alt="image-20240727145259676" style="zoom:67%;" />

  文章认为prefix会导致真实输入序列的长度变小，导致性能变差。

#### 4 OUR METHOD

这一章终于要介绍lora了，lora是应用于全连接层的（dense layer）

##### 4.1 LOW-RANK-PARAMETRIZED UPDATE MATRICES

首先文章提到了一个研究证明预训练的语言模型具有很低的Instrisic dimension，他们将模型随机投影到一个较小的子空间，模型仍然可以有效的学习。（*Aghajanyan et al. (2020) shows that the pre-trained language models have a low “instrisic dimension” and can still learn efficiently despite a random projection to a smaller subspace.*）

> 详解：这里其实就用到了秩的知识，首先一个m×n的矩阵可以认为是将n维空间的向量投影到m维空间的线性变化，矩阵相乘也正是利用了这个性质，矩阵的秩反映了这个矩阵所能表达的最大维度（例如一个3×3的矩阵秩为2，其基为2。当n=3的向量与其做线性变化时，可以认为是将三维向量投影到了2维平面）
>
> 回到这里，这篇文章认为预训练模型有效维度并不高，又或者，其本身维度很高，但是在适应某种特定任务的时候并不需要那么高的维度，只需要一个低维子空间。

所以Aghajanyan et al.这篇文章使用了随机投影来验证猜想。
$$
首先创建一个随机矩阵 R，其维度为 d×k，其中 d 是原始高维空间的维度，k 是目标低维空间的维度。\\R中元素是从某种分布例如高斯分布随机抽取的\\
将原始高维数据 𝑋乘以随机投影矩阵 𝑅，得到低维表示 𝑋′\\
X'=XR\\
接着使用X'作为输入做特定任务，结果并不差
$$
lora借鉴了这个思路
$$
h = W_0 x + \Delta W x = W_0 x + B A x\\
对于微调的参数\Delta W，将其分解为BA，其中B\in R^{d×r},A\in R^{r×k}，且r<<min(d, k)\\
其中A初始化使用高斯分布，B设为0
$$
<img src="https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407271629087.png" alt="image-20240727162934734" style="zoom:50%;" />
$$
接着又引入了一个参数\alpha\\
h = W_0 x + \frac{\Delta W x}{\alpha} = W_0 x + \frac{B A x}{\alpha}\\
用来平衡与训练权重与微调权重
$$
通过选择合适的r，可以让lora达到full fine-tuning的效果，另外lora也没有很高的推理延迟

##### 4.2 APPLYING LORA TO TRANSFORMER

这篇文章只验证了Lora应用在注意力层的权重的效果（Wq,Wk,Wv,Wo），取得了不错的效果

lora在推理时候也不会降低推理效率（这里我的理解是lora本来就是通过分解矩阵降低计算量，最开始需要d^2计算量，使用lora后变成了2dr的计算量，小了很多。在推理时候，再把以及确定的BA计算出来直接加到W0上，这样也不需要额外再计算BA了，跟普通的推理计算量一样）但是针对于多任务，lora会显得麻烦（不同任务需要不同的B和A，需要不断切换）

#### 5 EMPIRICAL EXPERIMENTS

这个地方比较了几种微调方法

1. fine-tuning：最常见的微调

2. Bias-only or BitFit：所有权重都矩阵都冻结，只修改bias

3. Prefix-embedding tuning：这种方法只在输入的地方加上可以训练的token

4. Prefix-layer tuning (PreLayer)：这种是在prefix-embedding的基础上也可以改变激活值（其实就是中间值。以transformer为例，qkv计算后的结果就可以称为激活值，layernormal后的也可以称为激活值），把激活值也当作一个可以学习的参数

   ```python
   x = self.transformer.layers[i](x)
   x = self.prefix_activations[i]  
   
   #结合激活值和学习参数的方法
   #1.加法
   x = x + prefix_activation
   #2.拼接
   x = torch.cat([prefix_activation, x], dim=1)
   ```

5. Adapter tuning：前面提到过这种

6. lora

接着就放出来实验结果表格，发现lora在保证低参数量的同时性能也很好

![image-20240727183200267](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407271832192.png)





