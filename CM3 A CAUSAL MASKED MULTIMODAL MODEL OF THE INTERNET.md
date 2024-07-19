## CM3: A CAUSAL MASKED MULTIMODAL MODEL OF THE INTERNET

#### 1.Introduction

生成序列建模（**generative sequence modeling**）对于多模态模型zero-shot能力提供了巨大的帮助（文本类有GPT，图像类有DALL-E）。最近的工作提到可以使用文档结构（例如HTML网页标记）来提高文本任务的zero-shot能力。（这里我感觉可能是HTML中会包含<em>、<table>之类的消息，能够让模型更好的提取信息，GPT3就使用过HTML作为训练数据） 这篇文章说他们将超文本与图像进行联合（后面也提到了超文本结构对于训练效果帮助巨大）

#### 2 CAUSALLY MASKED OBJECTIVE

先提出目前主流的三种架构：

**encodr-only：**代表模型BERT

**encoder-decoder：**代表模型BART,T5

**decoder-only：**代表模型GPT

接着有提出目前主流的建模方式（其实就是目标函数形式）

masked：利用mlm，能够很好提供双向编码能力，但只解码15%的序列（*Masking offers the critical ability to encode bi-directionality within the prompts at the cost of only decoding roughly 15% of the tokens of the input sequence during training*）

causal language modeling：对应causal language model，解码每一个token，但只能通过前文信息来推断因果。为此causal language model也提出了许多解决方法。(*Conversely, decoder-only causal language models decode every token in the input sequence in the training but are typically limited to left-only contexts. Empirically, more work has also been done on scaling causal decoder-only rather than their counterparts.*)

为此CM3提出了**casually masked  language modeling**，具体流程如下：

1. 假设文档的长度为s，其中s为token数

2. 使用泊松分布选取mask的数量，但最后会限制在0-16之间
   $$
   n \sim \text{Clamp}(\text{Poisson}(1), 1, 16)
   $$

3. 对于每个掩码，选择一个跨度 m，该跨度表示文档中一段连续的token，其满足均匀分布
   $$
   m \sim (\text{Uniform}(0, s), \text{Uniform}(0, s))
   $$

4. 需要保证跨度m不相交，若相交则需要选择新的跨度m

```
e.g
1.假设有一个大小为 s=100 的文档
2.生成的泊松分布值为5，但由于Clamp操作，可能被调整为介于1到16之间的值，比如3。
3.选择第一个掩码的跨度m1：
起点：使用均匀分布 Uniform(0,100)选择，比如得到 20。
终点：使用均匀分布 Uniform(0,100)选择，比如得到 50。
跨度 m1=[20,50]
以此类推，得到m2=[60,80]   m3=[10,30]
4.m3和m3有重叠部分，重新选择m3
```

文章说明选择Clamp(1,16)以及Uniform都是为了选择相对**较少**的相对**较长**的跨度

在训练时将mask进行填充，将其按照顺序放在文档最后

另外对于cusual language model的损失函数，其也做出了改变，不再计算mask位置的损失（*We also augment the standard cross-entropy loss to weigh the loss of predicting mask tokens as 0, as they are placed at random locations which carry no information to the underlying sequence modeling objective.*）

![image-20240719022226432](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407190222366.png)

这个图很好的体现了casually masked  language modeling，将<mask 0>的内容放在了最后，

#### 3 CM3

其使用decoder-only架构

##### 3.1 DATA

- 由于 Common Crawl 数据集存在严重的伦理问题（如包含不适当的内容），因此作者选择不处理整个 Common Crawl 数据集，而是使用 CC-NEWS 数据集的子集和全部的英文维基百科

- 对于每个 HTML 文档，首先删除所有不包含文本的元素（如图片、脚本等）。然后过滤掉头部、脚部、版权声明、表单、对话框和 iFrame 元素，因为这些通常不包含对最终任务有意义的内容。将连续的 <div> 元素合并成一个单独的 <div> 元素，并合并它们的属性。移除每个元素中所有非结构化图表属性（如 OpenGraph、Schema 和 Twitter 之外的属性）。

- 对于每个有有效 src 属性的 <img> 标签，下载图像并随机裁剪，调整大小至 256x256 像素。使用 VQVAE-GAN 模型对图像进行标记，每个图像转换为 256 个标记。将这些标记转换为字符串，并用空格连接起来，插入到 src 属性中

  ![](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407190234335.png)

##### 3.2 SIZE HINTS

大小提示是一个用于引导模型在生成样本时的机制。通过在掩码token后插入一个表示掩码大小的概率估计，模型能够更好地预测掩码部分的内容。然而，在CM3模型中，发现这种方法降低了模型的最终困惑度和零样本性能。为了克服这个问题，可以在生成单个掩码时，隐式地给出大小提示，通过因果生成的方式达到类似的效果。(不太懂隐式给出大小这里什么意思)

##### 3.3 TRAINING

一些参数，不详细看了

| 参数                  | 125M Model                    | 800M Model                    | 2.7B Model (CM3-Medium)       | 13B Model (CM3-Large)         |
| --------------------- | ----------------------------- | ----------------------------- | ----------------------------- | ----------------------------- |
| **参数数量**          | 125M                          | 800M                          | 2.7B                          | 13B                           |
| **训练目标**          | 基础超参数设定                | 基础超参数设定                | 下游任务评估                  | 下游任务评估                  |
| **训练设备**          | 未指定                        | 未指定                        | 240 V100 GPU                  | 384 A100 GPU                  |
| **训练时长**          | 未指定                        | 未指定                        | 28天                          | 24天                          |
| **实现框架**          | PyTorch + fairseq + fairscale | PyTorch + fairseq + fairscale | PyTorch + fairseq + fairscale | PyTorch + fairseq + fairscale |
| **每GPU批处理大小**   | 8                             | 8                             | 8                             | 8                             |
| **最大token序列长度** | 2048                          | 2048                          | 2048                          | 2048                          |
| **学习率调度器**      | Polynomial decay              | Polynomial decay              | Polynomial decay              | Polynomial decay              |
| **预热更新次数**      | 1500                          | 1500                          | 1500                          | 1500                          |
| **梯度裁剪**          | 1.0                           | 1.0                           | 1.0                           | 1.0                           |
| **优化器**            | Adam                          | Adam                          | Adam                          | Adam                          |
| **Adam优化器参数**    | β1 = 0.9, β2 = 0.98           | β1 = 0.9, β2 = 0.98           |                               |                               |

##### 3.4 SCALING LAWS

CM3模型的训练设置和多模态特性引入了一些新的参数和复杂性。过去image和text都是有输入顺序的，但是使用HTML后图像位置是随机的，增加了模型在处理这些数据时的复杂性。（打破了token遵循Zipfian分布的假设）

#### 4.ZERO/FEW-SHOT PROMPTING

我觉得第四章是最fancy的一部分，这个模型在没有训练纯图像文档的时候竟然做生成图片等一系列任务

##### 4.1.1 UNCONDITIONAL IMAGE GENERATION

在测试模型的时候，如果输入<img 来让其生成token，这个模型会生成图片token。不过其经常会加上一段alt描述（倒也符合HTML），不过可以通过<img, src=让其直接生成图片，不再加上描述。其效果能与GAN相匹配

![image-20240719032822777](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407190328954.png)

​                                                                                                                      *很惊人的效果*

##### 4.1.2 IMAGE IN-FILLING

```
Infilling Prompt: <img src="{prefix}<mask:0>{postfix}"><mask:0>
```

这个地方举一例子就能理解了

```
正确结果应该为
<p>The quick brown fox</p>
<img src="https://example.com/image_12345.jpg">
<p>jumps over the lazy dog.</p>
假设 {prefix} 是 https://example.com/image_，而 {postfix} 是 .jpg。
我们输入为：
<p>The quick brown fox</p>
<img src="https://example.com/image_<mask:0>.jpg"><mask:0>
<p>jumps over the lazy dog.</p>
最后可能会输出
<p>The quick brown fox</p>
<img src="https://example.com/image_98765.jpg">
<p>jumps over the lazy dog.</p>，并生成一张图像
```

##### 4.2.1 CONDITIONAL IMAGE IN-FILLING

```
Conditional Infilling Prompt:
<img alt="Photo: {text}" src="{prefix}<mask:0>{postfix}"><mask:0>
这样生成的图像会有显著提升
```

![image-20240719034512984](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407190345760.png)

照片结果很令人震惊

##### 4.2.2 CONDITIONAL IMAGE GENERATION

```
Conditional Generation Prompt: <img alt="{prompt}
这样根据alt生成一张图片
```

##### 4.2.3 CAPTIONING

有两种方式：mask和causal

```
Captioning Masked Prompt #1: 
<img alt="Photo: A photo taken of<mask:0>" src="{image}">

Captioning Causal Prompt #1: 
<img src="{image}" title="Photo: A photo taken of
```

##### 4.3.1 ENTITY DISAMBIGUATION

CM3也可以充当语言模型

先解释一下entity disambiguation（实体消歧）：是指在自然语言处理中，识别并区分文本中提到的不同实体。特别是，当一个词或短语可能指代多个不同的实体时，消歧过程决定具体指代的是哪个实体。

比如在句子 "Manetho writes that these kings ruled from Memphis" 中，"Memphis" 可能指代埃及的孟菲斯（Memphis, Egypt）或美国田纳西州的孟菲斯（Memphis, Tennessee）。实体消歧的任务就是确定这里的 "Memphis" 指的是哪一个。

**example：**

```
Original: Manetho writes that these kings ruled from <a title="Memphis, Egypt">Memphis</a>
原始句子中包含一个超链接，链接的 title 属性明确指向 "Memphis, Egypt"（孟菲斯，埃及），而显示的文本是 "Memphis"
原本的 "Memphis" 可以指多个不同的地点，但通过 title 属性，我们知道它指的是埃及的孟菲斯。
Prompt: Manetho writes that these kings ruled from <a title="<mask:0>">Memphis</a>...<mask:0>
在提示句子中，我们将 title 属性中的 "Memphis, Egypt" 替换为 <mask:0>，并在句子末尾添加了 <mask:0>。
这个提示符告诉CM3模型需要填充 <mask:0> 位置的内容。目的是让模型预测出原本在 title 属性中的确切内容
Target: Manetho writes that these kings ruled from <a title="<mask:0>">Memphis</a>...<mask:0> Memphis, Egypt
在目标句子中，我们希望模型在 <mask:0> 位置生成 "Memphis, Egypt"
```

