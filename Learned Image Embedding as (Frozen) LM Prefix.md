## Learned Image Embedding as (Frozen) LM Prefix

过去VLM在训练时，会根据loss调整模型的参数（包括visual和language，还有一种说法是过去的一些视觉语言模型视觉部分和语言部分的参数会分别进行调整和优化。）但是Frozen (Tsimpoukelli et al. 2021) 和 ClipCap (Mokady, Hertz & Hertz, 2021) 中，仅更新视觉模块的参数，使得生成的图像embedding能够与预训练的、冻结的语言模型兼容。

### Multimodal Few-Shot Learning with Frozen Language Models

#### 1.Introduction

文章指出，当前的方法通常需要大量标注数据来训练多模态模型，但实际应用中获取大量标注数据往往非常困难。为了应对这一问题，作者提出了一种基于冻结语言模型的方法，通过利用预训练的语言模型，减少对标注数据的依赖，实现多模态少样本学习的目标。

#### 2.Related Work

Prefix Tuning和Prompt Tuning是两种用于调整预训练语言模型的方法，它们都有助于在不改变模型大部分权重的情况下适应特定任务。

**Prefix Tuning：**通过学习任务特定的“前缀”（prefix）来调整语言模型的方法。这个前缀是一个连续嵌入向量，添加到输入序列的前面，起到提示的作用。

```
e.g.
1.对于一个预训练模型（如GPT3），我们希望它能生成电影评论，但不希望对整个模型进行微调（耗费大量的计算资源和时间），所以我们给每个输入序列初始化一个固定长度的前缀向量
2.将前缀向量附加在输入文本序列的前面。例如，如果输入文本是“这部电影太精彩了！”，前缀向量是一个长度为5的向量
[prefix_1, prefix_2, prefix_3, prefix_4, prefix_5, 这, 部, 电, 影, 太, 精, 彩, 了, ！]
3.在一个小的任务特定数据集（如一组电影评论数据集）上训练这些前缀向量，使得模型在生成输出时能够更好地适应电影评论的风格和内容。此时，只更新前缀向量的参数，语言模型的其他权重保持不变。
4.在推理时，根据此时的prefix，输入
[prefix_1, prefix_2, prefix_3, prefix_4, prefix_5, 请, 为, 这, 部, 新, 电, 影, 生, 成, 一, 条, 评, 论, ：]
5.得到输出
“这部电影是一部视觉盛宴，剧情紧凑，演员表现出色，绝对值得一看！”
```

这种方法相当于为了节省计算资源，只通过prefix这个部分进行微调，确实很巧妙。

**Prompt Tuning：**通过设计特定的文本提示（prompt）来引导预训练语言模型产生所需的输出。这些提示可以是手工设计的文本，也可以是通过训练生成的嵌入向量。（这里大概就是prompt engineering）

Frozen使用的是prefix tuning。

#### 3.The Frozen Method

##### 3.1 Architecture

Frozen使用的是动态prefix，上面讲的prefix长度固定，且只能针对于某一种特定任务。而动态prefix针对于输入进行改变，输入不同prefix就不同，这样可以支持多种任务和不同类型的输入数据（视觉、语言，不过frozen应该是针对图像设置prefix）

其预训练模型使用SentencePiece tokenizer将文本分解为一系列离散的token，词汇表大小为32,000。模型通过标准的最大似然目标在大规模互联网文本数据集上进行预训练（C4数据集，包含70亿参数）。

视觉编码器基于NF-ResNet-50架构。将原始图像转换为供Transformer使用的连续序列，使用NF-ResNet的全局池化层的最终输出向量。

**视觉prefix生成过程：**首先根据token的维度D生成一个与token同维的embedding，接着进入线性层将其转化为D×n维度（n代表的是prefix个数吧，文章认为n=2最好，但我觉得这样prefix的长度也固定了）

##### 3.2 Training

总结一下这个模型的流程吧

训练目的：使用配对的图像-标题数据（来自Conceptual Captions数据集）训练视觉编码器参数 ϕ。目标是最大化给定图像 x生成标题 y的条件概率
$$
\begin{align*}
\log p_{\theta, \phi}(\mathbf{y} | x) &= \sum_{l} \log p_{\theta, \phi}(y_l | x, y_1, y_2, \ldots, y_{l-1}) \\
&= \sum_{l} f_{\theta}(i_1, i_2, \ldots, i_n, t_1, t_2, \ldots, t_{l-1})_{y_l}
\end{align*}
$$

1. 先预训练一个语言模型，然后冻结其参数θ 保持冻结状态，不进行更新。
2. 然后使用视觉编码器，输入图像，输出视觉前缀。将visual prefix放在文本序列前面，形成[ i1 ,i2 ,t1 ,t2 , ..., tl−1 ]
3. 根据目标函数训练模型（使用朴实无华的SGD），只改变视觉编码器的参数 ϕ，
4. 加入相对位置编码，如果输入多个图像也能很好完成任务

![image-20240720022212432](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407200222347.png)

> ​                                                                                   *这个图就很清晰的表示出了上面的步骤*

#### 4 Experiments

Frozen scratch是从零开始训练整个模型，视觉编码器和语言模型都没有预训练权重。模型需要从头开始学习如何回答问题。

Frozen finetuned是在Frozen基础上，还会根据视觉问答任务数据微调语言模型。语言模型权重会更新，以更好地理解视觉问答任务。

Frozen train-blind是在训练过程中使用黑色图像输入，视觉编码器仍然被训练，但没有实际的视觉信息。这里主要是测试模型是否能在没有视觉输入的情况下，通过语言模型的知识回答问题。

经过实验，发现scratch和finetuned不能很好的完成任务，另外视觉输入不怎么影响知识问答。

![image-20240720230659399](Learned Image Embedding as (Frozen) LM Prefix.assets/image-20240720230659399.png)

> ​                                                                                         *左图为VQA，右图为OKVQA*