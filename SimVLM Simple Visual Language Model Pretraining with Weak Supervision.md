## SimVLM: Simple Visual Language Model Pretraining with Weak Supervision

这篇文章针对VLM预训练比较复杂（需要针对微调任务进行特定的预训练），提出了一个较为广泛的SimVLM框架，为现有的VLP框架提供了一个替代方案，能够有效降低VLP任务的复杂度。

#### Introduction

开头先说BERT等提供的预训练+微调方式被广泛采用，但是GPT3等模型展示出了无需微调也可以获得很好的效果。

这篇文章说出了曾经VLM模型都是怎么使用的，比如设计各种图像语言联合的任务（VQA），接着又说曾经的VLM训练数据都是从两个不同地方获取：先去训练一个目标检测模型找到图像中的ROI（region of interest，其实这个就是RCNN类工作的那个过程），接着再使用image-text的数据对VLM模型(基于transformer)进行训练。另外还增添了一些辅助loss函数

> 这里让我对VisualBERT的理解加深了
>
> ![image-20240716012109207](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407160121036.png)
>
> 还是这张图，原来一直在想论文里仅仅提了一句的Faster-RCNN是什么角色，现在才发现图像都是先通过Faster-RCNN等模型检测后生成ROI，然后再将ROI作为输入与text合并
>
> <img src="https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407161459449.png" alt="image-20240716012214748"  />

这种流程让模型缺少zero-shot的能力，而且这些流程也针对某些特定任务，不是通用的pretraining-finetuning方法（这里我觉得可能是想仿照GPT3做一个通用的VLM模型）。所以文章提出了SimVLM，这个模型不需要特殊的预训练也能实现很好的性能，也不需要目标检测模块

#### Related Work

之前VLM的各种尝试

#### SimVLM

##### 3.1 BACKGROUND

对于MLM，x是语言序列，xm是被mask的部分，被mask后的序列为x\m，则：
$$
\mathcal{L}_{\text{MLM}}(\theta) = - \mathbb{E}_{\mathbf{x} \sim \mathcal{D}} \left[ \log P_{\theta} (\mathbf{x}_m \mid  \mathbf{x}_{\setminus m} \right)]\\
  \left[P_{\theta} (\mathbf{x}_m \mid \mathbf{x}_{\setminus m}) \right]相当于在已知遮盖地方的情况下，猜测这个词为正确mask词的概率
$$
对于单向语言模型(LM)，一般使用直接最大化序列 X的似然函数，具体如下：
$$
\mathcal{L}_{\text{LM}}(\theta) = - \mathbb{E}_{\mathbf{x} \sim \mathcal{D}} \left[ \log P_{\theta} (\mathbf{x}) \right] = - \mathbb{E}_{\mathbf{x} \sim \mathcal{D}} \left[ \sum_{t=1}^{T} \log P_{\theta} (\mathbf{x}_{t} \mid \mathbf{x}_{<t}) \right]\\
其中 \left[ \sum_{t=1}^{T} \log P_{\theta} (\mathbf{x}_{t} \mid \mathbf{x}_{<t}) \right]代表从第一个单词开始生成后一个单词的概率\\
e.g.句子为I\quad love\quad NLP，则概率为\\
P(\mathbf{x}) = P(\text{I}) \cdot P(\text{love} \mid \text{I}) \cdot P(\text{NLP} \mid \text{I}, \text{love})
$$
文章认为LM损失函数具有强大的生成能力，具有不进行微调的便实现zero-shot学习（*Compared with MLM, the LM pretraining has also been shown to be highly effective for multiple NLP tasks (Radford et al., 2018). More importantly, it facilitates the model with strong generation capability that enables text induced zero-shot generalization without finetuning (Brown et al., 2020).*）

##### 3.2 PROPOSED OBJECTIVE: PREFIX LANGUAGE MODELING

SimVLM采用的是PrefixLM，其具体公式如下：
$$
\mathcal{L}_{\text{PrefixLM}}(\theta) = -\mathbb{E}_{\mathbf{x} \sim \mathcal{D}} \left[ \log P_{\theta}(\mathbf{x}_{\geq T_p} \mid \mathbf{x}_{< T_p}) \right] = -\mathbb{E}_{\mathbf{x} \sim \mathcal{D}} \left[ \sum_{t=T_p}^{T} \log P_{\theta}(\mathbf{x}_t \mid \mathbf{x}_{[T_p,t]}, \mathbf{x}_{< T_p}) \right]\\
\mathbf{x}_{\geq T_p} 代表的是后缀序列，\mathbf{x}_{< T_p}代表的是前缀序列\\
e.g.句子为The\quad quick\quad brown\quad fox\quad jumps\quad over\quad the\quad lazy\quad dog\\
假设前缀为4，即The\quad quick\quad brown\quad fox，\\
那么
\begin{align*}
P(\mathbf{x}) &= P(\text{jumps} \mid \text{The, quick, brown, fox}) \\
&\times P(\text{over} \mid \text{The, quick, brown, fox, jumps}) \\
&\times P(\text{the} \mid \text{The, quick, brown, fox, jumps, over}) \\
&\times P(\text{lazy} \mid \text{The, quick, brown, fox, jumps, over, the}) \\
&\times P(\text{dog} \mid \text{The, quick, brown, fox, jumps, over, the, lazy})
\end{align*}
$$
不过对于VLM类，前缀一般是图像(可能加text也可能不加text)，文章提到这种方法能够利用上下文的信息，我认为是图像中包含着全部的信息，也包括将要生成的单词，所以是上下文。(*Intuitively, images can be considered as prefix for their textual descriptions as they often appear before text in a web document. Therefore, for a given image-text pair, we prepend image feature sequence of length Ti to the text sequence, and enforce the model to sample a prefix of length Tp ≥ Ti to calculate LM loss on text data only*)

##### 3.3 ARCHITECTURE

文章将image转化为sequence采用的是patch思想，将R^(H×W×C)转化为R^(Ti×D)的形式，其中Ti = HW/P^2，经典的patch思想。不过与ViT不同的是，ViT使用线性变换将patch变换为sequence，而SimVLM使用Resnet的前几个卷积层对patch进行变换，这样有了更多的图像细节和上下文信息。对于text使用token embedding操作，最后对text与image加入1D位置编码，对图像又额外加入2D的relative position embedding

![image-20240717004259721](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407170043662.png)

这篇文章的架构图也是非常清晰，从图中可以看出这里的前缀既包含图像也包含文本，分别使用不同的方式处理，最后送入transformer的encoder中

##### 3.4 DATASETS

因为缺少前期目标检测模型生成ROI，因此数据集使用large-scale noisy image-text data.具有zero-shot泛化能力，另外又补充了一些纯文本训练数据

#### EXPERIMENTS

![image-20240717101153464](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407171012771.png)

​                                                                                                            *总体情况*

![image-20240717101307122](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407171013539.png)

​                                                                                            *我比较感兴趣的zero-shot部分*
