

## BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

BLIP提出了一种可以支持多种任务的VLP模型，另外其含有过滤器过滤数据噪声，可以提高模型的性能（他在摘要说很多模型都是通过增加数据量来提升性能的，其数据集本身不高，噪声较多）

#### Introduction

目前模型的两个问题：

1. 模型大部分都是encoder-based或者encoder-decoder架构，encoder架构不适合做文本生成工作（text generation tasks），而encoder-decoder架构不适合/从来没有应用于文本检索(image-text retrieval)
2. 目前模型取得好效果都是从网上找的image-text pair，但是这种含噪声的web text不适合提升模型性能。

基于此问题，BLIP提出了其解决方案：

1. Multimodal mixture of Encoder-Decoder (MED)，其可以做图文对比学习，图文匹配，根据图片进行语言建模三种任务（*imagetext contrastive learning, image-text matching, and imageconditioned language modeling.*）
2. Captioning and Filtering (CapFilt)，captioning是根据image生成captions，filtering用来对最初的web text以及生成的captions进行过滤

#### Related Work

##### 2.1. Vision-language Pre-training

主要还是叙述encoder-decoder，encoder-only架构的问题以及其提出MED模型的优势（今天看到了一个论述decoder-only和encoder-decoder的博客，是从低秩角度叙述的，[为什么现在的LLM都是Decoder only的架构](https://www.zhihu.com/question/588325646/answer/2940298964)）

##### 2.2. Knowledge Distillation

这里我先简要了解了一下知识蒸馏的知识。

Knowledge Distillation：

教师模型是一个已经训练好的强大模型，学生模型是一个较小或较简单的模型。通过让学生模型模仿教师模型的预测，可以提高学生模型的性能。

学生模型既要接受数据集中的hard label，同时也接受教师模型的soft label，所以其损失函数需要有改变
$$
L = \text{CE}(y, p) + \alpha \text{CE}(q, p)\\
其中y是真实类别，p是学生模型预测概率，q是教师模型预测概率\\
{CE}(q, p) = - \sum_{i} q_i \log(p_i)\\
为了不让softmax后类别数值相差过大，加入了temperature（我感觉跟注意力计算时候的\sqrt{q}有共同之处）\\
soft变为了q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}\\
其中z_i是教师模型的logit，T是温度\\
T越高，softmax的output\quad probability\quad distribution越趋于平滑，其分布的熵越大，负标签携带的信息会被相对地放大，模型训练将更加关注负标签。
$$
自蒸馏是知识蒸馏的一种特殊情况，在这种情况下，教师模型和学生模型的规模相同。

##### 2.3. Data Augmentation

这里文章提到了其synthetic captions等同于一种数据增强

#### 3. Method

##### 3.1. Model Architecture

- Unimodal encoder：其中有image encoder和text encoder，image encoder使用ViT架构，text encoder基于BERT架构
- Image-grounded text encoder：经过Bi self attention后再经过cross attention，最后再经过FFN
- Image-grounded text decoder：经过casual self attention后经过cross attention

Bi Self-Att和Casual Self-Att：Bi是双向自注意力机制，casual其实就是加了mask，使其只能关注上文而不能关注下文信息

cross attention：通过两个序列进行注意力机制，Q来自一个序列，KV来自另外一个序列，在VLM中就是image作为一个序列，text作为另外一个序列，做注意力计算

##### 3.2. Pre-training Objectives

在预训练期间训练了三种任务，对应了三种目标函数

- Image-Text Contrastive Loss：
  $$
  \mathcal{L}_{\text{ITC}} = - \frac{1}{N} \sum_{i=1}^{N} \left( \log \frac{\exp(\text{sim}(\mathbf{v}_i, \mathbf{t}_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{v}_i, \mathbf{t}_j) / \tau)} + \log \frac{\exp(\text{sim}(\mathbf{t}_i, \mathbf{v}_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{t}_i, \mathbf{v}_j) / \tau)} \right)\\
   \log \frac{\exp(\text{sim}(\mathbf{v}_i, \mathbf{t}_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{v}_i, \mathbf{t}_j) / \tau)}是图像特征与对应文本特征相似度在文本特征中的相对大小\\
   \log \frac{\exp(\text{sim}(\mathbf{t}_i, \mathbf{v}_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{t}_i, \mathbf{v}_j) / \tau)} 是文本特征与图像特征相似度在图像特征中的相对大小
  $$
  上文提到的自蒸馏应该给是用到了这里，另外文中提到的动量编码应该也是在这里（*where a momentum encoder is introduced to produce features, and soft labels are created from the momentum encoder as training targets to account for the potential positives in the negative pairs*）

- Image-Text Matching Loss ：

  ITM损失是一种二分类损失，旨在学习图像和文本之间的多模态表示，以捕捉视觉和语言之间的细粒度
  $$
  \mathcal{L}_{\text{ITM}} = - \frac{1}{N} \sum_{i=1}^{N} \left( y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right)\\
  其中y_i表示第i个样本的真实标签（1代表匹配，0代表不匹配）\\
  \hat y_i表示第i个样本的预测概率
  $$

- Language Modeling Loss ：

  比较基本是语言建模损失(GPT用的就是这个)
  $$
  \mathcal{L}_{\text{LM}} = - \frac{1}{T} \sum_{t=1}^{T} \log p(y_t | y_{1:t-1}, \mathbf{v})\\
  其中T是序列长度
  $$
  ![image-20240725013338424](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407250133358.png)

  > 这是MED的架构图吗，其中颜色相同的层是共享参数的

  ##### 3.3. CapFilt

  这里一个图就讲清楚了

  ![image-20240725013251155](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407250133575.png)

> 首先，数据分为web data和人工标注的human-annotated data，在进入MED时，先使用Ih和Th对模型进行微调，encoder通过ITC和ITM学会了筛选能力（image和text的匹配能力），decoder学会了根据image生成caption。decoder生成的image-text与web text筛选的以及人工标注的数据作为总的data。往后循环这个操作，类似于统计中的bootstrapping

#### 4. Experiments and Discussions

从论文中没有找到关于image-grounded text encoder和image-grounded text decoder的预训练模型是什么

| 参数/详细信息        | 描述                                                         |
| -------------------- | ------------------------------------------------------------ |
| 框架                 | PyTorch (Paszke et al., 2019)                                |
| 预训练硬件           | 两个16-GPU节点                                               |
| image encoder        | ViT 预训练于 ImageNet (Touvron et al., 2020; Dosovitskiy et al., 2021) |
| text encoder         | BERTbase (Devlin et al., 2019)                               |
| ViT 变体             | ViT-B/16 和 ViT-L/16                                         |
| 默认使用的 ViT 变体  | ViT-B                                                        |
| 预训练轮数           | 20 epochs                                                    |
| 批次大小             | 2880 (ViT-B) / 2400 (ViT-L)                                  |
| 优化器               | AdamW (Loshchilov & Hutter, 2017)                            |
| 权重衰减             | 0.05                                                         |
| 学习率               | 预热到 3e-4 (ViT-B) / 2e-4 (ViT-L)，然后以 0.85 的速率线性衰减 |
| 预训练图像裁剪分辨率 | 224 × 224                                                    |
| 微调图像分辨率       | 384 × 384                                                    |
| 预训练数据集         | 与 Li et al. (2021a) 相同，包括 14M 张图像和以下数据集：     |
| - 人工标注数据集     | COCO 和 Visual Genome (Krishna et al., 2017)                 |
| - 网络数据集         | Conceptual Captions (Changpinyo et al., 2021), Conceptual 12M (Changpinyo et al., 2021), SBU captions (Ordonez et al., 2011) |
| 额外实验数据集       | LAION (Schuhmann et al., 2021)，包含 115M 张图像及更多噪声文本 |