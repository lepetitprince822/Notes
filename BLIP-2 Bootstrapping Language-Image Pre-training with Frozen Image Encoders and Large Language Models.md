## BLIP-2 Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

#### 1. Introduction

VLP的视觉和语言部分都可以从现在的unimodal model获取（比如语言模型可以使用LLM），而且人家zero-shot能力也很强。另外为了减少计算以及防止模型遗忘，在训练的时候需要frozen这些模块。接着他提到了怎么让LLM模型在冻结的状态下能够学会图像特征（Frozen，Flamingo）

这篇文章提出了Q-former架构，用来对齐视觉与语言特征

#### 2. Related Work

##### 2.1. End-to-end Vision-Language Pre-training

这次叙述了VLP模型的架构种类，不过比上次多了一些：

- dual encoder：图像和文本通过各自独立的编码器进行编码（VisonBERT）

- fusion encoder：图像和文本通过一个融合编码器进行联合编码。（VisualBERT）
- encoder-decoder：
- unifined transformer：图像文本等多个模态特征输入同一个transformer（BLIP2，VL-BERT）

文章认为end to end方法的VLP模型需要image-text对数据来训练，预训练复杂，而且如果想使用现成的模型也不方便（我认为这里主要是想要说明其Q-Former的出色表现）

##### 2.2. Modular Vision-Language Pre-training

还有一种方法是利用现成的预训练模型，在VLP期间将其冻结。其中LiT就是冻结图像编码器（CLIP），另外也有使用冻结的语言编码器的（Frozen，Flamingo使用了cross attention将其放在LLM中，并使用了数十亿image-text对），但是BLIP2既使用了冻结图像编码器，也使用了LLM

#### 3. Method

##### 3.1. Model Architecture

Q-Former其实就是一个过渡模块，用来连接两个月预训练模型。其分为image transformer和text transformer两个部分。

![image-20240725171556464](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407251715523.png)

> ​                                                Q-Former架构，左侧为image transformer，右侧为text transformer

先看左侧image transformer，其中有一个可学习的queries，最开始的时候就是让Q作为输入做一次self attention，然后与image特征结合做cross attention（self-attention的作用一个是queries内部信息整合，能够提高表达能力吧）

右侧的text transformer通过image transformer对比学习来纠正queries与image融合的能力

初始化：Q-Former的自注意力层使用预训练的BERTbase权重初始化，而交叉注意力层则随机初始化。

参数量：Q-Former总共有188M参数。查询嵌入也被视为模型参数。

在实验中，使用了32个查询，每个查询的维度为768，用Z表示**查询的输出**表示，Z的尺寸为32 × 768，这比Frozen的图像特征的尺寸（ViT-L/14的257 × 1024）要小得多

##### 3.2. Bootstrap Vision-Language Representation Learning from a Frozen Image Encoder

与BLIP一样，训练的目标还是ITC，ITM，Image-Grounded Text Generation

**ITC：**

本质是让正样本与负样本之间进行对比学习学习，这里计算相似度是queries的输出Z与文本特征t计算相似度

由于对比学习学的就是Z和T之间的关系，所以在做self attention的时候，不能让其相互影响（参数相互共享证明text和query其实会进入同一个self attention模块），使用Uni-modal Self-Attention Mask，只让Q与Q相互计算，T与T相互计算

![image-20240725183144385](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407251831326.png)

**ITG：**

目的是给定输入图像，生成描述文本。需要query提取图像信息，通过自注意力层将信息传递给文本标记，用于生成文本。

因此使用Multi-modal Causal Self-Attention Mask，使得query只能互相访问，但不能访问文本标记，但是文本标记可以访问所有查询以及之前的文本标记

![image-20240725182844307](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407251828111.png)

> 行的QT代表的是作为query的标记，而列的QT的作为key的标记，所以右上角地方需要遮盖，而左下角不需要。因为T只能访问以前的文本标记，所以右下部分也有一块是遮盖的

**ITM：**

学习图像和文本表示之间的细粒度对齐。输出的查询嵌入（Z）捕捉多模态信息，输入到二分类器，预测图像-文本对是否匹配。

所以其不需要掩码

下面是整体的mask方法可视化

![image-20240725183436769](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407251834243.png)

##### 3.3. Bootstrap Vision-to-Language Generative Learning from a Frozen LLM

Q-Former最后与LLM进行连接，将query的输出Z通过全连接层投影到LLM前面

LLM也有两种形式，一种是只有LLM decoder的，这种Z充当了prompt的作用。另一种是既有encoder也有decoder的，这种Z充当了prefix(与SimVLM挺相似)

![image-20240725184941498](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407251849323.png)

##### 3.4. Model Pre-training

| **类别**         | **细节**                                                   |
| ---------------- | ---------------------------------------------------------- |
| **预训练数据**   | - 数据集：COCO, Visual Genome, CC3M, CC12M, SBU, LAION400M |
| **数据集总计**   | - 129M 图像                                                |
| **合成标题方法** | - 使用 CapFilt 方法                                        |
| **生成标题**     | - 使用 BLIP large 模型生成 10 个标题                       |
| **标题排序**     | - 使用 CLIP ViT-L/14 模型根据图像-文本相似性进行排序       |
| **保留标题**     | - 每张图像保留前两名标题，预训练时随机抽取一个             |

| **冻结的图像编码器** | **模型**                                 |
| -------------------- | ---------------------------------------- |
| **ViT-L/14**         | - 来自 CLIP                              |
| **ViT-g/14**         | - 来自 EVA-CLIP                          |
| **处理方式**         | - 移除最后一层，使用倒数第二层的输出特征 |

| **冻结的语言模型** | **模型**                  |
| ------------------ | ------------------------- |
| **OPT**            | - 适用于解码器模型        |
| **FlanT5**         | - 适用于编码器-解码器模型 |

| **优化器**     | **参数**                                    |
| -------------- | ------------------------------------------- |
| **AdamW**      | - β1 = 0.9, β2 = 0.98, 权重衰减 = 0.05      |
| **学习率衰减** | - 余弦衰减，峰值学习率 1e-4，线性预热 2k 步 |
| **最低学习率** | \- 第二阶段为 5e-5                          |