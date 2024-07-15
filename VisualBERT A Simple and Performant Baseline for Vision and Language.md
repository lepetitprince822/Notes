## VisualBERT: A Simple and Performant Baseline for Vision and Language

在视觉语言模型中，有几种常见的任务：**VQA、VCR、NLVR、GRE、Regionto-Phrase Grounding**

​	**VQA：**输入视频或者图像等视觉数据，能够正确提供一个问题答案的任务（通常被认为是一种分类任务）

​	**VCR：**与VQA相似，只不过在提供一个问题答案的时候，还需要，还需要提供证明其正确的理由

​	**NLVR：**输入两张图片和一个文本描述，输出图形和文本描述之间的对应关系是否一致

​	**GRE：**给定一个文本参考，对一个图像区域进行定位（做法一般是为每个区域打一个分数，分数最高的当作定位区域）本论文没有提到这一种

​	**Regionto-Phrase Grounding：**在这篇论文中，使用Flickr30K这个数据集来完成这个任务，这个数据集中每个图片都有五句描述图像内容的话，模型用来识别每句话描述的图像区域

#### 1.Introduction：

这个模型使用了BERT作为自然语言处理的部分，另外使用Faster-RCNN作为模型处理视觉与语言任务的模型。并且他们使用了两种任务用作预训练：

1. 遮盖部分文本，模型通过上下文以及视觉语境来推测出被覆盖的词语
2. 确定提供的文本是否与图像匹配

其预训练使用的数据集是COCO

#### 2.Related Work

过去完成这些任务的模型都是由text-encoder，image feature extractor，multi-modal fusion module（多模态融合模块),answer classifier构成，都是为了特定任务。

在BERT的基础上有以下工作：VideoBERT将视频转换为图像与字幕，将其应用于transformer进行联合学习。ViLBERT这个模型学习了图像和文本的联合表示，但是其将图像与文本分别使用不同点transformer（使用Co-attention两个encoder融合）

#### 3.A JOINT REPRESENTATION MODEL FOR VISION AND LANGUAGE

##### 3.1 BACKGROUND

这个部分先介绍了BERT的基本工作，我也是在这篇文章才开始有些明白BERT究竟是在做什么的

BERT是为了做分类任务。BERT主要有两个训练任务：masked language modeling和next sentence prediction

masked language modeling这个任务训练时候，需要随机mask15%的单词，然后将喂入模型进行训练。但是BERT采用的是预训练+微调的方式，pre-training的时候进行“完形填空”），微调的时候做的是某种分类任务，输入中没有屏蔽词，为了让预训练和微调的数据能够“接近”，mask采用80-10-10方法，即80%概率mask，10%概率用一个随机标记替换这个地方，10%不做改变。**另外**，因为tokenization的时候用的是wordpiece，可能一个token不是一个完整单词，这个时候需要将要mask的这个部分所处的完整单词全部mask、

next sentence prediction这个任务是给定两句话作为输入，使用模型判断这两句话是不是上下句关系（二分类isNext，NotNext）

所以BERT的预训练是：先给出两句话，将两句话合并后随机mask，经过模型后，先通过softmax选出mask处最有可能的单词，然后判断这两个句子是否是上下句。

##### 3.2 VISUALBERT

BERT中有token embedding，segment embedding和position embedding，相似的，针对于图像模块

𝑓𝑜是用CNN计算得到的一个个bounding region的特征向量（𝑓𝑜 is a visual feature vector computed for a bounding region of the image by a convolutional neural network）

𝑓𝑠是对于𝑓𝑜的segment embedding，如果将图像的bounding region合起来的时候，需要由这个来知道他们是否属于一个bounding region（𝑓𝑠,a segment embedding indicating it is an image embedding as opposed to a text embedding,）

𝑓𝑝是针对于图像的位置编码（𝑓𝑝 is a position embedding used for aligning the order of bounding regions.）

为了方便理解，我将根据这个图示进行对应

![image-20240713162006290](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407131620804.png)

上面的图片分别对应着检测出的几个bounding regions（上衣，帽子，网球拍，网球等），这些就是𝑓𝑜，segment embedding用于区分不同点bounding region，position embedding用于给图像embedding加上位置编码

就这样，text和image就可以当作embedding送入同一个模型中，形成joint representation

##### 3.3 TRAINING VISUALBERT

VisualBERT也设置了与BERT很相似的预训练（Task-Agnostic Pre-Training），第一种是mask部分文字，依据图像来进行补全（MLM），第二种是选取两个caption，其中一个是对图像的描述，另一个50%概率是对图像的描述，50%是随机生成的caption

去COCO官网搜了一下，具体的数据是这样的

![image-20240715205559415](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407152110960.png)

一个图片会有五句不同的描述，但其意思都是相近的。我推测使用这种预训练方法是为了预训练的数据与微调的数据相接近（针对于VQA这种生成caption的任务，需要有错误的caption来学习）

除了这两种预训练，这篇文章还提出了针对于特定任务的预训练（Task-Specific Pre-Training）

##### 4 EXPERIMENT

这一部分是一些实验的说明，简要看了一下

VisualBERT：完整的模型

VisualBERT w/o Early Fusion: 让text和image在最后一个transformer layer再进行融合(应该是前面用独立的transformer，最后单独定义一个transformer(*VisualBERT but where image representations are not combined with the text in the initial Transformer layer but instead at the very end with a new Transformer layer. This allows us to test whether interaction between language and vision throughout the whole Transformer stack is important to performance.*)

VisualBERT w/o COCO Pre-training:不用COCO数据进行预训练（*isualBERT but where we skip task-agnostic pre-training on COCO captions. This allows us to validate the importance of this step.*）

![image-20240713164430009](https://le-petit-prince.oss-cn-beijing.aliyuncs.com/img/202407131644468.png)

这张图最后也证明了COCO Pre-training和Early Fusion的重要性