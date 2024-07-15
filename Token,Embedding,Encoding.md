## Token，Embedding，Encodeing

原来一直不清楚这三个的区别，今天跟好朋友聊到了这个，试着总结一下。

#### Token

token是将文本文本分割的最小单元，一般是一个单词，字符或者一个标点等

例如**Hello，How are you？**这句话，token可能就是‘**Hello**’ 、‘**,**’、‘**How**’、‘**are**’、‘**you**’、‘**？**’

tokenization是将文本划分成token的方式，中文又叫做”分词“，tokenization有如下几种方式：

1. **Whitespace Tokenization**（基于空格的分词）

   依据空格进行分割。这种方法简单快速，对于英文单词来说可以很好的进行划分，但是对于中文等复杂语言效果不好，而且处理不了复合词

2. **Punctuation Tokenization**（标点符号分词）

   不仅基于空格，还基于标点符号进行分割。不过其也不能很好处理复合词

3. **Dictionary-based Tokenization**（字典分词）

   基于预先定义好的词典进行分割，通常用于中文。但是对于没有记录的词（OOV）效果不好

   **e.g.**

   输入："我爱北京天安门"

   输出：["我", "爱", "北京", "天安门"]

4. **Subword Tokenization**（子词分词）

   将文本分割成子词，通常用于处理未记录词，常见有**BPE**和**WordPiece**。这种方法的缺点是会生成较多的子词，影响处理效率。

   **BPE：**

   ①先准备一个很大的训练语料，从中统计单词在语料中出现的频数，并设定词表大小

   ②将单词拆分为最小单元（例如英语单词就是拆成一个个字母），将其当作初始词表

   ③在语料中统计各个**相邻单元对**的频数，选取频数最高的单元对合并成新的单元，然后更新语料库将单元对，再根据语料库更新词表

   ④重复这几步一直到设定的词表大小或者最高频**相邻单元对**频数为1（另外已经是一整个单词）

   **e.g.**

   ①设语料库为{low lower lowest}

   ②拆分并形成初始词表{l,o,w,e,r,s,t,_}(其中__是结束标志)

   ③统计频数

   ```
   (l, o): 3
   (o, w): 3
   (w, _): 1
   (w, e): 2
   (e, r): 1
   (r, _): 1
   (e, s): 1
   (s, t): 1
   (t, _): 1
   ```

   其中(l,o)频数最大，进行合并

   更新后的语料库

   ```
   lo w _
   lo w e r _
   lo w e s t _
   ```

   根据语料库更新后的词表为

   ```
   lo, w, e, r, s, t, _
   ```

   ④不断迭代

   (lo, w): 3
   (w, _): 1
   (w, e): 2
   (e, r): 1
   (r, _): 1
   (e, s): 1
   (s, t): 1
   (t, _): 1

   更新语料库

   ```
   low _
   low e r _
   low e s t _
   ```

   更新此词表

   ```
   low, e, r, s, t, _
   ```

   (low, _): 1
   (low, e): 2
   (e, r): 1
   (r, _): 1
   (e, s): 1
   (s, t): 1
   (t, _): 1

   更新语料库

   ```
   low _
   lowe r _
   lowe s t _
   注意此时语料库中low不再发生变化
   ```

   更新词表

   ```
   low, lowe, r, s, t, _
   ```

   (low, _): 1
   (lowe, r): 1
   (r, _): 1
   (lowe, s): 1
   (s, t): 1
   (t, _): 1

   均为1，不进行更新。

   **WordPiece：**

   与BPE非常相似，只不过在合并单元对的时候，BPE看的是频率，而wordpiece选择能够		提升语言模型概率最大的相邻单元对

   **e.g.**

   语料库

   ```
   lower lowest lowering
   ```

   初始词表

   ```
   {"l", "o", "w", "e", "r", "i", "n", "g", "s", "t"}
   ```

   统计条件概率：

   ```
   (l, o): 3
   (o, w): 3
   (w, e): 3
   (e, r): 2
   (r, i): 1
   (i, n): 1
   (n, g): 1
   (e, s): 1
   (s, t): 1
   则
   P(o|l) = 3/3 = 1.0
   P(w|o) = 3/3 = 1.0
   P(e|w) = 3/3 = 1.0
   P(r|e) = 2/3 ≈ 0.67
   P(i|r) = 1/2 = 0.5
   ```

   l,o合并，然后进行更新语料库和词表

   以此类推

5. Character Tokenization（字符分词）

   简单粗暴，直接分为一个个字符，不过序列长度会大大增加，而且没有词级别的语义信息

#### Embedding

embedding是将token映射到向量空间的一种方式，中文一般翻译为“嵌入”（有的时候看中文文献会懵逼的词），embedding一般与当前的模型有关系

1. One-hot

   简单粗暴，将每个token都分为不同的01向量，保证每个token都不同（其实这也是其他embedding的基础）

   最后形式大概就是[0,1,0,0...0]

2. word embedding

   经典的为Word2Vec，其实最开始也是将token变化为one-hot，然后将one-hot作为向量输入，然后使用CBOW或者Skip-Gram等网络进行训练，得到简洁且具有语义相似性的embedding

BERT和GPT等这种模型也都是用了word embedding，但是他们他们还会加入position embeddings，Segment Embeddings等东西

#### Encoding

将向量转化为模型适合的矩阵等形式，在模型中进行运算（其作用一般就是将文本或者图像等特征提取出来）。常见的架构有Autoencoder，encoder-decoder