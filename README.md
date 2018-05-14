# 基于TextCNN汽车行业评论文本的情感分析

使用卷积神经网络对汽车行业评论文本进行情感分析。


## 数据集

爬取汽车之家车主口碑评论文本，抽取口碑中最满意以及最不满意评论文本，分别作为正向情感语料库和负向情感语料库。

语料库基本信息如下：

- 训练集(data/ch_auto_train.txt): 40000 = 20000(pos) + 20000(neg)
- 验证集(data/ch_auto_dev.txt): 10000 = 5000(pos) + 5000(neg)
- 测试集(data/ch_auto_test.txt): 20000 = 10000(pos) + 10000(neg)


## 预处理

`utils.py`为数据的预处理代码。

- `cat_to_id()`: 分类类别以及id对应词典{pos:0, neg:1};
- `build_word2id()`: 构建词汇表并存储，形如{word: id};
- `load_word2id()`: 加载上述构建的词汇表;
- `build_word2vec()`: 基于预训练好的word2vec构建训练语料中所含词语的word2vec;
- `load_corpus_word2vec()`: 加载上述构建的word2ve;
- `load_corpus()`: 加载语料库：train/dev/test;
- `batch_index()`: 生成批处理id序列。

经过数据预处理，数据的格式如下：

- x: [1434, 5454, 2323, ..., 0, 0, 0]
- y: [0, 1]

x为构成一条语句的单词所对应的id。
y为onehot编码: pos-[1, 0], neg-[0, 1]。

## CNN卷积神经网络

### 配置项

CNN可配置的参数如下所示，在`cnn_model.py`中。

```python
 class CONFIG():
    update_w2v = True           # 是否在训练中更新w2v
    vocab_size = 37814          # 词汇量，与word2id中的词汇量一致
    n_class = 2                 # 分类数：分别为pos和neg
    max_sen_len = 75            # 句子最大长度
    embedding_dim = 50          # 词向量维度
    batch_size = 100            # 批处理尺寸
    n_hidden = 256              # 隐藏层节点数
    n_epoch = 10                # 训练迭代周期，即遍历整个训练样本的次数
    opt = 'adam'                # 训练优化器：adam或者adadelta
    learning_rate = 0.001       # 学习率；若opt=‘adadelta'，则不需要定义学习率
    drop_keep_prob = 0.5        # dropout层，参数keep的比例
    num_filters = 256           # 卷积层filter的数量
    kernel_size = 3             # 卷积核的尺寸；nlp任务中通常选择2,3,4,5
    print_per_batch = 100       # 训练过程中,每100词batch迭代，打印训练信息
    save_dir = './checkpoints/' # 训练模型保存的地址
    ...
```

### CNN模型

具体参看`TextCNN`的实现。代码cnn()部分主要参考gaussic的[cnn_model.py](https://github.com/gaussic/text-classification-cnn-rnn/blob/master/cnn_model.py)。

### 训练与验证

`train_and_eva.py`中的train()进行训练。


```
加载word2vec==========================
加载train语料库========================
总样本数为：40000
各个类别样本数如下：
pos 20000
neg 20000
加载dev语料库==========================
总样本数为：10000
各个类别样本数如下：
pos 5000
neg 5000
加载test语料库=========================
总样本数为：20000
各个类别样本数如下：
pos 10000
neg 10000
Training and evaluating...
Epoch: 1
Iter:      0, Train Loss:   0.71, Train Acc:  51.00%, Val Loss:   0.86, Val Acc:  49.96%, Time: 0:00:04 *
Iter:    100, Train Loss:   0.29, Train Acc:  89.00%, Val Loss:   0.26, Val Acc:  89.16%, Time: 0:04:37 *
Iter:    200, Train Loss:   0.22, Train Acc:  93.00%, Val Loss:    0.2, Val Acc:  91.85%, Time: 0:09:05 *
Iter:    300, Train Loss:  0.082, Train Acc:  96.00%, Val Loss:   0.17, Val Acc:  93.26%, Time: 0:13:26 *
Epoch: 2
Iter:    400, Train Loss:   0.16, Train Acc:  96.00%, Val Loss:   0.17, Val Acc:  93.19%, Time: 0:17:52 
Iter:    500, Train Loss:   0.11, Train Acc:  97.00%, Val Loss:   0.17, Val Acc:  93.51%, Time: 0:22:11 *
Iter:    600, Train Loss:   0.16, Train Acc:  97.00%, Val Loss:   0.15, Val Acc:  94.22%, Time: 0:26:36 *
Iter:    700, Train Loss:   0.15, Train Acc:  91.00%, Val Loss:   0.15, Val Acc:  94.05%, Time: 0:30:54 
Epoch: 3
Iter:    800, Train Loss:   0.11, Train Acc:  95.00%, Val Loss:   0.15, Val Acc:  94.13%, Time: 0:35:13 
Iter:    900, Train Loss:  0.058, Train Acc:  97.00%, Val Loss:   0.16, Val Acc:  94.33%, Time: 0:39:37 *
Iter:   1000, Train Loss:  0.048, Train Acc:  98.00%, Val Loss:   0.15, Val Acc:  94.33%, Time: 0:43:53 
Iter:   1100, Train Loss:  0.054, Train Acc:  97.00%, Val Loss:   0.16, Val Acc:  94.10%, Time: 0:48:21 
Epoch: 4
Iter:   1200, Train Loss:  0.065, Train Acc:  96.00%, Val Loss:   0.16, Val Acc:  94.52%, Time: 0:52:43 *
Iter:   1300, Train Loss:  0.056, Train Acc:  97.00%, Val Loss:   0.17, Val Acc:  94.55%, Time: 0:57:09 *
Iter:   1400, Train Loss:  0.016, Train Acc: 100.00%, Val Loss:   0.17, Val Acc:  94.40%, Time: 1:01:30 
Iter:   1500, Train Loss:    0.1, Train Acc:  97.00%, Val Loss:   0.16, Val Acc:  94.90%, Time: 1:05:49 *
Epoch: 5
Iter:   1600, Train Loss:  0.021, Train Acc:  99.00%, Val Loss:   0.16, Val Acc:  94.28%, Time: 1:10:00 
Iter:   1700, Train Loss:  0.045, Train Acc:  99.00%, Val Loss:   0.18, Val Acc:  94.40%, Time: 1:14:16 
Iter:   1800, Train Loss:  0.036, Train Acc:  98.00%, Val Loss:   0.21, Val Acc:  94.10%, Time: 1:18:36 
Iter:   1900, Train Loss:  0.014, Train Acc: 100.00%, Val Loss:    0.2, Val Acc:  94.18%, Time: 1:22:59
```

在验证集上的最佳效果为94.90%。

### 测试

`train_and_eva.py`中的test()进行测试。

```
INFO:tensorflow:Restoring parameters from ./checkpoints/sa-model
Precision, Recall and F1-Score...
             precision    recall  f1-score   support
        pos       0.96      0.96      0.96     10000
        neg       0.96      0.96      0.96     10000
avg / total       0.96      0.96      0.96     20000

Confusion Matrix...
[[9597  403]
 [ 449 9551]]
```

在测试集上的准确率达到了95.74%，且各类的precision, recall和f1-score都超过了95%。


## 预测

`predict.py`中的predict()进行预测。

```python
 >> test = ['噪音大、车漆很薄', '性价比很高，价位不高，又皮实耐用。']
 >> print(predict(test, label=True))
INFO:tensorflow:Restoring parameters from ./checkpoints/sa-model
['neg', 'pos']
```




