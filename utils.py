# encoding:utf8
import os
import time
from datetime import timedelta
import numpy as np
from collections import Counter
import tensorflow.contrib.keras as kr


def time_diff(start_time):
    """当前距初始时间已花费的时间"""
    end_time = time.time()
    diff = end_time - start_time
    return timedelta(seconds=int(round(diff)))


def batch_index(length, batch_size, is_shuffle=True):
    """
    生成批处理样本序列id.
    :param length: 样本总数
    :param batch_size: 批处理大小
    :param is_shuffle: 是否打乱样本顺序
    :return:
    """
    index = [idx for idx in range(length)]
    if is_shuffle:
        np.random.shuffle(index)
    for i in range(int(np.ceil(length / batch_size))):
        yield index[i * batch_size:(i + 1) * batch_size]


def cat_to_id(classes=None):
    """
    :param classes: 分类标签；默认为pos, neg
    :return: {分类标签：id}
    """
    if not classes:
        classes = ['pos', 'neg']
    cat2id = {cat: idx for (idx, cat) in enumerate(classes)}
    return classes, cat2id


def load_corpus(path, word2id, max_sen_len=50):
    """
    :param path: 样本语料库的文件
    :return: 文本内容contents，以及分类标签labels(onehot形式)
    """
    _, cat2id = cat_to_id()
    contents, labels = [], []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()
            label = sp[0]
            content = [word2id.get(w, 0) for w in sp[1:]]
            content = content[:max_sen_len]
            if len(content) < max_sen_len:
                content += [word2id['_PAD_']] * (max_sen_len - len(content))
            labels.append(label)
            contents.append(content)
    counter = Counter(labels)
    print('总样本数为：%d' % (len(labels)))
    print('各个类别样本数如下：')
    for w in counter:
        print(w, counter[w])

    contents = np.asarray(contents)
    labels = [cat2id[l] for l in labels]
    labels = kr.utils.to_categorical(labels, len(cat2id))

    return contents, labels


def build_word2id(file):
    """
    :param file: word2id保存地址
    :return: None
    """
    word2id = {'_PAD_': 0}
    path = [os.path.join('./data/', w) for w in os.listdir('./data/')]

    for _path in path:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                for word in sp[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)

    with open(file, 'w', encoding='utf-8') as f:
        for w in word2id:
            f.write(w+'\t')
            f.write(str(word2id[w]))
            f.write('\n')

# build_word2id('./data/word_to_id.txt')


def load_word2id(path):
    """
    :param path: word_to_id词汇表路径
    :return: word_to_id:{word: id}
    """
    word_to_id = {}
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()
            word = sp[0]
            idx = int(sp[1])
            if word not in word_to_id:
                word_to_id[word] = idx
    return word_to_id


def build_word2vec(fname, word2id, save_to_path=None):
    """
    :param fname: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    import gensim
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    return word_vecs

# word2id = load_word2id('./data/word_to_id.txt')
# w2v = build_word2vec('./data/wiki_word2vec_50.bin', word2id, save_to_path='./data/corpus_word2vec.txt')


def load_corpus_word2vec(path):
    """加载语料库word2vec词向量,相对wiki词向量相对较小"""
    word2vec = []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = [float(w) for w in line.strip().split()]
            word2vec.append(sp)
    return np.asarray(word2vec)

