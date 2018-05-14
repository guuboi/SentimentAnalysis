import numpy as np
import jieba as jb
import tensorflow as tf
from CONFIG import CONFIG
from utils import cat_to_id, load_word2id
from TextCNN import TextCNN


def sent_to_id(inputs):
    """
    将语句进行分词，然后将词语转换为word_to_id中的id编码
    :param inputs: 句子：列表的形式
    :return: 用id表征的语句
    """
    sentences = []
    cut_sents = [jb.cut(w) for w in inputs]
    config = CONFIG()
    word2id = load_word2id(config.word2id_path)

    for cut_sent in cut_sents:
        sentence = [word2id.get(w, 0) for w in cut_sent]
        sentence = sentence[:config.max_sen_len]
        if len(sentence) < config.max_sen_len:
            sentence += [word2id['_PAD_']] * (config.max_sen_len - len(sentence))

        sentences.append(sentence)

    return np.asarray(sentences)


def predict(x, label=False, prob=False):
    """
    :param x: 语句列表
    :param label: 是否以分类标签的形式：pos或neg输出。默认为：0/1
    :param prob: 是否以概率的形式输出。
    :return: 情感预测结果
    """
    if label and prob:
        raise Exception("label和prob两个参数不能同时为True!")

    x = sent_to_id(x)
    config = CONFIG()
    model = TextCNN(config)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(config.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        y = model.predict(sess, x, prob=prob)

    if label:
        cat, _ = cat_to_id()
        y = [cat[w] for w in y.tolist()]
    return y


if __name__ == '__main__':
    test = ['噪音大、车漆很薄', '性价比很高，价位不高，又皮实耐用。']
    print(predict(test, label=True))
