# encoding:utf8
import numpy as np
import tensorflow as tf
from sklearn import metrics
from utils import load_word2id, load_corpus_word2vec, load_corpus, cat_to_id
from TextCNN import TextCNN
from CONFIG import CONFIG


def train():
    config = CONFIG()
    print('加载word2id===========================')
    word2id = load_word2id(config.word2id_path)
    print('加载word2vec==========================')
    word2vec = load_corpus_word2vec(config.corpus_word2vec_path)
    print('加载train语料库========================')
    x_tr, y_tr = load_corpus(config.train_path, word2id, max_sen_len=config.max_sen_len)
    print('加载dev语料库==========================')
    x_val, y_val = load_corpus(config.dev_path, word2id, max_sen_len=config.max_sen_len)
    print('训练模型===============================')
    tc = TextCNN(CONFIG, embeddings=word2vec)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        tc.fit(sess, x_tr, y_tr, x_val, y_val, config.save_dir, config.print_per_batch)


def test():
    config = CONFIG()
    print('加载word2id===========================')
    word2id = load_word2id(config.word2id_path)
    config.vocab_size = len(word2id)
    print('加载test语料库=========================')
    x, y = load_corpus(config.test_path, word2id, max_sen_len=config.max_sen_len)
    # x, y = x[:10], y[:10]
    model = TextCNN(config)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(config.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        yhat = model.predict(sess, x)

    cat, cat2id = cat_to_id()
    y_cls = np.argmax(y, 1)
    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_cls, yhat, target_names=cat))
    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_cls, yhat)
    print(cm)


if __name__ == '__main__':
    train()
    test()
