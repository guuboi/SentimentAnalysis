import os
import time
import numpy as np
import tensorflow as tf
from utils import batch_index, time_diff


class TextCNN(object):
    def __init__(self, config, embeddings=None):
        self.update_w2v = config.update_w2v
        self.vocab_size = config.vocab_size
        self.n_class = config.n_class
        self.max_sen_len= config.max_sen_len
        self.embedding_dim = config.embedding_dim
        self.batch_size = config.batch_size
        self.num_filters = config.num_filters
        self.kernel_size = config.kernel_size
        self.n_hidden = config.n_hidden
        self.n_epoch = config.n_epoch
        self.opt = config.opt
        self.learning_rate = config.learning_rate
        self.drop_keep_prob = config.drop_keep_prob

        self.x = tf.placeholder(tf.int32, [None, self.max_sen_len], name='x')
        self.y = tf.placeholder(tf.int32, [None, self.n_class], name='y')
        # self.word_embeddings = tf.constant(embeddings, tf.float32)
        # self.word_embeddings = tf.Variable(embeddings, dtype=tf.float32, trainable=self.update_w2v)
        if embeddings:
            self.word_embeddings = tf.Variable(embeddings, dtype=tf.float32, trainable=self.update_w2v)
        else:
            self.word_embeddings = tf.Variable(
                tf.zeros([self.vocab_size, self.embedding_dim]),
                dtype=tf.float32,
                trainable=self.update_w2v)

        self.build()

    def cnn(self):
        """
        :param mode:默认为None，主要调节dropout操作对训练和预测带来的差异。
        :return: 未经softmax变换的fully-connected输出结果
        """
        inputs = self.add_embeddings()
        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(inputs, self.num_filters, self.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
            # dropout 卷积层后加dropout效果太差
            # gmp = tf.contrib.layers.dropout(gmp, self.drop_keep_prob)

        with tf.name_scope("score"):
            # fully-connected
            fc = tf.layers.dense(gmp, self.n_hidden, name='fc1')
            # dropout
            fc = tf.contrib.layers.dropout(fc, self.drop_keep_prob)
            # nonlinear
            fc = tf.nn.relu(fc)
            # fully-connected
            pred = tf.layers.dense(fc, self.n_class, name='fc2')
        return pred

    def add_embeddings(self):
        inputs = tf.nn.embedding_lookup(self.word_embeddings, self.x)
        return inputs

    def add_loss(self, pred):
        cost = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y)
        cost = tf.reduce_mean(cost)
        return cost

    def add_optimizer(self, loss):
        if self.opt == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95, epsilon=1e-6)
        else:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        opt = optimizer.minimize(loss)
        return opt

    def add_accuracy(self, pred):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy

    def get_batches(self, x, y=None, batch_size=100, is_shuffle=True):
        for index in batch_index(len(x), batch_size, is_shuffle=is_shuffle):
            n = len(index)
            feed_dict = {
                self.x: x[index]
            }
            if y is not None:
                feed_dict[self.y] = y[index]
            yield feed_dict, n

    def build(self):
        self.pred = self.cnn()
        self.loss = self.add_loss(self.pred)
        self.accuracy = self.add_accuracy(self.pred)
        self.optimizer = self.add_optimizer(self.loss)

    def train_on_batch(self, sess, feed):
        _, _loss, _acc = sess.run([self.optimizer, self.loss, self.accuracy], feed_dict=feed)
        return _loss, _acc

    def test_on_batch(self, sess, feed):
        _loss, _acc = sess.run([self.loss, self.accuracy], feed_dict=feed)
        return _loss, _acc

    def predict_on_batch(self, sess, feed, prob=True):
        result = tf.argmax(self.pred, 1)
        if prob:
            result = tf.nn.softmax(logits=self.pred, dim=1)

        res = sess.run(result, feed_dict=feed)
        return res

    def predict(self, sess, x, prob=False):
        yhat = []
        for _feed, _ in self.get_batches(x, batch_size=self.batch_size, is_shuffle=False):
            _yhat = self.predict_on_batch(sess, _feed, prob)
            yhat += _yhat.tolist()
            # yhat.append(_yhat)
        return np.array(yhat)

    def evaluate(self, sess, x, y):
        """评估在某一数据集上的准确率和损失"""
        num = len(x)
        total_loss, total_acc = 0., 0.
        for _feed, _n in self.get_batches(x, y, batch_size=self.batch_size):
            loss, acc = self.test_on_batch(sess, _feed)
            total_loss += loss * _n
            total_acc += acc * _n
        return total_loss / num, total_acc / num

    def fit(self, sess, x_train, y_train, x_dev, y_dev, save_dir=None, print_per_batch=100):
        saver = tf.train.Saver()
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        sess.run(tf.global_variables_initializer())

        print('Training and evaluating...')
        start_time = time.time()
        total_batch = 0 # 总批次
        best_acc_dev = 0.0  # 最佳验证集准确率
        last_improved = 0   # 记录上次提升批次
        require_improvement = 500  # 如果超过500轮模型效果未提升，提前结束训练
        flags = False
        for epoch in range(self.n_epoch):
            print('Epoch:', epoch + 1)
            for train_feed, train_n in self.get_batches(x_train, y_train, batch_size=self.batch_size):
                loss_train, acc_train = self.train_on_batch(sess, train_feed)
                loss_dev, acc_dev = self.evaluate(sess, x_dev, y_dev)

                if total_batch % print_per_batch == 0:
                    if acc_dev > best_acc_dev:
                        # 保存在验证集上性能最好的模型
                        best_acc_dev = acc_dev
                        last_improved = total_batch
                        if save_dir:
                            saver.save(sess=sess, save_path=os.path.join(save_dir, 'sa-model'))
                        improved_str = '*'
                    else:
                        improved_str = ''

                    time_dif = time_diff(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' + \
                          ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(total_batch, loss_train, acc_train, loss_dev, acc_dev, time_dif, improved_str))
                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    print('No optimization for a long time, auto-stopping...')
                    flags = True
                    break
            if flags:
                break

