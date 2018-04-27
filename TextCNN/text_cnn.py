# coding:utf-8
import numpy as np
import tensorflow as tf


class TextCNN(object):
    """
    ex_wv:引入外部词向量，默认为None，此时词向量随机生成
    wv_update:词向量训练过程中是否更新
    """

    def __init__(self,
                 sequence_length,
                 num_classes,
                 vocab_size,
                 embedding_size,
                 filter_sizes,
                 num_filters,
                 l2_reg_lambda=0.0,
                 embedding_mat=None,
                 wv_non_static=False):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            if embedding_mat is not None:
                wv_mat = tf.Variable(embedding_mat, name='wv_mat', trainable=wv_non_static)
                self.embedded_chars = tf.nn.embedding_lookup(wv_mat, self.input_x)
            else:
                wv_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                     name='wv_mat', trainable=wv_non_static)
                self.embedded_chars = tf.nn.embedding_lookup(wv_mat, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv')

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope('output'):
            W = tf.get_variable(
                'W',
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        # Calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            # losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)  # only named arguments accepted
            losses = -self.input_y * (tf.sigmoid(-4 * self.scores)) * tf.log(tf.sigmoid(self.scores)) - \
                            (1 - self.input_y) * tf.sigmoid(4 * self.scores) * tf.log(tf.sigmoid(-self.scores))
            # self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        # Accuracy
        with tf.name_scope('accuracy'):
            # correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            # self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
            # how to use tf.metrics.accuracy() function, how about tf.contrib.metrics.accuracy()
            # 在使用tf.metrics.accuracy()时一定要加tf.local_variables_initializer()
            self.accuracy, _ = tf.metrics.accuracy(labels=tf.argmax(self.input_y, 1), predictions=self.predictions)
            print(self.accuracy)

        with tf.name_scope('precision'):
            # correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            # self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
            self.precision, _ = tf.metrics.precision(labels=tf.argmax(self.input_y, 1), predictions=self.predictions)
            print(self.precision)

        with tf.name_scope('recall'):
            self.recall, _ = tf.metrics.recall(labels=tf.argmax(self.input_y, 1), predictions=self.predictions)
            print(self.recall)

        with tf.name_scope('confusion_matrix'):
            # self.confusion_matrix = tf.contrib.metrics.confusion_matrix(labels=tf.argmax(self.input_y, 1), predictions=self.predictions)
            self.confusion_matrix = tf.confusion_matrix(labels=tf.argmax(self.input_y, 1), predictions=self.predictions)
            print(self.confusion_matrix)

        with tf.name_scope('num_correct'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')
