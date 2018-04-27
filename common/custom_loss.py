# coding:utf-8
import tensorflow as tf
import numpy as np

y_ = tf.constant([[0.0, 1.0], [1.0, 0]])  # 正确标签
y1 = tf.constant([[0.2, 0.8], [1.25, 0.75]])  # 预测结果1

loss_more = 1
loss_less = 10
# 以下为未经过Softmax处理的类别得分
sess = tf.InteractiveSession()
tf.global_variables_initializer()

losses_mat_1 = y1 - y1 * y_ + tf.log(1 + tf.exp(-y1))  # 交叉熵损失
losses_mat_2 = -y_ * tf.log(tf.sigmoid(y1)) - (1 - y_) * tf.log(tf.sigmoid(-y1))  # 交叉熵损失
losses_mat_c = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y1)

losses_mat_3 = -y_ * (tf.sigmoid(-4 * y1)) * tf.log(tf.sigmoid(y1)) - (1 - y_) * tf.sigmoid(4 * y1) * tf.log(tf.sigmoid(-y1))
# losses_c_mat = -(tf.nn.sigmoid(((-1)**y_)*(4 * y1)) * tf.log(tf.clip_by_value(y_sigmoid2, 1e-10, 1.0))) * y_
loss = tf.reduce_sum(tf.where(tf.greater(y1, y_), (y1 - y_) * loss_more, (y_ - y1) * loss_less))
# tf.where(input)返回true的坐标位置，tf.where(input,x,y),True坐标位置用x的位置，FALSE坐标位置用y的位置

losses_c = tf.reduce_sum(losses_mat_c)
losses_1 = tf.reduce_sum(losses_mat_2)

print(sess.run(losses_mat_c))
print(sess.run(losses_mat_1))

print("-------------------------")

print(sess.run(losses_c))
print(sess.run(losses_1))

print(sess.run(tf.greater(y1, y_)))
print(sess.run(tf.where(tf.greater(y1, y_), (y1 - y_) * loss_more, (y_ - y1) * loss_less)))
