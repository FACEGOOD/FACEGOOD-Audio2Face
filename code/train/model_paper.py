# Copyright 2021 The FACEGOOD Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf


# import numpy as np


# 定义网络模型
def net(input_data, outputSize, keep_pro):
    # Formant network
    conv1 = tf.layers.conv2d(inputs=input_data, filters=72, kernel_size=[3, 1],
                             padding="same", activation=tf.nn.relu, strides=[2, 1])
    conv2 = tf.layers.conv2d(inputs=conv1, filters=108, kernel_size=[3, 1],
                             padding="same", activation=tf.nn.relu, strides=[2, 1])
    conv3 = tf.layers.conv2d(inputs=conv2, filters=162, kernel_size=[3, 1],
                             padding="same", activation=tf.nn.relu, strides=[2, 1])
    conv4 = tf.layers.conv2d(inputs=conv3, filters=243, kernel_size=[3, 1],
                             padding="same", activation=tf.nn.relu, strides=[2, 1])
    conv5 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=[2, 1],
                             padding="same", activation=tf.nn.relu, strides=[2, 1])

    E = 16  # emotional state
    emotion_input = tf.layers.conv2d(inputs=input_data, filters=E, kernel_size=[3, 1],
                                     padding="same", activation=tf.nn.relu, strides=[32, 1])

    # Articulation network
    conv6 = tf.layers.conv2d(inputs=conv5, filters=256, kernel_size=[1, 3],
                             padding="same", activation=tf.nn.relu, strides=[1, 2])
    emotion1 = tf.layers.conv2d(inputs=emotion_input, filters=E, kernel_size=[1, 3],
                                padding="same", activation=tf.nn.relu, strides=[1, 2])
    mixed1 = tf.concat([conv6, emotion1], 3)

    conv7 = tf.layers.conv2d(inputs=mixed1, filters=256, kernel_size=[1, 3],
                             padding="same", activation=tf.nn.relu, strides=[1, 2])
    emotion2 = tf.layers.conv2d(inputs=emotion_input, filters=E, kernel_size=[1, 3],
                                padding="same", activation=tf.nn.relu, strides=[1, 4])
    mixed2 = tf.concat([conv7, emotion2], 3)

    conv8 = tf.layers.conv2d(inputs=mixed2, filters=256, kernel_size=[1, 3],
                             padding="same", activation=tf.nn.relu, strides=[1, 2])
    emotion3 = tf.layers.conv2d(inputs=emotion_input, filters=E, kernel_size=[1, 3],
                                padding="same", activation=tf.nn.relu, strides=[1, 8])
    mixed3 = tf.concat([conv8, emotion3], 3)

    conv9 = tf.layers.conv2d(inputs=mixed3, filters=256, kernel_size=[1, 3],
                             padding="same", activation=tf.nn.relu, strides=[1, 2])
    emotion4 = tf.layers.conv2d(inputs=emotion_input, filters=E, kernel_size=[1, 3],
                                padding="same", activation=tf.nn.relu, strides=[1, 16])
    mixed4 = tf.concat([conv9, emotion4], 3)

    conv10 = tf.layers.conv2d(inputs=mixed4, filters=256, kernel_size=[1, 4],
                              padding="same", activation=tf.nn.relu, strides=[1, 4])
    emotion5 = tf.layers.conv2d(inputs=emotion_input, filters=E, kernel_size=[1, 3],
                                padding="same", activation=tf.nn.relu, strides=[1, 64])
    mixed5 = tf.concat([conv10, emotion5], 3)

    # Output network
    flat = tf.layers.flatten(mixed5)

    fc1 = tf.layers.dense(inputs=flat, units=150, activation=None)  # activation=None表示使用线性激活器
    dropout = tf.nn.dropout(fc1, keep_pro)
    output = tf.layers.dense(inputs=dropout, units=outputSize, activation=None)
    return output, emotion_input

#define losses functions
def losses(y, y_, emotion_input):
    # 计算 loss_P
    loss_P = tf.reduce_mean(tf.square(y - y_))

    # 计算 loss_M
    split_y = tf.split(y, 2, 0)  # 参数分别为：tensor，拆分数，维度
    split_y_ = tf.split(y_, 2, 0)  # 参数分别为：tensor，拆分数，维度
    # print(10)
    y0 = split_y[0]
    y1 = split_y[1]
    y_0 = split_y_[0]
    y_1 = split_y_[1]
    loss_M = 2 * tf.reduce_mean(tf.square(y0 - y1 - y_0 + y_1))

    # 计算loss_R
    # 拆分tensor。https://blog.csdn.net/liuweiyuxiang/article/details/81192547
    split_emotion_input = tf.split(emotion_input, 2, 0)  # 参数分别为：tensor，拆分数，维度
    # print(10)
    emotion_input0 = split_emotion_input[0]
    emotion_input1 = split_emotion_input[1]

    # 公式(3),Rx3即R'(x)
    Rx0 = tf.square(emotion_input0 - emotion_input1)  # 计算m[·]
    Rx1 = tf.reduce_sum(Rx0, 1)  # 4维。按shape(1)计算和，即：高
    Rx2 = tf.reduce_sum(Rx1, 1)  # 3维。按shape(1)计算和，即：4维的shape(2)，宽
    Rx3 = 2 * tf.reduce_mean(Rx2, 1)  # 2维。按shape(1)计算均值，即：4维的shape(3)，E

    # 公式(4),Rx是长度为batch_size/2的tensor
    e_mean0 = tf.reduce_sum(tf.square(emotion_input0), 2)  # 4维。按shape(2)计算和，即：宽。因为高为1，所以只算一次sum
    e_mean1 = tf.reduce_mean(e_mean0)  # 2维。按shape(1)计算均值，即：4维的shape(3)，E
    Rx = Rx1 / e_mean1

    # 公式(5)
    # R_vt = beta * R_vt_input + (1-beta) * tf.reduce_mean(tf.square(Rx)) #每个batch运行一次
    # R_vt_ = R_vt/(1-tf.pow(beta, step))

    # 公式(6)
    # loss_R = tf.reduce_mean(Rx)/(tf.sqrt(R_vt_)+epsilon)
    loss_R = tf.reduce_mean(Rx)

    # loss_R = tf.reduce_mean(tf.square(emotion_input1 - emotion_input0), name='loss_R')

    # 计算最终的 Loss
    loss = loss_P + loss_M + loss_R

    return loss
