# Copyright 2023 The FACEGOOD Authors. All Rights Reserved.
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
from tensorflow.keras import Model, models
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout


def conv2d_layer(filters=256, kernel_size=None, strides=None):
    """Conv2D Layer
    Args:
        filters: int, the output channels of the conv2d layer
        kernel_size: list, the kernel size of the conv2d layer
        strides: list, the strides of the conv2d layer
    """
    if kernel_size is None:
        kernel_size = [1, 1]
    if strides is None:
        strides = [1, 1]
    conv2d = Conv2D(filters=filters, kernel_size=kernel_size, padding="same", activation='relu', strides=strides)
    return conv2d

class FormantLayer(Model):
    """Formant Layer
    Args:
        kernels_size: list, the kernel size of each conv2d layer
        outputs: list, the output channels of each conv2d layer    
    """
    def __init__(self, kernels_size=None, outputs=None):
        super(FormantLayer, self).__init__()

        if kernels_size is None:
            kernels_size = [[3, 1], [3, 1], [3, 1], [3, 1], [2, 1]]
        if outputs is None:
            outputs = [72, 108, 162, 243, 256]

        self.kernels_size = kernels_size
        self.outputs = outputs

        self.formant_layers = models.Sequential()
        for i in range(len(self.kernels_size)):
            self.formant_layers.add(conv2d_layer(filters=self.outputs[i],
                                                 kernel_size=self.kernels_size[i],
                                                 strides=[2, 1]))

    def call(self, x):
        x = self.formant_layers(x)  
        return x

class ArticulationLayer(Model):
    """Articulation Layer
    Args:
        kernels_size: list, the kernel size of each conv2d layer
        E: int, the channels of the emotion layer
        conv2d_strides: list, the strides of each conv2d layer
        emotion_strides: list, the strides of each emotion layer
    """
    def __init__(self, kernels_size=None, E=16, conv2d_strides=None, emotion_strides=None):
        super(ArticulationLayer, self).__init__()

        self.E = E 
        if kernels_size is None:
            kernels_size = [[1, 3], [1, 3], [1, 3], [1, 3], [1, 4]]
        if emotion_strides is None:
            emotion_strides = [[1, 2], [1, 4], [1, 8], [1, 16], [1, 64]]
        if conv2d_strides is None:
            conv2d_strides = [[1, 2], [1, 2], [1, 2], [1, 2], [1, 4]]

        self.kernels_size = kernels_size
        self.emotion_strides = emotion_strides
        self.conv2d_strides = conv2d_strides

        self.emotion = conv2d_layer(self.E, [3, 1], [32, 1])
        self.articulation_layer = []
        for i in range(len(self.kernels_size)):
            self.articulation_layer.append([conv2d_layer(256, self.kernels_size[i], self.conv2d_strides[i]),
                                            conv2d_layer(self.E, [1, 3], self.emotion_strides[i])])
    def call(self, x):
        emotion_input = self.emotion(x)
        for i in range(len(self.kernels_size)):
            conv_x = self.articulation_layer[i][0](x)
            emotion_x = self.articulation_layer[i][1](emotion_input)
            mixed_x = tf.concat([conv_x, emotion_x], 3) # Concatenate the channels
            x = mixed_x
        return (x, emotion_input)

class OutputLayer(Model):
    """Output Layer
    Args:
        output_size: int, the output size of the output layer
        keep_pro: float, the keep probability of the dropout layer
    """
    def __init__(self, output_size, keep_pro):
        super(OutputLayer, self).__init__()
        self.output_layer = models.Sequential([
            # Flatten(),
            Dense(units=150, activation=None),
            Dropout(keep_pro),
            Dense(units=output_size, activation=None)
        ])
    def call(self, x):
        return self.output_layer(x)

class Audio2Face(Model):
    """Audio2Face Model
    Args:
        output_size: int, the output size of the output layer
        keep_pro: float, the keep probability of the dropout layer
    """
    def __init__(self, output_size, keep_pro):
        super(Audio2Face, self).__init__()
        self.output_size = output_size
        self.FormantLayer = FormantLayer()
        self.ArticulationLayer = ArticulationLayer()
        self.OutputLayer = OutputLayer(self.output_size, keep_pro)
        
    def call(self, x):
        x = self.FormantLayer(x)
        x, emotion_input = self.ArticulationLayer(x)
        x = self.OutputLayer(x)
        return (x, emotion_input)

def losses(y, output):
    """Loss Function
    Args:
        y: tensor, the ground truth
        output: tensor,[pred , emotion_input] the output of the model
    """
    y_, emotion_input = output

    y = tf.cast(y, dtype=tf.float32)    # Cast the type of y to float32
    y_ = tf.cast(y_, dtype=tf.float32)  # Cast the type of y_ to float32
    emotion_input = tf.cast(emotion_input, dtype=tf.float32) # Cast the type of emotion_input to float32

    loss_P = tf.reduce_mean(tf.square(y - y_)) # Calculate the loss_P

    # Calculate the loss_M
    split_y = tf.split(y, 2, 0)  # Parameter: tensor, split number, dimension
    split_y_ = tf.split(y_, 2, 0)

    y0 = split_y[0] # y0 is the first half of y
    y1 = split_y[1] # y1 is the second half of y
    y_0 = split_y_[0]   # y_0 is the first half of y_
    y_1 = split_y_[1]   # y_1 is the second half of y_
    loss_M = 2 * tf.reduce_mean(tf.square(y0 - y1 - y_0 + y_1)) # Calculate the loss_M

    # Calculate the loss_R
    split_emotion_input = tf.split(emotion_input, 2, 0)
    emotion_input0 = split_emotion_input[0] # emotion_input0 is the first half of emotion_input
    emotion_input1 = split_emotion_input[1] # emotion_input1 is the second half of emotion_input

    # Formula(3), Rx3 is R'(x)
    Rx0 = tf.square(emotion_input0 - emotion_input1)  # Calculate the m[·]
    Rx1 = tf.reduce_sum(Rx0, 1)  # 4-dim, sum of the height
    Rx2 = tf.reduce_sum(Rx1, 1)  # 3-dim, sum of the width
    Rx3 = 2 * tf.reduce_mean(Rx2, 1)  # 2-dim, mean of the emotion

    # Formula(4), Rx is R(x), length is batch_size/2
    e_mean0 = tf.reduce_sum(tf.square(emotion_input0), 2)  # 4-dim, sum of the width
    e_mean1 = tf.reduce_mean(e_mean0)  # 2-dim, mean of the emotion
    Rx = Rx3 / e_mean1  # R(x)

    # Formula(5)
    # beta = 0.99
    # R_vt = beta * R_vt_input + (1-beta) * tf.reduce_mean(tf.square(Rx)) # every epoch update
    # R_vt_ = R_vt/(1-tf.pow(beta, step))

    # Formula(6) Calculate the loss_R
    # loss_R = tf.reduce_mean(Rx)/(tf.sqrt(R_vt_)+epsilon)
    loss_R = tf.reduce_mean(Rx)
    # loss_R = tf.reduce_mean(tf.square(emotion_input1 - emotion_input0), name='loss_R')

    # Calculate the total loss
    loss = loss_P + loss_M + loss_R
    return loss
