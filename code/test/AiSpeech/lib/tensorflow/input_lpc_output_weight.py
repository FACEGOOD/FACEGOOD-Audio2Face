#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
    tensorflow.input_lpc_output_predict_by_pb.py
    ~~~~~~~~~~~~~
    :copyright: © 2019 by the facegood team.
"""
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.platform import gfile

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

class WeightsAnimation:
    def __init__(self, pb_path):
        self.sess = tf.Session()
        with gfile.FastGFile(pb_path, 'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(self.graph_def, name='')
        self.sess.run(tf.global_variables_initializer())

        self.initdata()

    def initdata(self):
        self.input_x = self.sess.graph.get_tensor_by_name('Placeholder_1:0')
        self.input_keep_prob_tensor = self.sess.graph.get_tensor_by_name(
            'Placeholder:0')
        self.out = self.sess.graph.get_tensor_by_name('dense_1/BiasAdd:0')


    def get_weight(self, data):
        weight = self.sess.run(self.out,
                                        feed_dict={
                                            self.input_x: data,
                                            self.input_keep_prob_tensor: 1.0
                                        })

        return weight


if __name__ == "__main__":
    import time
    sall = time.time()
    import os
    import numpy as np
    pb_path = r"W:\YY\1029_voice2face_shirley\tf.pb"
    pb_weights_animation = WeightsAnimation(pb_path)
    get_weight = pb_weights_animation.get_weight
    npy_path = r"G:\input_lpc.npy"
    data = np.load(npy_path)
    data1 = data[10:15]
    data1.shape
    weight_data = get_weight(data1)
    weight_data.shape
    weight = weight_data[45][0:20]
    weight_data[0].dtype
    for i in range(0, 30):
        print("time is:", i)
        s = time.time()
        data1 = data[(i * 1):(i + 1)]
        # cProfile.run("predict(data1)")
        data_predict = get_weight(data1)
        e = time.time()
        print("**:", e - s)
        print(len(data_predict))
    eall = time.time()
    print("**:", eall - sall)
    print("done")
    os.system("pause")
