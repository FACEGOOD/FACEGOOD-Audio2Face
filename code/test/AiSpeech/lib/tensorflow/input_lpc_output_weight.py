#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
    tensorflow.input_lpc_output_predict_by_pb.py
    ~~~~~~~~~~~~~
    :copyright: © 2019 by the facegood team.
"""
import os
# import tensorflow.compat.v1 as tf
import tensorflow as tf
import numpy as np
# tf.disable_v2_behavior()
# from tensorflow.python.platform import gfile
'''
该类主要是调用tensorflow 加载训练好的算法模型，输入音频数据输出动画表情权重。
this class load trained model and predict weights.
'''
class WeightsAnimation:
    '''
        在_init__中对模型文件进行加载，并在initdata 中构建输入输出函数句柄。
        init tensorflow enviroment and load graph.
    '''
    def __init__(self, model_path, pb_model_path=None):
        self.model_path = model_path
        if not os.path.exists(model_path):
            self.convert_to_tflite(pb_model_path)
        # Load the model
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=8)
        # Set model input
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def convert_to_tflite(self, pb_model_path):
        # Convert the model from saved model(.pb) to tflite
        converter = tf.lite.TFLiteConverter.from_saved_model(pb_model_path)
        tflite_model = converter.convert()
        with open(self.model_path, "wb") as f:
            f.write(tflite_model)
        print(f'Save TFLite model to {self.model_path} successfully!')


    def run(self, inputData):
        # Preprocess the image before sending to the network.
        inputData = np.expand_dims(inputData, axis=0)
        # The actual detection.
        self.interpreter.set_tensor(self.input_details[0]["index"], inputData)
        self.interpreter.invoke()
        # Save the results.
        mesh = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        return mesh
    
    
    def get_weight(self, data, label_len=37):
        frame_num = data.shape[0]
        weight = np.zeros((frame_num, label_len), dtype=np.float32)
        for i in range(frame_num): 
            # print(f"frame is {i}")
            data_temp = data[i].astype(np.float32)
            # import pdb; pdb.set_trace()
            output = self.run(data_temp).reshape((1,-1))
            weight[i] = output
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
