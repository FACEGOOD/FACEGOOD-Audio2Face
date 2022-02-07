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

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
from step1_LPC import audioProcess
import time
import math

########## 加载路径 ##########
project_dir = r'D:\voice2face\shirley_1119'

# wav_path = os.path.join(project_dir,'wav','1015_3_16.wav')
# print(wav_path)

# data = audioProcess(wav_path)
data = np.load(os.path.join(project_dir,'lpc','lpc_1114_2_06.npy'))
print(data.shape)

pb_path = os.path.join(project_dir,'pb','0317_dataSet16_var_epoch8_2200.pb')
# predict_csv_path = os.path.join(project_dir,'csv','1114_2_06_1213_dataSet16_var_epoch2_25485.csv')

sess = tf.Session()
with gfile.GFile(pb_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='') # 导入计算图
# 需要有一个初始化的过程    
sess.run(tf.global_variables_initializer())

# 输入
input_x = sess.graph.get_tensor_by_name('Placeholder_1:0')
input_keep_prob_tensor = sess.graph.get_tensor_by_name('Placeholder:0')
out = sess.graph.get_tensor_by_name('dense_1/BiasAdd:0')

# lpc_1021_3_14 = lpc_1021_3_14.reshape((-1,32*64,1))
# data = lpc_1021_3_14[222]
# data = data.reshape((-1))

# 每次处理300帧，避免OOM超内存
weight = []
for i in range(100): #data.shape[0]
    start_time = time.time()
    data_temp = data[i]
    data_temp = data_temp.reshape((1,32,64,1))
    weight_sub = sess.run(out, feed_dict={input_x:data_temp,input_keep_prob_tensor:1.0})
    end_time = time.time()
    print("time:",end_time - start_time)
    print(len(weight_sub[0]))
    # weight.extend(weight_sub)
# weight = np.array(weight)
# print(weight.shape)

