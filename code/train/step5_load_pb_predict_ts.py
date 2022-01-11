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
# import time
import math

########## 加载路径 ##########
project_dir = r'D:\audio2bs\shirley_1119'

# wav_path = os.path.join(project_dir,'wav','1015_3_16.wav')
# print(wav_path)

# data = audioProcess(wav_path)
data = np.load(os.path.join(project_dir,'lpc','lpc_1114_2_06.npy'))
print(data.shape)

pb_path = os.path.join(project_dir,'pb','1213_dataSet16_var_epoch2_25485.pb')
predict_csv_path = os.path.join(project_dir,'csv','1114_2_06_1213_dataSet16_var_epoch2_25485.csv')

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

# 每次处理300帧，避免OOM超内存
weight = []
for i in range(math.ceil(data.shape[0]/300)): #data.shape[0]
    weight_sub = sess.run(out, feed_dict={input_x:data[300*i:min(data.shape[0],300*(i+1))],input_keep_prob_tensor:1.0})
    weight.extend(weight_sub)
weight = np.array(weight)
print(weight.shape)

########## 给常值bs赋0值 ##########

# 参与训练的数据
name_list = ['1_01','1_02','2_01']
label_path_list = [os.path.join(project_dir,'bs_value','bs_value_1015_' + i + '.npy') for i in name_list]

label = np.zeros((1, 116))
for i in range(len(label_path_list)):
	label_temp = np.load(label_path_list[i]) 
	label = np.vstack((label,label_temp))

label = label[1:]

# 筛选常值bs和变值bs
bs_max_value = np.amax(label, axis=0) #每个BS的最大值
bs_min_value = np.amin(label, axis=0) #每个BS的最小值
# print('bs_max_value: ',bs_max_value)
# print('bs_min_value: ',bs_min_value)
const_bs_index = np.where(bs_max_value-bs_min_value==0) #常数BS的索引
const_bs_index = list(const_bs_index[0])
print('const_bs_index: ',const_bs_index)

var_bs_index = [x for x in range(len(bs_name)) if x not in const_bs_index] #变值BS的索引
print('var_bs_index: ',var_bs_index)

const_bs_value = label[0][const_bs_index] #值不变的BS其值
print('const_bs_value: ',const_bs_value)


# bs_name_1119
bs_name = np.load(os.path.join(project_dir,'shirley_1119_bs_name.npy'))
const_bs_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 38, 45, 46, 47, 48, 49, 54, 55, 60, 61, 64, 69, 70, 71, 72, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
var_bs_index = [9, 12, 13, 14, 17, 32, 36, 37, 39, 40, 41, 42, 43, 44, 50, 51, 52, 53, 56, 57, 58, 59, 62, 63, 65, 66, 67, 68, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83]
const_bs_value = [0.,-0.,0.,-0.,0.,-0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-0.,-0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.,0.]

bs_name_1015_v2
bs_name = np.load(os.path.join(project_dir,'shirley_1015_bs_name.npy'))
const_bs_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 39, 46, 47, 48, 49, 50, 55, 56, 61, 62, 65, 70, 71, 72, 73, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
var_bs_index = [10, 13, 14, 15, 18, 33, 38, 40, 41, 42, 43, 44, 45, 51, 52, 53, 54, 57, 58, 59, 60, 63, 64, 66, 67, 68, 69, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84]
const_bs_value = [0.,0.,-0.,0.,-0.,0.,-0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-0.,-0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.]


# 赋值
weights = np.zeros((weight.shape[0],len(bs_name)))
print(weights.shape)

weights[:,var_bs_index] = weight
weights[:,const_bs_index] = const_bs_value

# weights1 = np.zeros((weight.shape[0],len(bs_name)))
# bs_name_index = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 105, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 1, 115]
# for i in range(len(bs_name_index)):
# 	weights1[:,i] = weights[:,bs_name_index[i]]

# 导出权重的csv文件
import pandas as pd
df = pd.DataFrame(weights,columns=bs_name)
df.to_csv(predict_csv_path,index=0)
