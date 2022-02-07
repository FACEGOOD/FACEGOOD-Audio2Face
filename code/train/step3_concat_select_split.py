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

import numpy as np
import os

project_dir = r'D:\voice2face\shirley_1119'
bs_name = np.load(os.path.join(project_dir,'shirley_1119_bs_name.npy'))
dataSet_dir = os.path.join(project_dir,'dataSet16')

# 参与训练的数据
# name_list = ['1_01','1_02','2_01','2_02','2_03','2_04'] #dataSet1
# name_list = ['1_01','1_02','2_01','2_02','2_03','2_04','3_15','3_16'] #dataSet2
# name_list = ['2_02','2_03','2_04','3_15','3_16'] #dataSet3：修改后的2_02和2_04
# name_list = ['2_02','2_03','2_04','3_15','3_16'] #dataSet4：修改后的2_02和2_04
# name_list = ['1_01','1_02','2_02','2_03','2_04'] #dataSet5：修改后的1_01、1_02、2_02和2_04
# name_list = ['1_01','1_02','2_02','2_03','2_04','3_15','3_16'] #dataSet6：修改后的1_01、1_02、2_02和2_04
# name_list = ['1_01','1_02','2_02','2_03','2_04'] #dataSet7：修改后的1_01、1_02、2_02和2_04；1_01、1_02张嘴等系数*1.3
# name_list = ['1_01','1_02','2_02','2_03','2_04','3_15','3_16'] #dataSet8：修改后的1_01、1_02、2_02和2_04；1_01、1_02张嘴等系数*1.3
# name_list = ['1_01','1_02','2_02','2_03','2_04'] #dataSet9：修改后的1_01、1_02、2_02和2_04；1_01、1_02张嘴等系数*1.3；删除单音节和词语一半的0帧
# name_list = ['1_01','1_02','2_02','2_03','2_04','3_15','3_16'] #dataSet10：修改后的1_01、1_02、2_02和2_04；1_01、1_02张嘴等系数*1.3；删除单音节和词语一半的0帧
# name_list = ['1_01','1_02','2_01','2_02','2_03','2_04'] #dataSet11：修改后的1_01、1_02、2_01、2_02和2_04
# name_list = ['1_01','1_02','2_01','2_02','2_03','2_04','3_15','3_16'] #dataSet12：修改后的1_01、1_02、2_01、2_02和2_04
# name_list = ['1_01','1_02','2_01','2_02','2_03','2_04'] #dataSet11：修改后的1_01、1_02、2_01、2_02和2_04；删除单音节和词语一半的0帧
# name_list = ['1_01','1_02','2_01','2_02','2_03','2_04','3_15','3_16'] #dataSet12：修改后的1_01、1_02、2_01、2_02和2_04；删除单音节和词语一半的0帧
name_list = ['1_01','1_02','2_01','2_02','2_03','2_04','2_05','3_15','3_16'] #dataSet16：all_付


data_path_list = [os.path.join(project_dir,'lpc','lpc_1114_' + i + '.npy') for i in name_list]
label_path_list = [os.path.join(project_dir,'bs_value','bs_value_1114_' + i + '.npy') for i in name_list]


data = np.zeros((1, 32, 64, 1))
label = np.zeros((1, 116))
for i in range(len(data_path_list)):
	data_temp = np.load(data_path_list[i]) 
	label_temp = np.load(label_path_list[i])

	if data_path_list[i][-8] == '1' or data_path_list[i][-8] == '2':
		label_temp_sum = label_temp.sum(axis=1)
		zero_index = np.where(label_temp_sum == 0)[0]
		half_zero_index = [zero_index[i] for i in range(0,len(zero_index),2)]
		select_index = [i for i in range(label_temp.shape[0]) if i not in half_zero_index]

		data_temp = data_temp[select_index]
		label_temp = label_temp[select_index]

	data = np.vstack((data,data_temp))
	label = np.vstack((label,label_temp))

data = data[1:]
label = label[1:]

print(data.shape)
print(label.shape)

# label1 = np.zeros((label.shape[0],label.shape[1]))
# bs_name_index = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 105, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 1, 115]
# for i in range(len(bs_name_index)):
# 	label1[:,i] = label[:,bs_name_index[i]]
# label = label1

############### select ############### 
# # bs的相关性计算
# import pandas as pd
# df = pd.DataFrame(label,columns=bs_name)
# corr = df.corr() #计算相关系数
# data = corr.values
# print(data.shape)
# bs_value_dict = {} # key：bs原始索引；value：与key的bs相关的min(bs_index)
# for i in range(len(bs_name)):
#     bs_name_corr = corr[(corr[bs_name[i]]==1)].index.tolist() #与bs_name[i]极相关的bs_name。 0.99-0.995 33个；0.999 37个；0.9999 38个；1 41个
#     if len(bs_name_corr)==0:
#         bs_value_dict[i] = i
#     else:
#         ndx = np.asarray([np.nonzero(bs_name == bs_name0)[0][0] for bs_name0 in bs_name_corr]) #返回索引
#         bs_value_dict[i] = min(ndx)
# print(bs_value_dict)
# bs_index = list(set(list(bs_value_dict.values())))
# print('uncorr_bs_index: ',bs_index)
# print('uncorr_bs_num: ',len(bs_index))


# # 筛选常值bs和变值bs
# bs_max_value = np.amax(label, axis=0) #每个BS的最大值
# bs_min_value = np.amin(label, axis=0) #每个BS的最小值
# # print('bs_max_value: ',bs_max_value)
# # print('bs_min_value: ',bs_min_value)
# const_bs_index = np.where(bs_max_value-bs_min_value==0) #常数BS的索引
# const_bs_index = list(const_bs_index[0])
# print('const_bs_index: ',const_bs_index)

# var_bs_index = [x for x in range(len(bs_name)) if x not in const_bs_index] #变值BS的索引
# print('var_bs_index: ',var_bs_index)
# print('var_bs_index_len: ',len(var_bs_index))

# const_bs_value = label[0][const_bs_index] #值不变的BS其值
# print('const_bs_value: ',const_bs_value)

# bs_name_1119_v1
# const_bs_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 38, 45, 46, 47, 48, 49, 54, 55, 60, 61, 64, 69, 70, 71, 72, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
# var_bs_index = [9, 12, 13, 14, 17, 32, 36, 37, 39, 40, 41, 42, 43, 44, 50, 51, 52, 53, 56, 57, 58, 59, 62, 63, 65, 66, 67, 68, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83]
# const_bs_value = [0.,-0.,0.,-0.,0.,-0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-0.,-0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.,0.]

# bs_name_1015_v1
# const_bs_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 39, 46, 47, 48, 49, 50, 55, 56, 61, 62, 65, 70, 71, 72, 73, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
# var_bs_index = [10, 13, 14, 15, 18, 33, 38, 40, 41, 42, 43, 44, 45, 51, 52, 53, 54, 57, 58, 59, 60, 63, 64, 66, 67, 68, 69, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84]
# const_bs_value = [0.,0.,-0.,0.,-0.,0.,-0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-0.,-0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.]

# bs_name_1119_v2
# const_bs_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 38, 45, 46, 47, 48, 49, 54, 55, 60, 61, 64, 69, 70, 71, 72, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
# var_bs_index = [9, 12, 13, 14, 17, 32, 36, 37, 39, 40, 41, 42, 43, 44, 50, 51, 52, 53, 56, 57, 58, 59, 62, 63, 65, 66, 67, 68, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83]
# const_bs_value = [0.,-0.,0.,-0.,0.,-0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-0.,-0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.,0.]

# bs_name_1015_v2
const_bs_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 39, 46, 47, 48, 49, 50, 55, 56, 61, 62, 65, 70, 71, 72, 73, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
var_bs_index = [10, 13, 14, 15, 18, 33, 38, 40, 41, 42, 43, 44, 45, 51, 52, 53, 54, 57, 58, 59, 60, 63, 64, 66, 67, 68, 69, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84]
const_bs_value = [0.,0.,-0.,0.,-0.,0.,-0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-0.,-0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.]

# 保存训练数据
train_data = data
val_data = data[-1000:]
train_label_var = label[:,var_bs_index]
val_label_var = label[-1000:,var_bs_index]

print(train_data.shape)
print(val_data.shape)
print(train_label_var.shape)
print(val_label_var.shape)

np.save(os.path.join(dataSet_dir,'train_data.npy'),train_data)
np.save(os.path.join(dataSet_dir,'val_data.npy'),val_data)
np.save(os.path.join(dataSet_dir,'train_label_var.npy'),train_label_var)
np.save(os.path.join(dataSet_dir,'val_label_var.npy'),val_label_var)


