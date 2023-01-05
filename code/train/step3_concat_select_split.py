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

import numpy as np
import os

project_dir = r'./'
# bs_name = np.load(os.path.join(project_dir, 'shirley_1015_bs_name.npy'))
dataSet_dir = os.path.join(project_dir, 'dataSet4_6')
if not os.path.exists(dataSet_dir):
    os.mkdir(dataSet_dir)

# train data 
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
# name_list = ['1_01', '1_02', '2_01', '2_02', '2_03', '2_04', '2_05', '3_15', '3_16']  # dataSet16：all_付

train_data = []
val_data = []
train_label_var = []
val_label_var = []

# # dataSet 4-16(13,14,15)
# data_list = [['2_02','2_03','2_04','3_15','3_16'],
#              ['1_01','1_02','2_02','2_03','2_04'],
#             ['1_01','1_02','2_02','2_03','2_04','3_15','3_16'],
#             ['1_01','1_02','2_02','2_03','2_04'],
#             ['1_01','1_02','2_02','2_03','2_04','3_15','3_16'],
#             ['1_01','1_02','2_02','2_03','2_04'],
#             ['1_01','1_02','2_02','2_03','2_04','3_15','3_16'],
#             ['1_01','1_02','2_01','2_02','2_03','2_04'],
#             ['1_01','1_02','2_01','2_02','2_03','2_04','3_15','3_16'],
#            ['1_01', '1_02', '2_01', '2_02', '2_03', '2_04', '2_05', '3_15', '3_16']]

# dataSet 4-6
data_list = [['2_02','2_03','2_04','3_15','3_16'],
             ['1_01','1_02','2_02','2_03','2_04'],
            ['1_01','1_02','2_02','2_03','2_04','3_15','3_16'],
            ]


for i in range(len(data_list)):
    name_list = data_list[i]
        
    data_path_list = [os.path.join(project_dir, 'lpc', '1114_' + i + '.npy') for i in name_list]
    label_path_list = [os.path.join(project_dir, 'bs_value', 'bs_value_1114_' + i + '.npy') for i in name_list]

    data = np.zeros((1, 32, 64, 1))
    label = np.zeros((1, 116))

    # generate data and label
    for i in range(len(data_path_list)):
        data_temp = np.load(data_path_list[i])
        label_temp = np.load(label_path_list[i])

        # '1_01', '1_02', '2_01', '2_02', '2_03', '2_04', '2_05'
        if data_path_list[i][-8] == '1' or data_path_list[i][-8] == '2':
            label_temp_sum = label_temp.sum(axis=1)
            zero_index = np.where(label_temp_sum == 0)[0]
            half_zero_index = [zero_index[i] for i in range(0, len(zero_index), 2)]
            select_index = [i for i in range(label_temp.shape[0]) if i not in half_zero_index]

            data_temp = data_temp[select_index]
            label_temp = label_temp[select_index]

        data = np.vstack((data, data_temp))
        label = np.vstack((label, label_temp))

    data = data[1:]
    label = label[1:]


    # # bs_name_1119_v2
    # const_bs_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 38, 45, 46, 47, 48, 49, 54, 55, 60, 61, 64, 69, 70, 71, 72, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
    # var_bs_index = [9, 12, 13, 14, 17, 32, 36, 37, 39, 40, 41, 42, 43, 44, 50, 51, 52, 53, 56, 57, 58, 59, 62, 63, 65, 66, 67, 68, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83]
    # const_bs_value = [0.,-0.,0.,-0.,0.,-0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-0.,-0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.,0.]

    # bs_name_1015_v2
    const_bs_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                      34, 35, 36, 37, 39, 46, 47, 48, 49, 50, 55, 56, 61, 62, 65, 70, 71, 72, 73, 83, 85, 86, 87, 88, 89,
                      90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                      112, 113, 114, 115]
    var_bs_index = [10, 13, 14, 15, 18, 33, 38, 40, 41, 42, 43, 44, 45, 51, 52, 53, 54, 57, 58, 59, 60, 63, 64, 66, 67, 68,
                    69, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84]
    const_bs_value = [0., 0., -0., 0., -0., 0., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., -0., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0., 0., -0., 0., -0., 0., -0., 0., 0., -0., 0., -0., 0., -0.,
                      0., -0., 0., -0., 0.]
    
    train_data.extend(data)
    val_data.extend(data[-1000:])
    train_label_var.extend(label[:, var_bs_index])
    val_label_var.extend(label[-1000:, var_bs_index])

print(np.array(train_data).shape)
print(np.array(val_data).shape)
print(np.array(train_label_var).shape)
print(np.array(val_label_var).shape)


# save data and label to npy
np.save(os.path.join(dataSet_dir, 'train_data.npy'), np.array(train_data))
np.save(os.path.join(dataSet_dir, 'val_data.npy'), np.array(val_data))
np.save(os.path.join(dataSet_dir, 'train_label_var.npy'), np.array(train_label_var))
np.save(os.path.join(dataSet_dir, 'val_label_var.npy'), np.array(val_label_var))