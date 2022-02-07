import numpy as np
import pandas as pd

shirley_1015_bs_name = np.load(r'D:\voice2face\shirley_1015\shirley_1015_bs_name.npy')
shirley_1119_bs_name = np.load(r'D:\voice2face\shirley_1015\shirley_1119_bs_name.npy')
shirley_1119_bs_name316 = np.load(r'D:\voice2face\shirley_1119\shirley_1119_bs_name316.npy')
bs_value_1114_3_16 = np.load(r'D:\voice2face\shirley_1119\bs_value\bs_value_1114_3_16.npy')

print(bs_value_1114_3_16.shape)

weights1 = np.zeros((bs_value_1114_3_16.shape[0],len(shirley_1119_bs_name)))
bs_name_index = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 105, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 1, 115]
for i in range(len(bs_name_index)):
	weights1[:,i] = bs_value_1114_3_16[:,bs_name_index[i]]

# 导出权重的csv文件
import pandas as pd
df = pd.DataFrame(weights1,columns=shirley_1119_bs_name)
df.to_csv(r'D:\voice2face\shirley_1119\1.csv',index=0)

# print(len(shirley_1015_bs_name))
# print(len(shirley_1119_bs_name))
# print(len(shirley_1119_bs_name316))
# if shirley_1119_bs_name316[4]==shirley_1119_bs_name[4]:
# 	print(1)
# if shirley_1119_bs_name316[4]==shirley_1015_bs_name[4]:
# 	print(2)
# same_bs = 0
# for i in range(len(shirley_1015_bs_name)):
# 	if shirley_1119_bs_name316[i] == shirley_1119_bs_name[i]:
# 		same_bs += 1
# 		# print(i)
# 		# print(shirley_1015_bs_name[i])
# print(same_bs)
# # bs_name_dict = {}
# # bs_name_list = []
# # for i in range(len(shirley_1119_bs_name)):
# # 	bs_name_dict[i] = np.where(shirley_1015_bs_name == shirley_1119_bs_name[i])[0][0]
# # 	bs_name_list.append(np.where(shirley_1015_bs_name == shirley_1119_bs_name[i])[0][0])
# # print(bs_name_dict)
# # print(bs_name_list)


# # # print(shirley_1015_bs_name)
# # # print(shirley_1119_bs_name)
