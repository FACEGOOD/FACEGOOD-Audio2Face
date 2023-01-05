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

# encoding=utf-8
import os
import numpy as np
import scipy.io.wavfile as wavfile
# from audiolazy.lazy_lpc import lpc
from ctypes import *

# import time

project_dir = r'./'
dll = cdll.LoadLibrary(os.path.join(project_dir, 'LPC.dll'))  # 加载 LPC.dll
# wav_path = os.path.join(project_dir,'wav','1015_2_01.wav') # 音频路径
# save_path = os.path.join(project_dir,'lpc','lpc_1015_2_01.npy') # 保存LPC处理后的数组
name_list = ['1_01', '1_02', '2_01', '2_02', '2_03', '2_04', '2_05', '2_06', '2_07', '2_08', '2_09', '2_10',
             '3_01', '3_02', '3_03', '3_04', '3_05', '3_06', '3_07', '3_08', '3_09', '3_10', '3_11', '3_12',
             '3_13', '3_14', '3_15', '3_16']
wav_path = [os.path.join(project_dir, 'wav', '1114_' + file + '.wav') for file in name_list]    # 音频路径
save_path = [os.path.join(project_dir, 'lpc', '1114_' + file + '.npy') for file in name_list]   # 保存LPC处理后的数组

if not os.path.exists(os.path.join(project_dir, 'lpc')):
    os.makedirs(os.path.join(project_dir, 'lpc'))

def audioProcess(wav_path):
    # 读取wav文件，存入list
    rate, signal = wavfile.read(wav_path)  # rate：采样率
    # print(len(signal))
    # print('signal:',signal[1000:2000])
    print('rate: ', rate)  # 采样率

    frames_per_second = 30  # 视频fps
    chunks_length = 260  # 音频分割，520ms
    audio_frameNum = int(len(signal) / rate * frames_per_second)  # 计算音频对应的视频帧数
    print('audio_frameNum: ', audio_frameNum)

    # 前后各添加260ms音频
    a = np.zeros(chunks_length * rate // 1000, dtype=np.int16)
    signal = np.hstack((a, signal, a))

    # signal = signal / (2.**15)
    frames_step = 1000.0 / frames_per_second  # 视频每帧的时长间隔33.3333ms
    rate_kHz = int(rate / 1000)  # 采样率：48kHz

    # 分割音频
    audio_frames = [signal[int(i * frames_step * rate_kHz): int((i * frames_step + chunks_length * 2) * rate_kHz)]
                    for i in range(audio_frameNum)]
    inputData_array = np.zeros(shape=(1, 32, 64))  # 创建一个空3D数组，该数组(1*32*64)最后需要删除

    for i in range(len(audio_frames)):
        print(i)
        audio_frame = audio_frames[i]  # 每段音频，8320个采样点

        overlap_frames_apart = 0.008
        overlap = int(rate * overlap_frames_apart)  # 128 samples
        frameSize = int(rate * overlap_frames_apart * 2)  # 256 samples
        numberOfFrames = 64

        # initiate a 2D array with numberOfFrames rows and frame size columns
        frames = np.ndarray((numberOfFrames, frameSize))
        for k in range(0, numberOfFrames):
            for i in range(0, frameSize):
                if ((k * overlap + i) < len(audio_frame)):
                    frames[k][i] = audio_frame[k * overlap + i]
                else:
                    frames[k][i] = 0

        frames *= np.hanning(frameSize)
        frames_lpc_features = []

        # a = (c_double*frameSize)()
        b = (c_double * 32)()

        # linear predictive coding
        for k in range(0, numberOfFrames):
            # temp_list = frames[k]
            a = (c_double * frameSize)(*frames[k])
            # a = (c_double*len(frames[k]))()
            # b = (c_double*32)()
            # LPC(float *in, int size, int order, float *out)
            dll.LPC(pointer(a), frameSize, 32, pointer(b))  # linear predictive coding
            frames_lpc_features.append(list(b))

        image_temp1 = np.array(frames_lpc_features)  # list2array
        image_temp2 = image_temp1.transpose()  # array转置
        image_temp3 = np.expand_dims(image_temp2, axis=0)  # 升维
        inputData_array = np.concatenate((inputData_array, image_temp3), axis=0)  # array拼接

    # 删除第一行
    inputData_array = inputData_array[1:]

    # #扩展为4维:(,32,64,1)
    inputData_array = np.expand_dims(inputData_array, axis=3)
    print(inputData_array.shape)

    return inputData_array


if __name__ == '__main__':
    for i in range(len(name_list)):
        inputData_array = audioProcess(wav_path[i])
        np.save(save_path[i], inputData_array)
