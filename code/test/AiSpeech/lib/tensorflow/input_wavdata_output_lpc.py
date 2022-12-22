# -*- coding: utf-8 -*-
"""
    sound_animation/lib.input_wavdata_output_lpc
    ~~~~~
    :copyright:facegood © 2019 by the tang.

"""
import os
from ctypes import *
import numpy as np

'''
    load LPC.dll dynamically
'''
dll_path_win = "./lib/tensorflow/LPC.dll"
dll_path_linux = dll_path_win.replace('\\', '/')
dll = cdll.LoadLibrary(dll_path_linux)
LPC = dll.LPC

'''
get_audio_frames:
    this function separate audio data to audio frame datas.
    params:
        audio_data: audio data,bytes.
        rate : 16000 is default,this is sample rate.
        frames_per_second : default 30, frames per second.
        chunks_length :default 260. according to the theory
'''

def get_audio_frames(audio_data, rate=16000, frames_per_second = 30,chunks_length = 260):
    # 取得音频文件采样频率
    # 取得音频数据
    signal = audio_data
    # signal = list(signal)

    # 视频fps frames_per_second
    # 音频分割，520ms chunks_length
    audio_frameNum = int(len(signal) / rate * frames_per_second)  # 计算音频对应的视频帧数

    # 前后各添加260ms音频
    a = np.zeros(chunks_length * rate // 1000, dtype=np.int16)
    signal = np.hstack((a, signal, a))

    # if signal.dtype == 'int16':
    signal = signal / (2. ** 15)
    frames_step = 1000.0 / frames_per_second  # 视频每帧的时长间隔33.3333ms
    rate_kHz = int(rate / 1000)  # 采样率：48kHz

    # 分割音频
    audio_frames = [signal[int(i * frames_step * rate_kHz): int((i * frames_step + chunks_length * 2) * rate_kHz)] for i in range(audio_frameNum)]

    return audio_frames

'''
c_lpc:
    make the audio frame datas to lpc datas.
    For the audio frame data, perform linear 
    predictive coding (Linear Predictive Coding, LPC) 
    to extract the first K autocorrelation coefficients of the audio.

    params:
        audio_frames_data: audio frame datas,get from funciton get_audio_frames.
        rate : 16000 is default,this is sample rate.
        frames_per_second : default 30, frames per second.
        chunks_length :default 260. according to the theory

    return:
     LPC frame datas.
'''
def c_lpc(audio_frames_data, rate=16000, frames_per_second = 30,chunks_length = 260, overlap_frames_apart=0.008,
          numberOfFrames=64):
    inputData_array = np.zeros(shape=(1, 32,
                                      64))  # 创建一个空3D数组，该数组(1*32*64)最后需要删除
    overlap = int(rate * overlap_frames_apart)  # 128 samples
    frameSize = int(rate * overlap_frames_apart * 2)  # 256 samples
    frames = np.ndarray((numberOfFrames, frameSize))

    for i in range(len(audio_frames_data)):
        # print(i)
        audio_frame = audio_frames_data[i]  # 每段音频，8320个采样点
        for k in range(0, numberOfFrames):
            frames[k, :] = audio_frame[k * overlap:k * overlap + frameSize]

        k = numberOfFrames - 1
        for i in range(0, frameSize):
            if ((k * overlap + i) < len(audio_frame)):
                frames[k][i] = audio_frame[k * overlap + i]
            else:
                frames[k][i] = 0

        frames *= np.hanning(frameSize)
        frames_lpc_features = []
        a = (c_double * frameSize)()
        b = (c_double * 32)()
        for k in range(0, numberOfFrames):
            a = (c_double * frameSize)(*frames[k])
            LPC(pointer(a), frameSize, 32, pointer(b))
            frames_lpc_features.append(list(b))

        image_temp1 = np.array(frames_lpc_features)  # list2array
        image_temp2 = image_temp1.transpose()  # array转置
        image_temp3 = np.expand_dims(image_temp2, axis=0)  # 升维
        inputData_array = np.concatenate((inputData_array, image_temp3),
                                         axis=0)  # array拼接

    # 删除第一行
    inputData_array = inputData_array[1:]

    # 扩展为4维:9000*32*64*1
    inputData_array = np.expand_dims(inputData_array, axis=3)
    return inputData_array


if __name__ == "__main__":
    import scipy.io.wavfile as wavfile
    wav_path = "G:/input_wav.wav"
    rate,signal = wavfile.read(wav_path)
    len(signal)/16000

    audio_frame = get_audio_frames(signal)
    len(audio_frame)/30

    output = c_lpc(audio_frame)
    save_file_path = "G:/out.npy"
    np.save(save_file_path,output)