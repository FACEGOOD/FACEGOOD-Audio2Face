#!/usr/bin/python
# -*- coding: UTF-8 -*-

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


"""
    AiSpeech/main.py
    ~~~~~
    :copyright:facegood © 2019 by the tang.

"""
import os
import sys
import time
from os.path import abspath, dirname, join
import codecs
import json
# *******************************************
# *******************************************
package_path = "./"

# speakers = ["zsmeif"]
# *******************************************
# load config file
path_aispeech_config = join(package_path,"zsmeif_aispeech_config.json")
try:
    with codecs.open(path_aispeech_config, 'r', 'utf-8-sig') as fconfig:
        AiSpeechConfig = json.load(fconfig)
except Exception as err:
    print("Read file failed,",path_aispeech_config,".Error is :",err)
    os.system("pause")
    exit(1)


# *******************************************
# read parsing config data
productId = AiSpeechConfig['api_key']['productId']
publicKey = AiSpeechConfig['api_key']['publicKey']
secretkey = AiSpeechConfig['api_key']['secretKey']
productIdChat = AiSpeechConfig['api_key']['productIdChat']
SPEAKER = AiSpeechConfig['api_key']['speaker']

BA_URL = AiSpeechConfig['api_ba']['url']
WSURL = AiSpeechConfig['api_websocket']['url']+productId + "&token="
request_body_json=json.dumps(AiSpeechConfig['api_websocket']['request_body_first'])

info_print = AiSpeechConfig['config']['print']
ID_SESSION = AiSpeechConfig['config']['session']
FPS = AiSpeechConfig['config']['fps']
SPEED_PLAY = float(1.0 / FPS)

# *******************************************
# *******************************************
import lib.socket.ue4_socket as ue4
# read udp server and client config data
ADDR_BIND = (AiSpeechConfig['config']['server']['ip'],AiSpeechConfig['config']['server']['port'])
ADDR_SEND = (AiSpeechConfig['config']['client']['ip'],AiSpeechConfig['config']['client']['port'])

ue4.BUFF_SIZE = AiSpeechConfig['config']['ue4']['recv_size']
ue4.RECORDING = True
ue4.RECORDING_BEGIN = AiSpeechConfig['config']['ue4']['begin'].encode('utf-8')
ue4.RECORDING_END = AiSpeechConfig['config']['ue4']['end'].encode('utf-8')

# *******************************************
# *******************************************

# useless bs weight indices
const_bs_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 39, 46, 47, 48, 49, 50, 55, 56, 61, 62, 65, 70, 71, 72, 73, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
# useful bs weight indices
var_bs_index = [10, 13, 14, 15, 18, 33, 38, 40, 41, 42, 43, 44, 45, 51, 52, 53, 54, 57, 58, 59, 60, 63, 64, 66, 67, 68, 69, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84]
# default useless weight value
const_bs_value = [0.,0.,-0.,0.,-0.,0.,-0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-0.,-0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.,-0.,0.]
# the sort of bs name correspond to UE input sort
bs_name_index = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 105, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 1, 115]

BS_CONUNT = 116
# pbfile_path = join(package_path, 'zsmeif.pb')

CPU_Thread = AiSpeechConfig['config']['tensorflow']['cpu']
CPU_Frames = AiSpeechConfig['config']['tensorflow']['frames']

# *******************************************
# *******************************************
import numpy as np

from lib.audio.api_audio import AudioRecognition, AudioPlay
from lib.tensorflow.input_wavdata_output_lpc import c_lpc, get_audio_frames
from lib.tensorflow.input_lpc_output_weight import WeightsAnimation

tflitepath = './models/Audio2Face.tflite'
model_path = './models/Audio2Face'
#load tensorflow pb model file
pb_weights_animation = WeightsAnimation(tflitepath, model_path)
get_weight = pb_weights_animation.get_weight

# *******************************************
# work thread function
# use to deal with audio data to lpc data ,and input lpc data to get weights.
def worker(q_input, q_output, i):
    print("the cpus number is:", i)
    while True:
        input_data = q_input.get()
        for output_wav in input_data:
            # audio frame to lpc datas
            output_lpc = c_lpc(output_wav)
            #lpc data to weights
            output_data = get_weight(output_lpc)
            # 赋值
            weights = np.zeros((output_data.shape[0],BS_CONUNT))
            # print(weights.shape)

            weights[:,var_bs_index] = output_data
            weights[:,const_bs_index] = const_bs_value

            weights1 = np.zeros((output_data.shape[0],BS_CONUNT))
            for i in range(len(bs_name_index)):
                weights1[:,i] = weights[:,bs_name_index[i]]

            q_output.put(weights1)


# *******************************************
import threading
from queue import Queue


'''
    mutli thread deal with audio to weights

'''

class SoundAnimation:
    def __init__(self,cpus = 1,input_nums = 30):
        self.cpus = cpus
        self.input_nums = input_nums
        self.init_multiprocessing()
        self.flag_start = False

    def __del__(self):
        if self.flag_start:
            self.stop_multiprocessing()

    # init  threads
    def init_multiprocessing(self):
        self.q_input = [Queue() for i in range(0, self.cpus)]
        self.q_output = [Queue() for i in range(0, self.cpus)]
        self.process = []
        for i in range(0, self.cpus):
            self.process.append(
                threading.Thread(target=worker, args=(self.q_input[i], self.q_output[i], i)))

    #start threads
    def start_multiprocessing(self):
        self.flag_start = True
        for i in range(0, self.cpus):
            self.process[i].setDaemon(True)
            self.process[i].start()
    
    #stop threads
    def stop_multiprocessing(self):
        for i in range(0, self.cpus):
            self.process[i].terminate()
    
    #input audio frame data
    def input_frames_data(self, input_date):
        input_data_nums = [input_date[i:i + self.input_nums] for i in range(0, len(input_date), self.input_nums)]
        self.flag_nums = len(input_data_nums)
        for i in range(0, self.cpus):
            self.q_input[i].put(input_data_nums[i::self.cpus])

    def yield_output_data(self):
        num = 0
        flag_end = True
        while flag_end:
            for i in range(0, self.cpus):
                if num == self.flag_nums:
                    flag_end = False
                    break
                data_output = self.q_output[i].get()
                for data in data_output:
                    yield data
                num += 1


# *******************************************
# *******************************************
from lib.audio.api_audio import AudioRecognition
from lib.aispeech.api_aispeech import AiSpeech
from lib.aispeech.api_websocket import AiSpeechWebSocket


'''
    main function is a work process.


'''

def main(fun_socket_send):
    #**********************************************#
    # token 过期时间
    # check if token is expired
    expireTime = aispeech.update_token()
    if not aispeech.token:
        print("Eerrr: get token, please wait a moment")
        # exit(1)
    url = WSURL + aispeech.token
    #init ars websocket
    asr_websocket = AiSpeechWebSocket(url,request_body_json)
    expireTimeSecs = 300

    while True:
        if expireTime < int((time.time()+expireTimeSecs)*1000):
            print("getToken")
            expireTime = aispeech.update_token()
            if not aispeech.token:
                print("Eerrr: get token, please wait a moment")
                # exit(1)
            asr_websocket.url = WSURL + aispeech.token
        time.sleep(0.01)

        #if click button is hold in ue project
        while ue4.RECORDING:
            # 1 asr create connection with AiSpeech
            num = 0
            flag = asr_websocket.ws_asr_create_connection(timeout=20)
            if not flag:
                print("ERROR: Could not create connection:",url)
                exit()
            #send audio data to get text
            asr_websocket.send_ws_asr_data(send_status=1)
            while ue4.RECORDING:
                if info_print:
                    print("Recording:",num)
                num += 1
                buf=record.recording()
                asr_websocket.send_ws_asr_data(data = buf,send_status=2)
                # this will break when recording is end
            # send finish status 
            asr_websocket.send_ws_asr_data(send_status=3)
            if info_print:
                print("wait get asr:")
                get_time = time.time()
            # get answer text from asr
            text = asr_websocket.get_text_from_ws_asr()
            if info_print:
                print("asr time:",time.time()-get_time)
            r_asr_websocket = json.loads(text)
            if r_asr_websocket['status'] == 200:
                text_asr = r_asr_websocket['result']['asr']['text']
                if text_asr:
                    if info_print:
                        print("asr:",text_asr)
                        print("wait get tts:")
                        dm_tts_time = time.time()
                    #answer text to audio data
                    dm_tts = aispeech.dm_tts(url=BA_URL,text=text_asr,speaker= SPEAKER)
                    if info_print:
                        print("tts time is:",time.time()-dm_tts_time)
                    if dm_tts.status_code == 200:
                        b_wav_data = dm_tts.content
                        if b_wav_data:
                            # 4 animation
                            def play_audio_animation():
                                voice = np.frombuffer(b_wav_data[44:], dtype=np.int16)
                                #audio data to frames
                                input_data = get_audio_frames(voice)
                                try:
                                    #input audio data to get weights data
                                    sound_animation.input_frames_data(input_data)
                                    is_first = True
                                    f_num = 0
                                    f_btime = time.time()
                                    for weight in sound_animation.yield_output_data():
                                        f_num += 1
                                        # f_time = time.time()
                                        #play audio and send weights data at the same time
                                        if is_first:
                                            player.play_audio_data_thread(b_wav_data[44:])
                                            f_btime = time.time()
                                            is_first = False
                                        fun_socket_send(weight)
                                        time.sleep(SPEED_PLAY*f_num-(time.time()-f_btime))
                                        # print(f_num,":frame:",time.time()-f_time)
                                except Exception as err:
                                    print("Sound animation type error: ", err)
                                    # break
                            voice_thread = threading.Thread(target=play_audio_animation)
                            voice_thread.setDaemon(True)
                            voice_thread.start()
                            voice_thread.join()
                        # end b_wav_data
                # end text_asr
            else:
                print("ERROR:",r_asr_websocket)
            # end asr
            if info_print:
                print("wait recording")
        # end while RECORDING:
    # end while True:

# *******************************************
if __name__ == "__main__":
    udp_recv_handler = ue4.UdpRecvHandler(ADDR_BIND)
    udp_send_handler = ue4.UdpSendHandler(ADDR_SEND)
    #***************tensorflow*******************************#
    player = AudioPlay()
    sound_animation = SoundAnimation(CPU_Thread, CPU_Frames)
    sound_animation.start_multiprocessing()
    #****************aispeech******************************#
    record = AudioRecognition()
    aispeech = AiSpeech(productId,publicKey,secretkey,productIdChat=productIdChat)
    #****************main******************************#
    while True:
        print("run main")
        try:
            main(udp_send_handler.send_handler)
        except Exception as error:
            print("Error Main loop:",error)
            time.sleep(60)
    #****************main******************************#
    print("# *******************************************")

