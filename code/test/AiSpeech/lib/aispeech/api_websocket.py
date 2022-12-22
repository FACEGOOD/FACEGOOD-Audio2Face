# -*- coding: utf-8 -*-
# !/usr/bin/env python36
"""
    tgshg/aispeech/api_aispeech.py
    :copyright:facegood © 2019 by the tang.
    url: https://help.tgenie.cn/#/ba_asr_websocket
"""
# pip3.8 install websocket-client
from websocket import create_connection
import threading


"""
this class is used to translate audio data to text.In order to achieve this function,
we also use th Speech's remote service.
该类主要用于语音识别，将语音转换成对应的文本。
"""

class AiSpeechWebSocket:
    def __init__(self,url, request_body):
        self.url = url
        self.request_body = request_body
        self.ws = None

    def __del__(self):
        pass
        # if self.ws:
        #     self.close_iat_asr()

    '''
    create connection with remote server
    与云端服务器建立连接
    params:
        timeout: time for wait
        enable_multithread : if enable multi thread to create connection.
    '''
    def ws_asr_create_connection(self,timeout=5,enable_multithread = True):
        try:
            self.ws = create_connection(self.url,timeout=timeout, enable_multithread=enable_multithread)
            return True
        except Exception as err:
            print("ERROR:", err)
            return False


    '''
        send audio data to remote server
        发送语音数据
        params:
            data ：audio data
            send_status : if send_status = 1, is json format data. else if send_status = 2 ,is binary data.
    '''
    def send_ws_asr_data(self, data=b'', send_status=0):
        flag = None
        if send_status == 2:
            flag = self.ws.send_binary(data)
        elif send_status == 1:
            flag = self.ws.send(self.request_body)
        elif send_status == 3:
            flag = self.ws.send_binary("")
        return flag

    '''
        get text from remote server
        获取语音对应的文本
        return text
    '''

    def get_text_from_ws_asr(self):
        # while True:
        message = self.ws.recv()
        return message

    '''
        close the remote connection
        关闭连接
    '''

    def close_ws_asr(self):
        self.ws.close()

    def _send_ping(self, interval, event):
        while not event.wait(interval):
            if self.ws:
                try:
                    self.ws.ping()
                    print("ping")
                except Exception as ex:
                    print("send_ping routine terminated: {}".format(ex))
                    break
    
    def ping(self,ping_interval=0):
        if ping_interval:
            event = threading.Event()
            thread = threading.Thread(
                target=self._send_ping, args=(ping_interval, event))
            thread.setDaemon(True)
            thread.start()
    
    def ping_one(self):
        if self.ws:
            self.ws.ping()


if __name__ == "__main__":
    import time
    import json
    import os
    #**********************************************#
    # your setting
    productId = "914008290"
    token = "ba5bf192-ccec-4128-9edd-b75aa25ad01d"
    audio_file = "G:/zisumei.wav"
    sampleRate = 16000
    buf_size = 3200
    url = "ws://api.tgenie.cn/runtime/v3/recognize?res=comm&productId="+productId + "&token=" + token
    #**********************************************#

    request_body = {"asr":{"res":"comm","lmld":"914008290_DuanPianJie_20191126","enablePunctuation": True,"language": "zh-CN"},
                    "audio": {"audioType": "wav","sampleRate": 16000,"channel": 1,"sampleBytes":2},
                    "dialog":{"productId":914008290}
                    }
    request_body_json=json.dumps(request_body)

    asr = AiSpeechWebSocket(url,request_body_json)
    flag = asr.ws_asr_create_connection()
    if not flag:
        exit()

    asr.ping(ping_interval = 1)
    audio_file = r"F:\work\Synchronize\Pro\AiSpeech\res\xxx_00001.wav"
    with open(audio_file,"rb") as fb:
        asr.send_ws_asr_data(send_status=1)
        while True:
            buf = fb.read(buf_size)
            if(len(buf)):
                asr.send_ws_asr_data(data = buf,send_status=2)
                time.sleep(0.1)
            else:
                break
        asr.send_ws_asr_data(send_status=3)

    # print message
    get_time = time.time()
    text = asr.get_text_from_ws_asr()
    print("get time:",time.time() - get_time)
    print(text)
    r_tts = json.loads(text)
    if r_tts['status'] != 200:
        print("error tts ")

    print(text)
    print(r_tts['result']['asr']['text'])
    asr.close_iat_asr()
    os.system("pause")