# -*- coding: utf-8 -*-
# !/usr/bin/env python36
"""
    tgshg/aispeech/api_aispeech.py
    :copyright:facegood © 2019 by the tang.
    url: https://help.tgenie.cn/#/ba_token
"""
import requests
import hashlib
import time
import json


'''
this class is used to connect to speech server and ask for text chat and tts services.
该类主要用于思必驰云端的文本交互及tts功能的使用。
主体功能包括
1.确保token的有效性。（update_token)
2.文本对答，根据输入文本返回答案文本。(chat)
3.将文本转换成对应参数要求的语音。(tts/dm_tts)
'''

class AiSpeech(object):
    '''
    input params that will use to connect to speech server.
    params:
        productId,publicKey and secretKey are provide by SPEECH(思必驰)

    '''
    def __init__(self,productId,publicKey,secretKey,productIdChat=None,token = None,expireTimeSecs=5):
        self.productId = productId
        self.publicKey = publicKey
        self.secretKey = secretKey
        if productIdChat is None:
            self.productIdChat = productId
        else:
            self.productIdChat = productIdChat
        self.token = token
        self.expireTime = None
        # 过期时间 5秒前更新token
        self.expireTimeSecs = expireTimeSecs

    '''
        this function is use to update the token str before expired
        这个函数主要用于更新过期token字符串
    '''

    def update_token(self,url = None):
        if url is None:
            url = "https://api.talkinggenie.com/api/v2/public/authToken"

        headers = {'Content-Type': 'application/json;charset=UTF-8'}
        requests_body = {}
        timeStamp = str(int(time.time()*1000)) # 1565589691183
        sign = hashlib.md5((self.publicKey+self.productId+timeStamp+self.secretKey).encode('utf-8')).hexdigest()

        requests_body['productId'] = self.productId
        requests_body['publicKey'] = self.publicKey
        requests_body['sign'] = sign
        requests_body['timeStamp']=timeStamp

        r_token = requests.post(url, headers = headers, data=json.dumps(requests_body))
        if r_token.status_code != 200:
            print("error requests post url:",url,"\nheaders:",headers,"\nrequest body:",requests_body,"\ncode:",r_token.status_code)
        r_data = json.loads(r_token.text)
        if r_data['code'] == '200':
            self.token = r_data['result']['token']
            self.expireTime = int(r_data['result']['expireTime'])
            return self.expireTime
        else:
            print("ERROR:",r_data)
            return False

    '''
    chat function,send question text  to the server and receive the answer text.
    2.文本对答，根据输入文本返回答案文本。(chat)
    param:
        url : the remote api port url
        text: the question text str
    '''

    def chat(self,url = None,text=None):
        if url is None:
            url = "https://api.talkinggenie.com/api/v1/ba"
        # 判断是否过期
        if self.expireTime < int((time.time()+self.expireTimeSecs)*1000):
            self.update_token()
        headers = {}
        headers['Content-Type'] = 'application/json; charset=utf-8'
        headers['X-AISPEECH-TOKEN']= self.token
        headers['X-AISPEECH-PRODUCT-ID'] = self.productId

        requests_body = {}
        query = {
            "type":"text",
            "text":text
        }
        context= {
                "session":"1",
                "recordId":"100"
        }
        dialog={
                "productId":self.productIdChat
            }
        output = {
            "type":"text"
        }
        requests_body['query'] = query
        requests_body['dialog'] = dialog
        requests_body['context'] = context
        requests_body['output'] = output
        # print("text chat***********:",json.dumps(requests_body))
        r_chat = requests.post(url, headers = headers, data=json.dumps(requests_body))
        if r_chat.status_code != 200:
            print("error requests post url:",url,"\nheaders:",headers,"\nrequest body:",requests_body,"\ncode:",r.status_code)
            return False
        r_data = json.loads(r_chat.text)

        try:
            if r_data["status"] == 200:
                question = r_data['result']['query']
                answer = r_data['result']['answer']
                if question == text:
                    return answer
        except Exception as error:
            print(error)
            print("ERROR:query is not this answer\n","query is:",text,"request is:",r_data)
            return False

    '''
    tts function use to input text and return the audio data
    3.将文本转换成对应参数要求的语音。(tts/dm_tts)
    param :
        url :the remote api url .
        text : text str wait for transmit.
        speaker : the sounds owner id.

    '''

    def tts(self,url = None,text=None,speaker="zsmeif"):
        if url is None:
            url = "https://api.talkinggenie.com/api/v1/ba/tts"

        # 判断是否过期
        if self.expireTime < int((time.time()+self.expireTimeSecs)*1000):
            self.update_token()

        headers = {}
        headers['Content-Type'] = 'application/json;charset=UTF-8'
        headers['X-AISPEECH-TOKEN']= self.token
        headers['X-AISPEECH-PRODUCT-ID'] = self.productId

        requests_body = {}

        tts = {
            "speed": 1.1,
            "volume": 100,
            "voiceId": speaker,
            "enableRealTimeFeedback": False,
            "text": text
        }
        audio = {
            "audioType": "wav",
            "sampleRate": "16000",
            "sampleBytes": 2
        }

        requests_body['tts'] = tts
        requests_body['audio'] = audio
        requests_body['type'] = "tts"

        r_tts = requests.post(url, headers = headers, data=json.dumps(requests_body))
        if r_tts.status_code == 200:
            return r_tts.content
        else:
            print("ERROR:tts is failed\n","text is:",text)
    
    '''
    dm_tts function use to input text and return the audio data
    3.将文本转换成对应参数要求的语音。(tts/dm_tts)
    param :
        url :the remote api url .
        text : text str wait for transmit.
        speaker : the sounds owner id.

    '''    
    def dm_tts(self,url = None,text=None,speaker="zsmeif"):
        if url is None:
            url = "https://api.talkinggenie.com/api/v1/ba"

        headers = {}
        headers['Content-Type'] = 'application/json;charset=UTF-8'
        headers['X-AISPEECH-TOKEN']= self.token
        headers['X-AISPEECH-PRODUCT-ID'] = self.productId

        requests_body = {
            "query":{
                "type":"text",
                "text":text
            },
            "tts":{
                "speed": 1.1,
                "volume": 100,
                "voiceId": speaker
            },
            "dialog":{
                "productId":self.productIdChat
            },
            "output":{
                "audio": {
                    "audioType": "wav",
                    "channel":1,
                    "sampleRate": "16000",
                    "sampleBytes": 2
                    },
                "type": "tts"
            }
        }
        
        r_dm_tts = requests.post(url, headers = headers, data=json.dumps(requests_body))
        return r_dm_tts


if __name__ == "__main__":
    productId = "914008290"
    publicKey = "c315edc2bab941cbae6a3591a06281bc"
    secretkey ="81D620703D63A63A200AEB94125FBAFB"
    productIdChat = "914008349"

    ai = AiSpeech(productId,publicKey,secretkey,productIdChat)
    ai.update_token()
    print(ai.token)

    dm_tts = ai.dm_tts(text = "你是谁？")
    
    dm_tts.status_code
    dm_tts

    chat_text = ai.chat(text="你是谁？")

    b_wav_data = ai.tts(text = chat_text)

    text_self = "您好！我是子书美，来自数字虚拟世界。我可以告诉你你想知道的一切，想与我面对面聊天吗？12月5日到8日来第十届中国国际新媒体短片节找我！"
    b_wav_data = ai.tts(text = text_self,speaker = "lchuam")

    import wave
    wav_path = "G:/test.wav"
    f = open(wav_path,"wb")
    f.write(b_wav_data)
    f.close()
    signal = b_wav_data[44:]


