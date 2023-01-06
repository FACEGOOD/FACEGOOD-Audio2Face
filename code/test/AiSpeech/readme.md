AiSpeech 
    lib 使用的模块存放
    res 项目参考测试资源
    script 各模块测试
    zsmeif.pb
    zsmeif_multiRun.py
    zsmeif_thread.py
    zsmeif.py exe打包多线程

zsmeif
    \FaceGoodLiveLink\Plugins\FaceGood\Content\BlendShape\bsname.bsname
zsmeif_py
    zsmeif\zsmeif.exe


aispeech_config.json
```json
"config":{
    "print":true, 是否显示输出打印，true打印，false不打印
    "session":10000, 对话识别id，
    "fps":30, 语音播放的fps速率
    "server":{"ip":"127.0.0.1","port":43015}, 接收录音开始结束ip和端口
    "client":{"ip":"127.0.0.1","port":43014}, 发送动画数据ip和端口
    "ue4":{"recv_size":1024,"begin":"1","end":"2"}, 一次接收数据大小，开始标示，结束标示
    "tensorflow":{"cpu":2,"frames":20} tensorflow多进程、线程的数量（cpu数）和一次处理的帧数
},
    "api_key":{
        "productId":"914008290", 会话精灵 id
        "publicKey":"c315edc2bab941cbae6a3591a06281bc", 会话精灵 key
        "secretKey":"81D620703D63A63A200AEB94125FBAFB", 会话精灵 key
        "productIdChat":"914008349", 会话精灵 智能闲聊id，具体设置参考2
        "speaker":"zsmeif" , 会话精灵 tts语音模型，zsmeif为子书美，lchuam为陆川

        "request_body_first": {"asr":{"res": "aicallcenter","lmld":"FESTIVAL_1122b_BA914008290_LM","enablePunctuation": true,"language": "zh-CN"},
        "audio": {"audioType": "wav","sampleRate": 16000,"channel": 1,"sampleBytes":2},
        "dialog":{"productId":914008290}
        },   ws请求头参数表
```

1，参考 aispeech会话精灵参考文档 https://help.tgenie.cn/#/ba_asr_websocket


3，账户信息
思必驰->会话精灵 
    PID 914008290
    Public Key：c315edc2bab941cbae6a3591a06281bc
    Secret Key：81D620703D63A63A200AEB94125FBAFB

4 python打包

C:\Anaconda3\Scripts\pyinstaller.exe F:\work\AiSpeech\exe\zsmeif.spec

完成后复制dist目录中的exe替换对应的exe即可
