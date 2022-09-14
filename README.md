# Audio2Face

## 读取本地文件

运行`zsmeif_test.py`，其他步骤相同

## Notice

   The Test part and  The UE project for xiaomei created by FACEGOOD  is not available for commercial use.they are for testing purposes only.

## Description
---
![ue](rsc/ue.png)


We create a project that transforms audio to blendshape weights,and drives the digital human,xiaomei,in UE project.


## Base Module
---

![figure1](rsc/net.png)

![figure2](rsc/layers.png)


The framework we used contains three parts.In Formant network step,we perform fixed-function analysis of the input audio clip.In the articulation network,we concatenate an emotional state vector to the output of each convolution layer after the ReLU activation. The fully-connected layers at the end expand the 256+E abstract features to  blendshape weights .


## Usage
---
![pipeline](/rsc/pipeline.PNG)


this pipeline shows how we use FACEGOOD Audio2Face.



[Test video 1](https://www.youtube.com/watch?v=f6DcsZCsOWM&ab_channel=MicrosoftDeveloper)
[Test video 2](https://www.youtube.com/watch?v=bvT9yg3Uab4)
Ryan Yun from columbia.edu


### Prepare data

- step1: record voice and video ,and create animation from video in maya.
    note: the voice must contain vowel ,exaggerated talking and normal talking.Dialogue covers as many pronunciations as possible.
- step2: we deal the voice with LPC,to split the voice into segment frames corresponding  to the animation frames in maya.


### Input data

Use ExportBsWeights.py to export weights file from Maya.Then we can get BS_name.npy and BS_value.npy .

Use step1_LPC.py to deal with wav file to get lpc_*.npy .
Preprocess the wav to 2d data.

### train

we recommand that uses FACEGOOD avatary to produces trainning data.its fast and accurate.
http://www.avatary.com

the data for train is stored in dataSet1 

> python step14_train.py --epochs 8 --dataSet dataSet1

### test

In folder /test,we supply a test application named AiSpeech.  
wo provide a pretrained model,zsmeif.pb  
In floder /example/ueExample, we provide a packaged ue project that contains a digit human created by FACEGOOD can drived by /AiSpeech/zsmeif.py.

you can follow the steps below to use it:
1.  make sure you connect the microphone to computer.
2.  run the script in terminal. 
    > python zsmeif.py
3.  when the terminal show the message "run main", please run FaceGoodLiveLink.exe which is placed in /example/ueExample/ folder.
4.  click and hold on the left mouse button on the screen in UE project, then you can talk with the AI model and wait for the voice and animation response. 


## Dependences

tersorflow-gpu 1.15  cuda 10.0 

python-libs:
    pyaudio
    requests
    websocket
    websocket-client

note: test can run with cpu.

## Data
---

The testing data, Maya model, and ue4 test project can be downloaded from the link below.

[data_all](https://pan.baidu.com/s/1CGSzn639PUE7cUYnX4I3fQ) code : n6ty

[GoogleDrive](https://drive.google.com/drive/folders/1r7b7sfMebhtG0NSZk1yHzMaHRosb8xd1?usp=sharing)


## Update

uploaded the LPC source into code folder.

## Reference
---
[Audio-Driven Facial Animation by Joint End-to-End Learning of Pose and Emotion](chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/https://research.nvidia.com/sites/default/files/publications/karras2017siggraph-paper_0.pdf)

## Contact


<img src="rsc/fgcode.JPG" width = "200" height = "200" alt="" align=center />

Wechat: FACEGOOD_CHINA  
Email：support@facegood.cc  
Discord: https://discord.gg/V46y6uTdw8

## License

Audio2Face Core is released under the terms of the MIT license.See COPYING for more information or see https://opensource.org/licenses/MIT.
