import numpy as np
import torch
import torch.nn as nn
class net_torch(nn.Module):
    def __init__(self,outputSize,keep_pro):
        super(net_torch, self).__init__()
        self.E = 16
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=72,kernel_size=(3,1),padding=(1,0),stride=(2,1))
        self.conv2 = nn.Conv2d(in_channels=72,out_channels=108,kernel_size=(3,1),padding=(1,0),stride=(2,1))
        self.conv3 = nn.Conv2d(in_channels=108,out_channels=162,kernel_size=(3,1),padding=(1,0),stride=(2,1))
        self.conv4 = nn.Conv2d(in_channels=162,out_channels=243,kernel_size=(3,1),padding=(1,0),stride=(2,1))
        self.conv5 = nn.Conv2d(in_channels=243,out_channels=256,kernel_size=(2,1),padding=(0,0),stride=(2,1))
        self.relu = nn.ReLU()

        self.emotion_input = nn.Conv2d(in_channels=1,out_channels=self.E,kernel_size=(3,1),padding=(0,0),stride=(32,1))

        self.conv6 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(1,3),padding=(0,1),stride=(1,2))
        self.emotion1 = nn.Conv2d(in_channels=self.E,out_channels=self.E,kernel_size=(1,3),padding=(0,1),stride=(1,2))

        self.conv7 = nn.Conv2d(in_channels=256 + self.E,out_channels=256,kernel_size=(1,3),padding=(0,1),stride=(1,2))
        self.emotion2 = nn.Conv2d(in_channels=self.E,out_channels=self.E,kernel_size=(1,3),padding=(0,1),stride=(1,4))

        self.conv8 = nn.Conv2d(in_channels=256+self.E,out_channels=256,kernel_size=(1,3),padding=(0,1),stride=(1,2))
        self.emotion3 = nn.Conv2d(in_channels=self.E,out_channels=self.E,kernel_size=(1,3),padding=(0,1),stride=(1,8))

        self.conv9 = nn.Conv2d(in_channels=256+self.E,out_channels=256,kernel_size=(1,3),padding=(0,1),stride=(1,2))
        self.emotion4 = nn.Conv2d(in_channels=self.E,out_channels=self.E,kernel_size=(1,3),padding=(0,1),stride=(1,16))

        self.conv10 = nn.Conv2d(in_channels=256+self.E,out_channels=256,kernel_size=(1,3),padding=(0,1),stride=(1,4))
        self.emotion5 = nn.Conv2d(in_channels=self.E,out_channels=self.E,kernel_size=(1,3),padding=(0,1),stride=(1,64))

        self.fc1 = nn.Linear(in_features=256 + self.E,out_features=150)
        self.dropout = nn.Dropout(keep_pro)
        self.output= nn.Linear(in_features=150,out_features=outputSize)

    def forward(self,input_data):
        # input_data : (bs,1,32,64)
        x = self.relu(self.conv1(input_data)) # (bs,72,16,64)
        x = self.relu(self.conv2(x)) # (bs,108,8,64)
        x = self.relu(self.conv3(x)) # (bs,162,4,64)
        x = self.relu(self.conv4(x)) # (bs,243,2,64)
        x = self.relu(self.conv5(x)) # (bs,256,1,64)

        emotion_input = self.relu(self.emotion_input(input_data)) # (bs,self.E,1,64)

        x = self.relu(self.conv6(x)) # (bs,256,1,32)
        emotion = self.relu(self.emotion1(emotion_input)) # (bs,self.E,1,32)
        mixed = torch.cat((x,emotion),dim=1)

        x = self.relu(self.conv7(mixed))
        emotion = self.relu(self.emotion2(emotion_input))
        mixed = torch.cat((x,emotion),dim=1)

        x = self.relu(self.conv8(mixed))
        emotion = self.relu(self.emotion3(emotion_input))
        mixed = torch.cat((x,emotion),dim=1)

        x = self.relu(self.conv9(mixed))
        emotion = self.relu(self.emotion4(emotion_input))
        mixed = torch.cat((x,emotion),dim=1)

        x = self.relu(self.conv10(mixed))
        emotion = self.relu(self.emotion5(emotion_input))
        mixed = torch.cat((x,emotion),dim=1)

        flat = torch.flatten(mixed,start_dim=1)
        fc1 = self.fc1(flat)
        fc1 = self.dropout(fc1)
        output = self.output(fc1)

        return output,emotion_input


class loss_torch(nn.Module):
    def __init__(self):
        super(loss_torch, self).__init__()
    def forward(self,y,y_,emotion_input):
        # 计算loss_P
        loss_P = torch.mean((y-y_) * (y-y_))

        # 计算loss_M
        split_y = torch.split(y,2,0)
        split_y_ = torch.split(y_,2,0)
        y0 = split_y[0]
        y1 = split_y[1]
        y_0 = split_y_[0]
        y_1 = split_y_[1]
        loss_M = 2.0 * torch.mean((y0-y1-y_0+y_1)*(y0-y1-y_0+y_1))

        # 计算loss_R
        split_emotion_input = torch.split(emotion_input,2,0)
        emotion_input0 = split_emotion_input[0]
        emotion_input1 = split_emotion_input[1]
        Rx0 = (emotion_input0 - emotion_input1) * (emotion_input0 - emotion_input1)
        Rx1 = torch.sum(Rx0,dim=1)
        Rx2 = torch.sum(Rx1,dim=1)
        Rx3 = 2.0 * torch.mean(Rx2,dim=1)

        e_mean0 = torch.sum(emotion_input0 * emotion_input0,dim=2)
        e_mean1 = torch.mean(e_mean0)
        Rx = Rx1 / e_mean1

        loss_R = torch.mean(Rx)

        loss = loss_P + loss_M + loss_R
        return loss