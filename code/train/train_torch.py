import numpy as np
import torch
from model_torch import *
from torch.utils.data import Dataset,DataLoader,TensorDataset
import os
epochs = 10
dataSet = 'dataSet1'
project_dir = '/home/shaomingqi/projects/facegood'
data_dir = os.path.join(os.path.join(project_dir,'DataForAudio2Bs/train/'),dataSet)
logs_dir = os.path.join(project_dir,'logs')

x_train = torch.from_numpy(np.load(os.path.join(data_dir,'train_data.npy'))).float()
y_train = torch.from_numpy(np.load(os.path.join(data_dir,'train_label_var.npy'))).float()
x_val = torch.from_numpy(np.load(os.path.join(data_dir,'val_data.npy'))).float()
y_val = torch.from_numpy(np.load(os.path.join(data_dir,'val_label_var.npy'))).float()
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
train_dataset = TensorDataset(x_train,y_train)
test_dataset = TensorDataset(x_val,y_val)

# Training Parameters
batch_size = 128
starter_learning_rate = 0.001
output_size = y_val.shape[1]

# Create DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=False)

def train():
    # Carete dirs
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)

    model = net_torch(outputSize= y_val.shape[1],keep_pro=0.5).cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=starter_learning_rate)

    loss_fn = loss_torch()
    for i in range(epochs):

        # train loop
        for batch,(x,y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            x = torch.permute(x,(0,3,1,2)) # (bs,channel,W,H)
            y_pred,emotion_input = model(x)

            loss = loss_fn(y_pred,y,emotion_input)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if batch % 100 ==0:
            #     loss = loss.item()
            #     print("Epochs {0} Train loss:{1}".format(i,loss))


        # test loop
        num_batches = len(test_loader)
        test_loss = 0.0
        with torch.no_grad():
            for x,y in test_loader:
                x = x.cuda()
                y = y.cuda()
                x = torch.permute(x, (0, 3, 1, 2))
                y_pred,emotion_input = model(x)
                test_loss += loss_fn(y_pred,y,emotion_input).item()
        test_loss /= num_batches
        print("Epochs {0} Test Avg loss:{1}".format(i, test_loss))

    # Save Checkpoint
    chckpt_path = os.path.join(logs_dir,'model_torch_{0}.pth'.format(epochs))
    torch.save(model,chckpt_path)




if __name__ =="__main__":
    train()







