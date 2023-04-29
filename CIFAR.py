# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 19:44:44 2022

@author: user
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

import torch  #使用pytorch
import torch.nn as nn #使用pytorch 裡的NN架構
import torch.nn.functional as F #使用各種activation function
import torch.optim as optim #要使用pytorch裡的各種optimizers
from torch.optim import lr_scheduler
import torchvision #很多圖像處理會用到的function
import pandas as pd

# detect if a GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))
#%%

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
class CIFARdataset(torch.utils.data.Dataset):
    def __init__(self,mode,transform=None):
        self.transform = transform
        if mode=="train":
            pickle=unpickle("D:\上課\深度學習\DL_HW2\cifar-10-batches-py\data_batch_1")
            data=pickle[b'data']
            labels=pickle[b'labels']
            for  i in range(2,6):
                pickle=unpickle("D:\上課\深度學習\DL_HW2\cifar-10-batches-py\data_batch_"+str(i))
                data=np.concatenate((data,pickle[b'data']),axis=0)
                labels=np.concatenate((labels,pickle[b'labels']),axis=0)
        if mode=="test":
            pickle=unpickle("D:/上課/深度學習/DL_HW2/cifar-10-batches-py/test_batch")
            data=pickle[b'data']
            labels=pickle[b'labels']
            labels=np.array(labels)
        N,D=data.shape
        data=data.reshape(N,3,32,32)   #read the given data
        data=np.transpose(data,(0,2,3,1))   #to tensor need(HxWxC)

        self.data=data
        self.labels=labels
        self.len=N
    def __getitem__(self,index):
        if self.transform:
            return self.transform(self.data[index]),int(self.labels[index])
        return self.data[index],int(self.labels[index])
    def __len__(self):
        return self.len

#%%
load_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()                               
                                                 ])
trainset=CIFARdataset("train",load_transform)
testset=CIFARdataset("test",load_transform)
trainset, valset = torch.utils.data.random_split(trainset, [45000, 5000])

test , _=trainset[0]
plt.imshow(np.transpose(test,(1,2,0)))

trainset_loader = torch.utils.data.DataLoader(trainset,batch_size=512,shuffle=True)
valset_loader=torch.utils.data.DataLoader(valset,batch_size=512,shuffle=True)
testset_loader = torch.utils.data.DataLoader(testset,batch_size=512,shuffle=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#%%

#verify if import correctlly

dataiter = iter(trainset_loader)
imgs,labels=dataiter.next()

test , _=trainset[0]
plt.imshow(np.transpose(test,(1,2,0)))

img16=imgs[:16]
plt.imshow(img16[0][0].numpy())
test=torchvision.utils.make_grid(img16)
plt.imshow(np.transpose(test,(1,2,0)))
testla=labels[:16].numpy()
print(classes[i] for i in testla)
for i in testla:
    print(classes[i])
#%%
"""
Convolution Neuron Network
"""
class Net(nn.Module): #nn.Modeul: parent的class
  def __init__(self):
    super(Net, self).__init__()
    """
    below describe eaxh layer's parameters
        #notice that the order in init doesn't matter
        #the process sequece of Net is defined by forward()
    """
    self.conv1 = nn.Conv2d(in_channels=3, padding=1, out_channels=32, kernel_size=3) #output size=32*32*32
    self.conv2 = nn.Conv2d(in_channels=32,padding=1, out_channels=32, kernel_size=3) #output size=32*32*32
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #output size=32*16*16
    self.dropout1=nn.Dropout(p=0.25)
    self.conv3 = nn.Conv2d(in_channels=32, padding=1, out_channels=64, kernel_size=3)#output size=64*16*16
    self.conv4 = nn.Conv2d(in_channels=64,padding=1, out_channels=64, kernel_size=3)#output size=64*16*16
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output size= 64*8*8
    self.dropout2=nn.Dropout(p=0.25)
    self.fc1 = nn.Linear(in_features=64*8*8, out_features=1024) # 全連接層 提取圖片各個池的特性
    self.fc2 = nn.Linear(in_features=1024, out_features=10) #最後一個全連接層->logits
    self.dropout = nn.Dropout(p=0.5) #Avoid overfitting
    
  def forward(self, x):
    # convolution->ReLU->pooling
    x=F.relu(self.conv1(x))
    x = self.pool1(F.relu(self.conv2(x)))
    x=self.dropout1(x)
    x=F.leaky_relu(self.conv3(x))
    x = self.pool2(F.relu(self.conv4(x)))
    x=self.dropout2(x)
    
    x = x.view(-1, 64*8*8) #flatten to 1 diemention for full-connected-layer
    
    x = F.relu(self.fc1(x))# 全連接層
    x = self.dropout(x) #每個features 都有0.5的機率被deactive
    x = self.fc2(x) # 最後一層不用activate function
    
    return x  #return logits

     
#%%
def compute_acc(model, data_iter):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        inputs, labels = data_iter.next()
        
        # move data and labels to GPU if available
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        logits = model.forward(inputs)
        _, pred = torch.max(logits.data, 1)   #後面的如果是0回傳整column中最大的,1的話回傳整row最大的
                                              #第一個回傳最大值 第二個回傳該index
        # 用來計算訓練集的分類準確率
        total += labels.size(0) #取int
        correct += (pred == labels).sum().item()   #.item把tensor 轉為純量
      
    acc = correct / total
    return acc
#%%
net = Net()    #initialize a net
net.to(device) #move to GPU
print(net)     #see summary of net
#%%
#train
criterion = nn.CrossEntropyLoss() #

lr=0.001 #初始學習率
optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=0.001) #weight_decay for L2

epochs = 25

# set the network in "training mode". This affects the behaviour of the dropout 
# layer by "activating" it.
net.train() #set the network to train

#list for plot
train_acc=[]
val_acc=[]
test_acc=[]
loss_list=[]
for epoch in range(epochs):
  
  running_loss = 0 # calculate the average loss of some batches
  for i, data in enumerate(trainset_loader): #get 1 batch each iteration
     
    inputs, labels = data 
    
    # move data and labels to GPU if available
    inputs = inputs.to(device)
    labels = labels.to(device)

    optimizer.zero_grad() #先輕零再計算gradient都要

    outputs=net.forward(inputs)
    loss=criterion(outputs,labels) #一個batch 所有的loss
    loss.backward()

    optimizer.step() #更新學習率
    
    running_loss += loss.item() #.item(): tensor to scalar
    loss_list.append(loss.item())
    #----------statistic----------
    #iterator for statistic (100 data)
    net.eval()
    train_iter=iter(trainset_loader)
    val_iter=iter(valset_loader)
    test_iter=iter(testset_loader)
    train_acc.append(compute_acc(net,train_iter))
    val_acc.append(compute_acc(net,val_iter))
    test_acc.append(compute_acc(net,test_iter))
    net.train()
    #---------statistic-----------
    if i % 50 == 49:
      # every 100 minibatches, print some information in the cell output.
      print("[epoch {}, iter {}] loss: {:.3f}".format(epoch+1, i+1, running_loss/50))
      print("val acc=",val_acc[-1])
      running_loss = 0

print("Finished training")
print("Avg of last 100 val:",sum(val_acc[-100:])/100)
net.eval()
#%% Accuracy

plt.xlim(0,2200 ) # 設定 x 軸座標範圍
plt.ylim(0, 1) # 設定 y 軸座標範圍

plt.xlabel('iteration', fontsize="10") # 設定 x 軸標題內容及大小
plt.ylabel('Accuracy', fontsize="10") # 設定 y 軸標題內容及大小
plt.title('Training Accuracy', fontsize="18") # 設定圖表標題內容及大小


plt.plot(range(len(train_acc)),train_acc, color='blue',label="Training Accuracy")
plt.plot(range(len(val_acc)),val_acc, color='red',label="Validation Accuracy")
plt.plot(range(len(test_acc)),test_acc, color='green',label="Test Accuracy")

plt.legend()
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.savefig("Training Accuracy.png",dpi=100)

plt.clf()
#%% Learning Curve
plt.xlim(0, 2200) # 設定 x 軸座標範圍
plt.ylim(0,2) # 設定 y 軸座標範圍

plt.xlabel('iteration', fontsize="10") # 設定 x 軸標題內容及大小
plt.ylabel('Loss', fontsize="10") # 設定 y 軸標題內容及大小
plt.title('Learning curve', fontsize="18") # 設定圖表標題內容及大小
plt.plot(range(len(loss_list)),loss_list, color='blue',label="cross entropy")
plt.legend()
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.savefig("Learning curve.png",dpi=100)
plt.clf()

#%%
total_para=np.array([])
with torch.no_grad():
    for para in net.conv1.parameters():
        p=para.cpu()
        total_para=np.concatenate((total_para,p.flatten()), axis=0)
        print(para.size())
plt.ylabel('number', fontsize="10") # 設定 y 軸標題內容及大小
plt.xlabel('value', fontsize="10")
plt.title('parameters of conv1', fontsize="18")
plt.hist(total_para,bins=100)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.savefig("parameters of conv1.png",dpi=100)
plt.clf()
#%%
total_para=np.array([])
with torch.no_grad():
    for para in net.conv2.parameters():
        p=para.cpu()
        total_para=np.concatenate((total_para,p.flatten()), axis=0)
        print(para.size())
plt.ylabel('number', fontsize="10") # 設定 y 軸標題內容及大小
plt.xlabel('value', fontsize="10")
plt.title('parameters of conv2', fontsize="18")
plt.hist(total_para,bins=100)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.savefig("parameters of conv2.png",dpi=100)
plt.clf()
#%%
total_para=np.array([])
with torch.no_grad():
    for para in net.conv3.parameters():
        p=para.cpu()
        total_para=np.concatenate((total_para,p.flatten()), axis=0)
        print(para.size())
plt.ylabel('number', fontsize="10") # 設定 y 軸標題內容及大小
plt.xlabel('value', fontsize="10")
plt.title('parameters of conv3', fontsize="18")
plt.hist(total_para,bins=100)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.savefig("parameters of conv3.png",dpi=100)
plt.clf()
#%%
total_para=np.array([])
with torch.no_grad():
    for para in net.conv4.parameters():
        p=para.cpu()
        total_para=np.concatenate((total_para,p.flatten()), axis=0)
        print(para.size())
plt.ylabel('number', fontsize="10") # 設定 y 軸標題內容及大小
plt.xlabel('value', fontsize="10")
plt.title('parameters of conv4', fontsize="18")
plt.hist(total_para,bins=100)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.savefig("parameters of conv4.png",dpi=100)
plt.clf()
#%%
total_para=np.array([])
with torch.no_grad():
    for para in net.fc1.parameters():
        p=para.cpu()
        total_para=np.concatenate((total_para,p.flatten()), axis=0)
        print(para.size())
plt.ylabel('number', fontsize="10") # 設定 y 軸標題內容及大小
plt.xlabel('value', fontsize="10")
plt.title('parameters of fc1(dense)', fontsize="18")
plt.hist(total_para,bins=100)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.savefig("parameters of fc1.png",dpi=100)
plt.clf()
#%%
total_para=np.array([])
with torch.no_grad():
    for para in net.fc2.parameters():
        p=para.cpu()
        total_para=np.concatenate((total_para,p.flatten()), axis=0)
        print(para.size())
plt.ylabel('number', fontsize="10") # 設定 y 軸標題內容及大小
plt.xlabel('value', fontsize="10")
plt.title('parameters of fc2(output logits)', fontsize="18")
plt.hist(total_para,bins=100)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.savefig("parameters of fc2.png",dpi=100)
plt.clf()
#%%
#find the wrong label
data_iter = iter(testset_loader)
with torch.no_grad():
    inputs, labels = data_iter.next()
    
    # move data and labels to GPU if available
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    logits = net.forward(inputs)
    _, pred = torch.max(logits.data, 1)   #後面的如果是0回傳整column中最大的,1的話回傳整row最大的
                                          #第一個回傳最大值 第二個回傳該index
    wrong_index=[]
    inputs=inputs.cpu()
    for i,label in enumerate(labels):
        if pred[i]!=label:
            wrong_index.append(i)
    for i in wrong_index:
        plt.imshow(inputs[i][0],cmap="gray")       
        plt.title("True:"+str(labels[i].item())+"\n"+"Predict:"+str(pred[i].item()))
#%%
#plot of sample
#
wrong_index[0]+=1 #debug should delete
#
store_img=inputs[wrong_index[0]]
store_label=labels[wrong_index[0]].cpu().item()
store_label=classes[store_label]
store_pred=pred[wrong_index[0]].cpu().item()
store_pred=classes[store_pred]
plt.imshow(np.transpose(store_img,(1,2,0)))
plt.title("True:"+str(store_label)+"\n"+"Predict:"+str(store_pred),fontsize=20)

fig = plt.gcf()
fig.set_size_inches(8, 8)
plt.savefig("sample.png",dpi=100)
plt.clf()
#%%
#grid plot of conv1 parameter
weights = net.conv1.weight.data.to("cpu")
img = torchvision.utils.make_grid(weights, normalize=True, scale_each=True)
img=np.transpose(img,(1,2,0))
plt.imshow(img)
#%%
cur_x=store_img.to(device)
#%%
#grid plot of conv1 given input
cur_x=net.conv1(cur_x)

tmp=cur_x.clone()
tmp = tmp.to("cpu")
tmp=tmp.reshape(32,1,32,32)
img = torchvision.utils.make_grid(tmp)
img=np.transpose(img,(1,2,0))
img_n=img.numpy()
plt.imshow(img)

fig = plt.gcf()
fig.set_size_inches(16, 8)
plt.savefig("after conv1.png",dpi=100)
plt.clf()
#%%
#grid plot of conv2 given input
cur_x=net.conv2(F.relu(cur_x))

tmp=cur_x.clone()
tmp = tmp.to("cpu")
tmp=tmp.reshape(32,1,32,32)
img = torchvision.utils.make_grid(tmp)
img=np.transpose(img,(1,2,0))
img_n=img.numpy()
plt.imshow(img)

fig = plt.gcf()
fig.set_size_inches(16, 8)
plt.savefig("after conv2.png",dpi=100)
plt.clf()
#%%
#grid plot of pool1 given input
cur_x=net.pool1(F.relu(cur_x))

tmp=cur_x.clone()
tmp = tmp.to("cpu")
tmp=tmp.reshape(32,1,16,16)
img = torchvision.utils.make_grid(tmp)
img=np.transpose(img,(1,2,0))
img_n=img.numpy()
plt.imshow(img)

fig = plt.gcf()
fig.set_size_inches(16, 8)
plt.savefig("after pool1.png",dpi=100)
plt.clf()
#%%
#grid plot of conv3
cur_x=net.conv3(F.relu(cur_x))

tmp=cur_x.clone()
tmp = tmp.to("cpu")
tmp=tmp.reshape(64,1,16,16)
img = torchvision.utils.make_grid(tmp)
img=np.transpose(img,(1,2,0))
img_n=img.numpy()
plt.imshow(img)

fig = plt.gcf()
fig.set_size_inches(10, 10)
plt.savefig("after conv3.png",dpi=100)
plt.clf()
#%%
#grid plot of conv4
cur_x=net.conv4(F.relu(cur_x))

tmp=cur_x.clone()
tmp = tmp.to("cpu")
tmp=tmp.reshape(64,1,16,16)
img = torchvision.utils.make_grid(tmp)
img=np.transpose(img,(1,2,0))
img_n=img.numpy()
plt.imshow(img)

fig = plt.gcf()
fig.set_size_inches(10, 10)
plt.savefig("after conv4.png",dpi=100)
plt.clf()
#%%
#grid plot of pool2
cur_x=net.pool2(F.relu(cur_x))

tmp=cur_x.clone()
tmp = tmp.to("cpu")
tmp=tmp.reshape(64,1,8,8)
img = torchvision.utils.make_grid(tmp)
img=np.transpose(img,(1,2,0))
img_n=img.numpy()
plt.imshow(img)

fig = plt.gcf()
fig.set_size_inches(10, 10)
plt.savefig("after pool2.png",dpi=100)
plt.clf()
    