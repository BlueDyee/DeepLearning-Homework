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

# detect if a GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))
#%%
data_folder = "./data" #specify where the data should load
# can add other transform ex:normalize if needed
load_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),\
                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                 ])

trainset= torchvision.datasets.MNIST(root=data_folder, train=True,\
                            download=True,transform=load_transform)
testset = torchvision.datasets.MNIST(root=data_folder, train=False,\
                            download=True, transform=load_transform)
trainset, valset = torch.utils.data.random_split(trainset, [55000, 5000])
"""
root=where we store MINIST data
train:因為MINIST有分train的data跟test的data
    如果=True取Train data 如果=False 取test data
download=True:如果在root的位置沒有找到會重新下載
transform=...ToTensor:原本是60000個PIL image 把他換成1個Tensor
    #transform 當這個dataset被拿去使用ex:dataloader 就會使用這個trainst對應的transform

NOTICE:PIL to Tensor will change the value and order
Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                            
"""

trainset_loader = torch.utils.data.DataLoader(trainset,batch_size=512,shuffle=True)
valset_loader=torch.utils.data.DataLoader(valset,batch_size=512,shuffle=True)
testset_loader = torch.utils.data.DataLoader(testset,batch_size=512,shuffle=True)
#%%

#verify if import correctlly

dataiter = iter(trainset_loader)
imgs,labels=dataiter.next()
img16=imgs[:16]
plt.imshow(img16[0][0].numpy(),cmap='gray')
test=torchvision.utils.make_grid(img16)
plt.imshow(np.transpose(test,(1,2,0)))
test=test.flatten()

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
    self.conv1 = nn.Conv2d(in_channels=1, padding=2, out_channels=32, kernel_size=5) #output size=32*28*28
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #output size=32*14*14
    self.conv2 = nn.Conv2d(in_channels=32, padding=2, out_channels=64, kernel_size=5)#output size=64*14*14
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output size= 64*7*7
    self.fc1 = nn.Linear(in_features=64*7*7, out_features=1024) # 全連接層 提取圖片各個池的特性
    self.fc2 = nn.Linear(in_features=1024, out_features=10) #最後一個全連接層->logits
    self.dropout = nn.Dropout(p=0.5) #Avoid overfitting
    
  def forward(self, x):
    # convolution->ReLU->pooling
    x = self.pool1(F.leaky_relu(self.conv1(x)))
    x = self.pool2(F.leaky_relu(self.conv2(x)))
    
    x = x.view(-1, 64*7*7) #flatten to 1 diemention for full-connected-layer
    
    x = F.leaky_relu(self.fc1(x))# 全連接層
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

lr=0.0015 #初始學習率
optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=0.001) #weight_decay for L2

epochs = 4

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
       if i % 100 == 99:
          # every 100 minibatches, print some information in the cell output.
          print("[epoch {}, iter {}] loss: {:.3f}".format(epoch+1, i+1, running_loss/100))
          running_loss = 0

print("Finished training")
net.eval()
#%% Accuracy

plt.xlim(0, 400) # 設定 x 軸座標範圍
plt.ylim(0.94, 1) # 設定 y 軸座標範圍

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
plt.show()

plt.clf()
#%% Learning Curve
plt.xlim(0, 400) # 設定 x 軸座標範圍
plt.ylim(0,0.25) # 設定 y 軸座標範圍

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
wrong_index[0]+=1 #debug should delete
store_img=inputs[wrong_index[0]]
store_label=labels[wrong_index[0]].cpu().item()
store_pred=pred[wrong_index[0]].cpu().item()
plt.imshow(store_img[0],cmap="gray")       
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
#grid plot of conv1 given input
tmp=store_img.to(device)
tmp=net.conv1(tmp)

tmp = tmp.to("cpu")
tmp=tmp.reshape(32,1,28,28)
img = torchvision.utils.make_grid(tmp)
img=np.transpose(img,(1,2,0))
img_n=img.numpy()
plt.imshow(img)

fig = plt.gcf()
fig.set_size_inches(16, 8)
plt.savefig("after conv1.png",dpi=100)
plt.clf()
#%%
#grid plot of pool1 given input
tmp=store_img.to(device)
tmp=net.pool1(F.leaky_relu(net.conv1(tmp)))

tmp = tmp.to("cpu")
tmp=tmp.reshape(32,1,14,14)
img = torchvision.utils.make_grid(tmp)
img=np.transpose(img,(1,2,0))
img_n=img.numpy()
plt.imshow(img)

fig = plt.gcf()
fig.set_size_inches(16, 8)
plt.savefig("after pool1.png",dpi=100)
plt.clf()
#%%
#grid plot of conv2
tmp=store_img.to(device)
tmp=net.pool1(F.leaky_relu(net.conv1(tmp)))
tmp=net.conv2(tmp)

tmp = tmp.to("cpu")
tmp=tmp.reshape(64,1,14,14)
img = torchvision.utils.make_grid(tmp)
img=np.transpose(img,(1,2,0))
img_n=img.numpy()
plt.imshow(img)

fig = plt.gcf()
fig.set_size_inches(10, 10)
plt.savefig("after conv2.png",dpi=100)
plt.clf()
#%%
#grid plot of pool2
tmp=store_img.to(device)
tmp=net.pool1(F.leaky_relu(net.conv1(tmp)))
tmp=net.pool2(F.leaky_relu(net.conv2(tmp)))

tmp = tmp.to("cpu")
tmp=tmp.reshape(64,1,7,7)
img = torchvision.utils.make_grid(tmp)
img=np.transpose(img,(1,2,0))
img_n=img.numpy()
plt.imshow(img)

fig = plt.gcf()
fig.set_size_inches(10, 10)
plt.savefig("after pool2.png",dpi=100)
plt.clf()
    