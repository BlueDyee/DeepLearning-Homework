# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 18:27:31 2022

@author: user
"""
#%%
#preprocessing
import numpy as np
import matplotlib.pyplot as plt
import io
import os

abs_path=os.path.dirname(os.path.abspath(__file__))
abs_path=abs_path.replace('\\','/')
abs_path+='/'
dataURL='shakespeare_train.txt'
with io.open(abs_path+dataURL,'r',encoding='utf8')as f:
    text=f.read()
#Characters’collection
vocab=set(text)
#Construct character dictionary
vocab_to_int={c:i for i,c in enumerate(vocab)}
int_to_vocab=dict(enumerate(vocab))
#Encodedata,shape=[number of characters]
traindata=np.array([vocab_to_int[c] for c in text],dtype=np.int32)
dict_size=len(vocab_to_int)
#%%
valURL='shakespeare_valid.txt'
with io.open(abs_path+valURL,'r',encoding='utf8')as f:
    text=f.read()
valdata=np.array([vocab_to_int[c] for c in text],dtype=np.int32)
#%%
#check
print("int",traindata[0:62])
print("char",list(int_to_vocab[c] for c in traindata[0:62]))
#%%
sentences=[]
i,j=0,1
max_length=20
seq_len=max_length-1
for i in range(len(traindata)):
   if i ==j*max_length:
       sentences.append(traindata[i-max_length:i])
       j+=1
sentences_val=[]
i,j=0,1

for i in range(len(text)):
   if i ==j*max_length:
       sentences_val.append(valdata[i-max_length:i])
       j+=1

#%%
import torch
import torch.nn as nn    
import torch.nn.functional as F
from torch.autograd import Variable

#%%
def one_hot_encode(sequence, data_size, seq_len, dict_size):
    features = np.zeros((data_size, seq_len, dict_size), dtype=np.float32)
    
    for i in range(data_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features
class RNN_text_dataset(torch.utils.data.Dataset):
    def __init__(self,sentences,seq_len,dict_size,transform=None):
        self.transform = transform
        N=len(sentences)
        
        input_sentences=[]
        target_sentences=[]
        for i in range(len(sentences)):
            input_sentences.append(sentences[i][:-1])
            target_sentences.append(sentences[i][1:])
            
        data=one_hot_encode(input_sentences,N,seq_len,dict_size)
        labels=np.array(target_sentences)
    
        self.data=torch.tensor(data)
        self.labels=torch.tensor(labels)
        self.len=N
    def __getitem__(self,index):
        if self.transform:
            return self.transform(self.data[index]),int(self.labels[index])
        return self.data[index],self.labels[index]
    def __len__(self):
        return self.len
    
seq_len=max_length-1
dict_size=len(vocab_to_int)
train_dataset=RNN_text_dataset(sentences,seq_len,dict_size)
valid_dataset=RNN_text_dataset(sentences_val,seq_len,dict_size)
#%%
t=train_dataset[5]

#%%
#version2
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    def forward(self, x,hidden): #x is batch of sentences

        # Passing in the input and hidden state into the model
        out1, hidden = self.rnn(x,hidden)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        #ex:origin (2,99,128)=>((198,128))
        out2 = out1.contiguous().view(-1, self.hidden_dim)
        #out = out.view(-1, self.hidden_dim)
        out = self.fc(out2)
        return out, hidden
    
    def init_hidden(self, batch_size):
        # Hidden to zero at each epoch
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
RNN_=RNN(input_size=dict_size,output_size=dict_size,hidden_dim=128,n_layers=1)
#h_dim=256 n_layers=4 loss=3~4
#256 n_layers=1 loss=1.8
#%%
"""
test_in,test_lb=train_dataset[0:2]
RNN.init_hidden(2)
o,h=RNN.forward(test_in)

o,h=RNN.rnn(test_in)

o = o.contiguous().view(-1,128)
"""
#%%
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
#%%
BATCH_SIZE = 32
from torch.utils.data import DataLoader
train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,drop_last=True)
valid_dataloader=DataLoader(valid_dataset,batch_size=BATCH_SIZE,drop_last=True)
#%%
"""
test_data=next(iter(train_dataloader))
test_in,x=test_data
hidden=RNN.init_hidden(1).to(device)
test_in=test_in.to(device)
output,h=RNN(test_in[0:1],hidden)
output=torch.argmax(output,axis=1)

#%%
input_sentence=torch.argmax(test_in[0],axis=1)
print(' '.join([int_to_vocab[c.item()] for c in input_sentence]))
print("prediction")
print(' '.join([int_to_vocab[c.item()] for c in output]))
"""
#%%
#train
#version2
lr=0.003
RNN_=RNN_.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(RNN_.parameters(), lr=lr)
epochs=20
#torch.autograd.set_detect_anomaly(True)
train_acc_list=[]
val_acc_list=[]
train_loss_list=[]
val_loss_list=[]
for epoch in range(1, epochs + 1):

    hidden=RNN_.init_hidden(BATCH_SIZE).to(device)

    for i,data in enumerate(train_dataloader):
        seqs,labels=data
        seqs=seqs.to(device)
        labels=labels.to(device)
        optimizer.zero_grad() # Clears existing gradients from previous epoch
        
        output,h = RNN_(seqs,hidden)
        h=h.detach() #detach is required or it would record history gradient
        #ACC
        correctness=(torch.argmax(output,axis=1)==labels.view(-1).long())
        acc=correctness.sum()/(BATCH_SIZE*seq_len)
        train_acc_list.append(acc.item())
        #loss
        loss = criterion(output, labels.view(-1).long())# long is needed int32 to int64
        loss.backward() # retain_graph=True ?
        optimizer.step() # Updates the weights accordingly
        hidden=h
        train_loss_list.append(loss.item())
        #val
        with torch.no_grad():
            hidden_val=RNN_.init_hidden(BATCH_SIZE).to(device)
            data=next(iter(valid_dataloader))
            seqs,labels=data
            seqs=seqs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            
            output,h = RNN_(seqs,hidden_val)
            #ACC
            correctness=(torch.argmax(output,axis=1)==labels.view(-1).long())
            acc=correctness.sum()/(BATCH_SIZE*seq_len)
            val_acc_list.append(acc.item())
            #loss
            loss = criterion(output, labels.view(-1).long())# long is needed int32 to int64

            val_loss_list.append(loss.item())

    with torch.no_grad():
        if epoch%1 == 0:
            print('Epoch: {}/{}.............'.format(epoch,epochs))
            print("Train acc: {:.4f}".format(train_acc_list[-1]))
            print("Validation acc: {:.4f}".format(val_acc_list[-1]))
            print("Train Loss: {:.4f}".format(train_loss_list[-1]))
            print("Validation Loss: {:.4f}".format(val_loss_list[-1]))
        if epoch%4== 1:
            print("sample at epoch=",epoch)
            test_data=next(iter(train_dataloader))
            test_in,x=test_data
            hidden=RNN_.init_hidden(1).to(device)
            test_in=test_in.to(device)
            output,h=RNN_(test_in[0:1],hidden)
            output=torch.argmax(output,axis=1)
            input_sentence=torch.argmax(test_in[0],axis=1)
            print(' '.join([int_to_vocab[c.item()] for c in input_sentence]))
            print("prediction:")
            print(' '.join([int_to_vocab[c.item()] for c in output]))
#%% Accuracy
import matplotlib.pyplot as plt

#plt.xlim(0,2200 ) # 設定 x 軸座標範圍
plt.ylim(0, 1) # 設定 y 軸座標範圍

plt.xlabel('iteration(avg of every 20 iterations)', fontsize="10") # 設定 x 軸標題內容及大小
plt.ylabel('Accuracy', fontsize="10") # 設定 y 軸標題內容及大小
plt.title('Training Accuracy', fontsize="18") # 設定圖表標題內容及大小

avg_train_acc_list=[]
avg_val_acc_list=[]
train_sum=0
val_sum=0
#average iteration ex 1000->50
INTERVAL=20
for i in range(len(val_acc_list)):
    train_sum+=train_acc_list[i]
    val_sum+=val_acc_list[i]
    if i%INTERVAL==INTERVAL-1:
        avg_train_acc_list.append(train_sum/INTERVAL)
        avg_val_acc_list.append(val_sum/INTERVAL)
        train_sum=0
        val_sum=0

plt.plot(range(len(avg_train_acc_list)),avg_train_acc_list, color='blue',label="Training Accuracy")
plt.plot(range(len(avg_val_acc_list)),avg_val_acc_list, color='red',label="Validation Accuracy")

plt.legend()
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.savefig("RNN Training Accuracy.png",dpi=100)

plt.clf()
#%% Learning Curve
#plt.xlim(0, 2200) # 設定 x 軸座標範圍
#plt.ylim(0,0.002) # 設定 y 軸座標範圍

avg_train_loss_list=[]
avg_val_loss_list=[]
train_sum=0
val_sum=0
INTERVAL=20
for i in range(len(val_loss_list)):
    train_sum+=train_loss_list[i]
    val_sum+=val_loss_list[i]
    if i%INTERVAL==INTERVAL-1:
        avg_train_loss_list.append(train_sum/INTERVAL)
        avg_val_loss_list.append(val_sum/INTERVAL)
        train_sum=0
        val_sum=0
plt.xlabel('iteration(avg of every 20 iterations)', fontsize="10") # 設定 x 軸標題內容及大小
plt.ylabel('Loss', fontsize="10") # 設定 y 軸標題內容及大小
plt.title('Learning curve', fontsize="18") # 設定圖表標題內容及大小
plt.plot(range(len(avg_train_loss_list)),avg_train_loss_list, color='blue',label="BPC train")
plt.plot(range(len(avg_val_loss_list)),avg_val_loss_list, color='red',label="BPC val")
plt.legend()
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.savefig("RNN Learning curve.png",dpi=100)
plt.clf()
#%%
#Changing parameter of RNN
def RNN_eval(hiddens,max_length,lr=0.001,BATCH_SIZE=32):
    
    sentences=[]
    i,j=0,1
    seq_len=max_length-1
    for i in range(len(traindata)):
       if i ==j*max_length:
           sentences.append(traindata[i-max_length:i])
           j+=1
    #
    seq_len=max_length-1
    dict_size=len(vocab_to_int)
    train_dataset=RNN_text_dataset(sentences,seq_len,dict_size)
    
    from torch.utils.data import DataLoader
    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,drop_last=True)
    valid_dataloader=DataLoader(valid_dataset,batch_size=BATCH_SIZE,drop_last=True)
    #
    RNN_=RNN(input_size=dict_size,output_size=dict_size,hidden_dim=hiddens,n_layers=1)
    RNN_=RNN_.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(RNN_.parameters(), lr=lr)
    epochs=10
    train_acc_list=[]
    train_loss_list=[]

    for epoch in range(1, epochs + 1):

        hidden=RNN_.init_hidden(BATCH_SIZE).to(device)
        for i,data in enumerate(train_dataloader):
            seqs,labels=data
            seqs=seqs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            
            output,h = RNN_(seqs,hidden)
            h=h.detach() #detach is required or it would record history gradient
            #ACC
            correctness=(torch.argmax(output,axis=1)==labels.view(-1).long())
            acc=correctness.sum()/(BATCH_SIZE*seq_len)
            train_acc_list.append(acc.item())
            #loss
            loss = criterion(output, labels.view(-1).long())# long is needed int32 to int64
            loss.backward() # retain_graph=True ?
            optimizer.step() # Updates the weights accordingly
            hidden=h
            train_loss_list.append(loss.item())
        print(loss)
    return sum(train_loss_list[-100:])/100,sum(train_acc_list[-100:])/100

#%%
loss_avg_list=[]
acc_avg_list=[]
for h in [8,16,32,64,128,256,512,1024]:
    loss_avg,acc_avg=RNN_eval(hiddens=h, max_length=100)
    loss_avg_list.append(loss_avg)
    acc_avg_list.append(acc_avg)
#%%
#plt.xlim(0,2200 ) # 設定 x 軸座標範圍
plt.ylim(0, 1) # 設定 y 軸座標範圍

plt.xlabel('parameter(hidden state)', fontsize="10") # 設定 x 軸標題內容及大小
plt.ylabel('Accuracy(avg of last 100 iter in 10 epochs )', fontsize="10") # 設定 y 軸標題內容及大小
plt.title('Accuracy in different hidden state(seq_len=100-1,B_size=32,lr=0.001)', fontsize="18") # 設定圖表標題內容及大小

plt.plot([8,16,32,64,128,256,512,1024],acc_avg_list,marker='o', color='blue',label="Training Accuracy")
for x,y in zip([8,16,32,64,128,256,512,1024],acc_avg_list): 
    plt.text(x, y+0.01, "({}, {:.3f})".format(x,y))

plt.legend()
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.savefig("RNN Accuracy in different hiddens.png",dpi=100)

plt.clf()
#%%
#plt.xlim(0, 2200) # 設定 x 軸座標範圍
#plt.ylim(0,0.002) # 設定 y 軸座標範圍

plt.xlabel('parameter(hidden state)', fontsize="10") # 設定 x 軸標題內容及大小
plt.ylabel('BPC(avg of last 100 iter in 10 epochs )', fontsize="10") # 設定 y 軸標題內容及大小
plt.title('loss in different hidden state(seq_len=100-1,B_size=32,lr=0.001)', fontsize="18") # 設定圖表標題內容及大小

plt.plot([8,16,32,64,128,256,512,1024],loss_avg_list,marker='o', color='blue',label="BPC train")
for x,y in zip([8,16,32,64,128,256,512,1024],loss_avg_list): 
    plt.text(x, y+0.05, "({}, {:.5f})".format(x,y))


plt.legend()
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.savefig("RNN Learning curve in different hiddens.png",dpi=100)
plt.clf()
#%%
loss_avg_list=[]
acc_avg_list=[]
for s in [25,50,100,150,200,500]:
    loss_avg,acc_avg=RNN_eval(hiddens=128, max_length=s)
    loss_avg_list.append(loss_avg)
    acc_avg_list.append(acc_avg)
#%%
#plt.xlim(0,2200 ) # 設定 x 軸座標範圍
plt.ylim(0, 1) # 設定 y 軸座標範圍

plt.xlabel('parameter(seqence length)', fontsize="10") # 設定 x 軸標題內容及大小
plt.ylabel('Accuracy(avg of last 100 iter in 10 epochs )', fontsize="10") # 設定 y 軸標題內容及大小
plt.title('Accuracy in different seq_len(hiddens=128,B_size=32,lr=0.001)', fontsize="18") # 設定圖表標題內容及大小

plt.plot([25,50,100,150,200,500],acc_avg_list,marker='o', color='blue',label="Training Accuracy")
for x,y in zip([25,50,100,150,200,500],acc_avg_list): 
    plt.text(x, y+0.01, "({}, {:.3f})".format(x,y))

plt.legend()
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.savefig("RNN Accuracy in different seq_len.png",dpi=100)

plt.clf()
#%%
#plt.xlim(0, 2200) # 設定 x 軸座標範圍
plt.ylim(0,3) # 設定 y 軸座標範圍

plt.xlabel('parameter(seqence length)', fontsize="10") # 設定 x 軸標題內容及大小
plt.ylabel('BPC(avg of last 100 iter in 10 epochs )', fontsize="10") # 設定 y 軸標題內容及大小
plt.title('loss in different seq_len(hiddens=128,B_size=32,lr=0.001)', fontsize="18") # 設定圖表標題內容及大小

plt.plot([25,50,100,150,200,500],loss_avg_list,marker='o', color='blue',label="BPC train")
for x,y in zip([25,50,100,150,200,500],loss_avg_list): 
    plt.text(x, y+0.05, "({}, {:.5f})".format(x,y))


plt.legend()
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.savefig("RNN Learning curve in different seq_len.png",dpi=100)
plt.clf()
#%%
#Priming
def preprocess(test_sentence):
    test_tokens=[]

    test_tokens=[vocab_to_int[c] for c in test_sentence]
    test_tokens=np.array(test_tokens)
    
    features = np.zeros((1, len(test_sentence), dict_size), dtype=np.float32)
    
    for u in range(len(test_sentence)):
        features[0, u, test_tokens[u]] = 1
    
    return features
test_sentence="JULIET"
test_in=preprocess(test_sentence)
#%%
"""
#single tiral
test_sentence="JULIET"

test_in=preprocess(test_sentence)
test_in=torch.tensor(test_in)
test_in=test_in.to(device)

h=RNN_.init_hidden(1)
h=h.to(device)

output,h=RNN_.forward(test_in,h)
output=torch.argmax(output,axis=1)
print("input:JULIET")
print("prediction:")
print(int_to_vocab[output[-1].item()])
"""
#%%
#
m = nn.Softmax(dim=0)
cur_sentence="JULIET"
temperature=0.9
h=RNN_.init_hidden(1)
h=h.to(device)
with torch.no_grad():
    #for pack in range(5)
    h=RNN_.init_hidden(1)
    h=h.to(device)
    while len(cur_sentence)<500:

        
        test_in=preprocess(cur_sentence)
        test_in=torch.tensor(test_in)
        test_in=test_in.to(device)
        #chang the h with new h perform better than all initial
        logits,h=RNN_.forward(test_in,h)  
        choice=torch.multinomial(m(logits[-1]/temperature),1)
        prediction=int_to_vocab[choice.item()]
        cur_sentence+=prediction
print(cur_sentence)
