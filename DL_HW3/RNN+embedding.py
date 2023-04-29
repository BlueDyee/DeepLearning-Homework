# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 20:30:30 2022

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 18:27:31 2022

@author: user
"""
#%%
#preprocessing
import numpy as np
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
max_length=100
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
class RNN_text_dataset(torch.utils.data.Dataset):
    def __init__(self,sentences,seq_len,dict_size,transform=None):
        self.transform = transform
        N=len(sentences)     
        input_sentences=[]
        target_sentences=[]
        for i in range(len(sentences)):
            input_sentences.append(sentences[i][:-1])
            target_sentences.append(sentences[i][1:])
            
        data=np.array(input_sentences)
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
    def __init__(self, input_size, output_size,embed_dim, hidden_dim, n_layers):
        super(RNN, self).__init__()
        self.embed_dim=embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        #
        self.embedding=nn.Embedding(input_size,embed_dim)
        # RNN Layer
        self.rnn = nn.RNN(embed_dim, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    def forward(self, x,hidden): #x is batch of sentences
         
        embed=self.embedding(x)
        # Passing in the input and hidden state into the model
        out1, hidden = self.rnn(embed,hidden)
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
RNN=RNN(input_size=dict_size,output_size=dict_size,embed_dim=256,hidden_dim=128,n_layers=1)
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
RNN=RNN.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(RNN.parameters(), lr=lr)
epochs=25
torch.autograd.set_detect_anomaly(True)
train_loss_list=[]
val_loss_list=[]
for epoch in range(1, epochs + 1):

    hidden=RNN.init_hidden(BATCH_SIZE).to(device)
    for i,data in enumerate(train_dataloader):
        seqs,labels=data
        seqs=seqs.to(device)
        labels=labels.to(device)
        optimizer.zero_grad() # Clears existing gradients from previous epoch
        
        output,h = RNN(seqs,hidden)
        h=h.detach() #detach is required or it would record history gradient
        #BPC loss
        loss = criterion(output, labels.view(-1).long())# long is needed int32 to int64
        train_loss_list.append(loss)
        loss.backward() # retain_graph=True ?
        optimizer.step() # Updates the weights accordingly
        hidden=h
    with torch.no_grad():
        hidden=RNN.init_hidden(BATCH_SIZE).to(device)
        for i,data in enumerate(valid_dataloader):
            seqs,labels=data
            seqs=seqs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            #BPC loss
            output,h = RNN(seqs,hidden)
            h=h.detach() #detach is required or it would record history gradient
            loss = criterion(output, labels.view(-1).long()) # long is needed int32 to int64
            val_loss_list.append(loss)
            hidden=h
        if epoch%1 == 0:
            print('Epoch: {}/{}.............'.format(epoch,epochs), end=' ')
            print("Train Loss: {:.4f}".format(train_loss_list[-1].item()))
            print("Validation Loss: {:.4f}".format(val_loss_list[-1].item()))
            print("sample:",)
        if epoch%4== 1:
            print("sample at epoch=",epoch)
            test_data=next(iter(train_dataloader))
            test_in,x=test_data
            hidden=RNN.init_hidden(1).to(device)
            test_in=test_in.to(device)
            output,h=RNN(test_in[0:1],hidden)
            output=torch.argmax(output,axis=1)
            input_sentence=test_in[0]
            print(' '.join([int_to_vocab[c.item()] for c in input_sentence]))
            print("prediction:")
            print(' '.join([int_to_vocab[c.item()] for c in output]))
#%%
input_sentence=test_in[0]
print(' '.join([int_to_vocab[c.item()] for c in input_sentence]))
print("prediction:")
print(' '.join([int_to_vocab[c.item()] for c in output]))