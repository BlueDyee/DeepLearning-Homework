# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 22:16:36 2022

@author: user
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Network(object):
   def __init__(self, biases,weights):
       self.biases = biases  # list of ndarray
       self.weights = weights # list of ndarray
       self.num_layers=len(biases)+1
   def backprop(self, x, y):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        #x=encoding(x)
        x.shape=(-1,1)
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for i in range(self.num_layers-1):
            w=self.weights[i]
            b=self.biases[i]
            z = np.dot(w, activation)+b
            #print("z",z)
            zs.append(z)
            if i==(self.num_layers-2):  
                activation=z         #last layer linear
                activations.append(activation)
                continue
            activation = relu(z)
            #print("act:",activation)
            activations.append(activation)
        # backward pass
        #print(activations)
        #print("ERR=",activations[-1]-y)
        #print("Z=",zs)
        delta = self.cost_derivative(activations[-1], y) #last layer don't use activation func
        nabla_b[-1] = delta

        tmp_T=activations[-2].reshape(1,len(activations[-2]))
        #activations[-2]=activations[-2].reshape(1,len(activations[-2]))       
        nabla_w[-1] = np.dot(delta, tmp_T)
        #print(nabla_w[-1])
        #print(delta)
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = relu_prime(z)

            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
        
            tmp_T=activations[-l-1].reshape(1,len(activations[-l-1]))
            delta.shape=(len(delta),1)
            nabla_w[-l] = np.dot(delta, tmp_T)

        return (nabla_b, nabla_w)

   def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
   def forward(self,x):
       activation=x
       x.shape=(-1,1)
       for i in range(self.num_layers-1):
            w=self.weights[i]
            b=self.biases[i]
            z = np.dot(w, activation)+b
            if i==(self.num_layers-2):  
                activation=z         #last layer linear
                continue
            activation = relu(z)
       return activation
   def SGD(self,batch_x,batch_y):
       lr=0.0001
       
       N,D=batch_x.shape
       nabla_b = [np.zeros(b.shape) for b in self.biases]
       nabla_w = [np.zeros(w.shape) for w in self.weights]
       for i in range(N):
           tmp_b,tmp_w=self.backprop(batch_x[i], batch_y[i])
           for j in range(len(tmp_b)):
               nabla_b[j]+=tmp_b[j]
               nabla_w[j]+=tmp_w[j]   
       for j in range(len(tmp_b)):
               self.weights[j] -=lr*nabla_w[j]/N
               self.biases[j]  -=lr*nabla_b[j]/N  
 
       
   def RMS(self,test_X,test_Y,mean,std):
        N,D=test_X.shape
        count=0.
        for i in range(N):
            predict=self.forward(test_X[i])
            count+=((predict-test_Y[i])*std+mean)**2
        return np.sqrt(count/N)

"""
def encoding(x):
    for i in range(1,4):
        x[i]/=500
    x[4]/=7
    x[5]/=5
    return x
"""
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
def relu(z):
    tmp_list=[]
    for e in z:
        tmp_list.append(max(0,e)+min(0,0.1*e))
    tmp=np.array(tmp_list)
    tmp.shape=(-1,1)
    return tmp
def relu_prime(z):
    tmp_list=[]
    for e in z:
        if e>0:
            tmp_list.append(1)
        else:
            tmp_list.append(0.1)
    tmp=np.array(tmp_list)
    tmp.shape=(-1,1)
    return tmp

#
#

data=pd.read_csv("D:\上課\深度學習\DL_HW1\DL_HW1\energy_efficiency_data.csv")
target_data=data.iloc[:,-1]
input_data=data.iloc[:,:-2]

target_data=target_data.sample(frac=1).reset_index(drop=True)
input_data=input_data.sample(frac=1).reset_index(drop=True)
#print(input_data)
#print(data)
#print(target)


width=[8,15,10,10,1] #first index input layer , last index output layer


B=[]
W=[]
#print(type([initial]*5))
for i in range(len(width)-1):
   
    B.append(np.random.random((width[i+1],1)))
    W.append(np.random.random((width[i+1],width[i])))
    #print(W[i].shape)
    #print(B[i].shape)
#print(B)
#print(W)
network=Network(B,W)
X=input_data.to_numpy()
Y=target_data.to_numpy()
t=X[0]
#print("t:",t.shape)
#print(np.dot(W[1],X[0]))
#new_B,new_W=network.backprop(X[0],Y[0])
input_mean=input_data.mean()
input_std=input_data.std()

target_mean=target_data.mean()
target_std=target_data.std()

normalized_input=(input_data-input_mean)/input_std
normalized_target=(target_data-target_mean)/target_std

X=normalized_input.to_numpy()
Y=normalized_target.to_numpy()

N,D=X.shape
"""
batch_size=50
mini_batches = [
    X[k:k+batch_size]
    for k in range(0, N, batch_size)]
for bat in mini_batches:
    print(bat.shape)
"""
train_size=500
epoch=1000
train_RMS=[]
test_RMS=[]
min_RMS=100
for t in range(epoch):
     train_X=X[:train_size]
     train_Y=Y[:train_size]
     network.SGD(train_X,train_Y)
     if t<20:
         continue
     RMS=network.RMS(train_X, train_Y,target_mean,target_std).flatten()
     print("train",t,":",RMS)
     train_RMS.append(RMS)
     test_X=X[train_size:]
     test_Y=Y[train_size:]
     RMS=network.RMS(test_X, test_Y,target_mean,target_std).flatten()
     test_RMS.append(RMS)
     print("test",t,":",RMS)
     if RMS<min_RMS :
         min_RMS=RMS
"""
for t in range(epoch):
    np.random.shuffle(X)
    mini_batches = [
    X[k:k+batch_size]
    for k in range(0, N, batch_size)]
    
    for i in range(train_size):
        for k in range(5):
            network.backprop(X[i],Y[i])
"""
"""
test_X=X[train_size:,:]
test_Y=Y[train_size:]
print("------------")

print("RMS",network.RMS(test_X, test_Y,target_mean,target_std))
"""
plt.ylabel("MSE") # y label
plt.xlabel("epochs") # x label
plt.plot(range(20,epoch),train_RMS, color=(255/255,100/255,100/255),label = 'training set')
plt.plot(range(20,epoch),test_RMS, '--', color=(100/255,100/255,255/255),label = 'testing set')

plt.title("Erms")
plt.legend()
plt.show()

print("min",min_RMS)
#W-=new_W*0.01
#B-=new_B*0.01
#print(new_B)
#print(new_W)
