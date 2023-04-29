# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 20:19:11 2022

@author: user
"""
#--------import------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("D:\上課\深度學習\DL_HW1\DL_HW1\energy_efficiency_data.csv")
#--------------------
#--------network------
class Network(object):
   def __init__(self, biases,weights,lr=0.002):
       self.biases = biases  # list of ndarray
       self.weights = weights # list of ndarray
       self.num_layers=len(biases)+1
       self.lr=lr
   def backprop(self, x, y):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        x.shape=(-1,1) #original stored in row vector change into column vector
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for i in range(self.num_layers-1):
            w=self.weights[i]
            b=self.biases[i]
            z = np.dot(w, activation)+b
            zs.append(z)
            if i==(self.num_layers-2):  #last layer is linear don't activate
                activation=z         
                activations.append(activation)
                continue
            activation = relu(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) #last layer don't use activation func
        nabla_b[-1] = delta   
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = relu_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

            nabla_b[-l] = delta
            delta.shape=(len(delta),1)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

   def cost_derivative(self, output_activations, y):
        return (output_activations-y)
   def forward(self,x):
       x.shape=(-1,1)
       activation=x

       for i in range(self.num_layers-1):
            w=self.weights[i]
            b=self.biases[i]
            z = np.dot(w, activation)+b
            if i==(self.num_layers-2):
                activation=z    #last layer linear
                continue
            activation = relu(z)
       return activation
   def SGD(self,batch_x,batch_y):
       lr=self.lr
  
       N,D=batch_x.shape
       nabla_b = [np.zeros(b.shape) for b in self.biases]
       nabla_w = [np.zeros(w.shape) for w in self.weights]
       for i in range(N):
           tmp_b,tmp_w=self.backprop(batch_x[i], batch_y[i])
           for j in range(len(tmp_b)):
               nabla_b[j]+=tmp_b[j]
               nabla_w[j]+=tmp_w[j]
       #--------updating---------
       for j in range(len(tmp_b)):
               self.weights[j] -=lr*nabla_w[j]/N
               self.biases[j]  -=lr*nabla_b[j]/N  
       
   def RMS(self,test_X,test_Y):
        N,D=test_X.shape
        count=0.
        for i in range(N):
            predict=self.forward(test_X[i])
            count+=(predict-test_Y[i])**2            
        return np.sqrt(count/N)
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
def relu(z,slope=0.1):
    return np.where(z > 0, z, slope*z)
def relu_prime(z,slope=0.1):
    return np.where(z > 0, 1, slope)

def init_layers(width):     #width:list indicate width of each layers from input to output
    B=[]
    W=[]

    for i in range(len(width)-1):
        B.append(np.random.random((width[i+1],1))-0.5) #range from-0.5~0.5
        W.append(np.random.random((width[i+1],width[i]))-0.5)
    return B, W
    
#---------------------
def train(X, Y, network, epochs):
    #-----Hyper parameter----------
    N,D=X.shape
    train_size=int(N*0.75)

    train_X=X[:train_size]
    train_Y=Y[:train_size]
    test_X=X[train_size:]
    test_Y=Y[train_size:]
    
    
    #-----------------------
    #------Record----
    train_RMS=[]
    test_RMS=[]
    min_RMS=100
    start_observe=0
    print_interval=20
    
    #----------------
    for t in range(epochs):
        network.SGD(train_X,train_Y)    #contain updating
        if t<start_observe:
             continue
        RMS_train=network.RMS(train_X, train_Y).flatten()
        RMS_test=network.RMS(test_X, test_Y).flatten()
        train_RMS.append(RMS_train)
        test_RMS.append(RMS_test)
        if t%print_interval==0:
            print("times:",t," train: ",RMS_train," test:",RMS_test)
        if RMS_test<min_RMS :
            min_RMS=RMS_test
    #-------plot--------
    plt.ylabel("E_RMS") # y label
    plt.xlabel("epochs") # x label
    plt.plot(range(start_observe,epochs),train_RMS, color=(255/255,100/255,100/255),label = 'training set')
    plt.plot(range(start_observe,epochs),test_RMS, '--', color=(100/255,100/255,255/255),label = 'testing set')
    
    plt.title("NNNN")
    plt.legend()
    plt.show()
    
    print("Minimum:",min_RMS)
    #-------------------
    return

##################################################################

####---------------DATA PREPROCESSING----------------#####
data=data.sample(frac=1).reset_index(drop=True)   #shuffle data
target_data=data.iloc[:,-1]
input_data=data.iloc[:,:-2]
#-----standard normalization input----
input_mean=input_data.mean()
input_std=input_data.std()

normalized_input=(input_data-input_mean)/input_std
normalized_target=target_data   #don't normalized_target
#-----one hot----Orientation 2 3 4 5
orien_list=[[] for i in range(4)]

for orien in input_data["Orientation"]:
  for category in range(4):
    if (orien-2)==category:
      orien_list[category].append(1)
    else :
      orien_list[category].append(0)
normalized_input=normalized_input.drop("Orientation",axis=1)

for i in range(4):
  normalized_input["orien:"+str(i+2)]=orien_list[i]
  
#---one hot-----Glazing Area Distribution 0~5
distribution_list=[[] for i in range(6)]

for orien in input_data["Glazing Area Distribution"]:
  for category in range(6):
    if (orien)==category:
      distribution_list[category].append(1)
    else :
      distribution_list[category].append(0)
normalized_input=normalized_input.drop("Glazing Area Distribution",axis=1)

for i in range(6):
  normalized_input["distribution:"+str(i)]=distribution_list[i]
  
#####----------------------------------------------#####


#---------start-------
X=normalized_input.to_numpy()
Y=normalized_target.to_numpy()

width=[16,48,24,12,1]
biases, weights=init_layers(width)

weights_copy=[]
biases_copy=[]
for i in range(len(weights)):
    weights_copy.append(weights[i].copy())
    biases_copy.append(biases[i].copy())

network=Network(biases,weights,lr=0.002)
train(X,Y,network,500)

    
#%%
N,D=X.shape
train_size=576
plt.figure(figsize=(10, 7))
plt.ylabel("heating load") # y label
plt.xlabel("#cases") # x label
predict_list=[]
for x in X[train_size:]:
    predict_list.append(network.forward(x).flatten())
plt.plot(range(N-train_size),Y[train_size:].tolist(), color=(255/255,100/255,100/255),label = 'label')
plt.plot(range(N-train_size),predict_list, '--', color=(100/255,100/255,255/255),label = 'predict')
    
plt.title("Prediction for Testing")
plt.legend()

plt.savefig("HW1.png")
#plt.show()
#%%
#--------------train with one feature eliminated------------
features=normalized_input.columns
X_copy=X.copy()
Y_copy=Y.copy()
weights_copy=[]
biases_copy=[]
for i in range(len(weights)):
        weights_copy.append(weights[i].copy())
        biases_copy.append(biases[i].copy())

for i in range(6):
    X_copy[:,i]=0
    network=Network(biases_copy,weights_copy,lr=0.002)
    train(X_copy, Y_copy, network,100)
    print("----------Above is blocked:",features[i],"---------")
    
    X_copy=X.copy()
    weights_copy=[]
    biases_copy=[]
    for i in range(len(weights)):
        weights_copy.append(weights[i].copy())
        biases_copy.append(biases[i].copy())
for i in range(10,16):
    X_copy[:,i]=0
network=Network(biases_copy,weights_copy,lr=0.002)
train(X_copy, Y_copy, network,100)
print("----------Above is blocked:","Glazing area distribution","---------")

X_copy=X.copy()
weights_copy=[]
biases_copy=[]
for i in range(len(weights)):
    weights_copy.append(weights[i].copy())
    biases_copy.append(biases[i].copy())
#%%
#--------------train with single feature------------
width=[1,48,24,12,1]
biases, weights=init_layers(width)

features=normalized_input.columns
X_copy=X.copy()
Y_copy=Y.copy()
weights_copy=[]
biases_copy=[]
for i in range(len(weights)):
        weights_copy.append(weights[i].copy())
        biases_copy.append(biases[i].copy())

for i in range(6):
    X_copy=X[:,i]
    X_copy.shape=(-1,1)
    network=Network(biases_copy,weights_copy,lr=0.002)
    train(X_copy, Y_copy, network,100)
    print("----------Above is choosing only:",features[i],"---------")
    
    weights_copy=[]
    biases_copy=[]
    for i in range(len(weights)):
        weights_copy.append(weights[i].copy())
        biases_copy.append(biases[i].copy())

#%%
width=[6,48,24,12,1]
biases, weights=init_layers(width)

features=normalized_input.columns
X_copy=X.copy()
Y_copy=Y.copy()
weights_copy=[]
biases_copy=[]

for i in range(len(weights)):
        weights_copy.append(weights[i].copy())
        biases_copy.append(biases[i].copy())
        
X_copy=X[:,10:]
network=Network(biases_copy,weights_copy,lr=0.002)
train(X_copy, Y_copy, network,100)
print("----------Above is choosing only:","Glazing area distribution","---------")
