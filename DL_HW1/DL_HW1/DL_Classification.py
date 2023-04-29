# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 20:19:11 2022

@author: user
"""
#--------import------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("D:\上課\深度學習\DL_HW1\DL_HW1\ionosphere_data.csv")
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
            if i==(self.num_layers-2):  #last layer is softmax
                activation=softmax(z)         
                activations.append(activation)
                continue
            activation = relu(z)
            activations.append(activation)
        # backward pass
        y.shape=(-1,1) #original stored in row vector change into column vector
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
                activation=softmax(z)    #last layer linear
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
   def Error(self,test_X,test_Y):
       N,D=test_X.shape
       wrong=0;
       correct=0;
       for x,y in zip(test_X,test_Y):
           prediction=np.argmax(self.forward(x))
           answer=np.argmax(y)
           if prediction== answer:
               correct+=1
           else:
               wrong+=1
       #return correct/ (correct+wrong)
       return correct/ (correct+wrong)
   def predict(self,x):
       return np.argmax(self.forward(x))
   def latent_features(self,x,index):
       x.shape=(-1,1)
       activation=x
       activations=[x]
       for i in range(self.num_layers-1):
            w=self.weights[i]
            b=self.biases[i]
            z = np.dot(w, activation)+b
            if i==(self.num_layers-2):
                activation=softmax(z)    #last layer softmax
                continue
            activation = relu(z)
            activations.append(activation)
       return activations[-1][index]
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
def relu(z,slope=0.1):
    return np.where(z > 0, z, slope*z)
def relu_prime(z,slope=0.1):
    return np.where(z > 0, 1, slope)
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z),axis=0) 
def softmax_prime(z):
    return
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
    train_size=int(N*0.80)

    train_X=X[:train_size]
    train_Y=Y[:train_size]
    test_X=X[train_size:]
    test_Y=Y[train_size:]
    
    #-----------------------
    #------Record----------
    train_precisions=[]
    test_precisions=[]
    max_test=0
    start_observe=0
    print_interval=50
    network.SGD(train_X,train_Y)    #contain updating
    
    #--------------
    print(network.Error(test_X,test_Y))
    for t in range(epochs):
        network.SGD(train_X,train_Y)
        precision_train=network.Error(train_X, train_Y)
        precision_test=network.Error(test_X, test_Y)
        train_precisions.append(precision_train)
        test_precisions.append(precision_test)
        if t%print_interval==0:
            print("times:",t," train: ",precision_train," test:",precision_test)
        if precision_test>max_test :
            max_test=precision_test   
            
    #-------plot--------   
    plt.ylabel("Precision") # y label
    plt.xlabel("epochs") # x label
    plt.plot(range(start_observe,epochs),train_precisions, color=(255/255,100/255,100/255),label = 'training set')
    plt.plot(range(start_observe,epochs),test_precisions, '--', color=(100/255,100/255,255/255),label = 'testing set')
    
    plt.title("NNNN")
    plt.legend()
    plt.show()
    
    print("Max_test_precision:",max_test)
    
    #-------------------
    return

##################################################################

####---------------DATA PREPROCESSING----------------#####
data=data.sample(frac=1).reset_index(drop=True)
target_data=data.iloc[:,-1]
input_data=data.iloc[:,:-1]

input_data=input_data.drop(columns=input_data.columns[1]) #all zero 
#-----standard normalization input----
input_mean=input_data.mean()
input_std=input_data.std()

normalized_input=input_data     #data seems have already been processed
#normalized_target=target_data   
#-----one hot----target g b to 0 1
target_table={'g':[1,0],
              'b':[0,1]}
normalized_list=[]
for i in range(len(target_data)):
    normalized_list.append(target_table[target_data[i]])

#####----------------------------------------------#####

#%%
#---------start-------
X=normalized_input.to_numpy()
Y=np.array(normalized_list)
width=[33,132,3,2] 
biases, weights=init_layers(width)
network=Network(biases, weights,lr=0.23)
train(X,Y,network,500)
#%%
#train for plot latent feature
"""
def train(X, Y, network, epochs):
    #-----Hyper parameter----------
    N,D=X.shape
    train_size=int(N*0.80)

    train_X=X[:train_size]
    train_Y=Y[:train_size]
    test_X=X[train_size:]
    test_Y=Y[train_size:]
    
    
    #-----------------------
    #------Record----------
    train_precisions=[]
    test_precisions=[]
    max_test=0
    start_observe=0
    print_interval=50
    network.SGD(train_X,train_Y)    #contain updating
    
    checkpoint_1=8
    checkpoint_2=450
    
    good_1=[]
    good_2=[]
    bad_1=[]
    bad_2=[]
    #----------------------
    print(network.Error(test_X,test_Y))
    for t in range(epochs):
        network.SGD(train_X,train_Y)
        precision_train=network.Error(train_X, train_Y)
        precision_test=network.Error(test_X, test_Y)
        train_precisions.append(precision_train)
        test_precisions.append(precision_test)
        if t%print_interval==0:
            print("times:",t," train: ",precision_train," test:",precision_test)
        if precision_test>max_test :
            max_test=precision_test   
        if (t==checkpoint_1)or(t==checkpoint_2):
            for x,y in zip(test_X,test_Y):
                if y[0]==1:
                    good_1.append(network.latent_features(x, 1))
                    good_2.append(network.latent_features(x, 2))
                else:
                    bad_1.append(network.latent_features(x, 1))
                    bad_2.append(network.latent_features(x, 2))
            #plt.subplot(1, 2, cur)
            plt.figure(figsize=(8, 6))
            title="2D_feature_at_epochs_"+str(t)
            plt.title(title)
            plt.scatter(good_1, good_2, color = '#990000',s = 50,alpha =0.5,label = 'good')
            plt.scatter(bad_1, bad_2, color = '#009900',s = 50,alpha = 0.5,label = 'bad')
            plt.legend()
            plt.savefig(title+".png")
            good_1=[]
            good_2=[]
            bad_1=[]
            bad_2=[]
    return
"""