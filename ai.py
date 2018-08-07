#AI
#Importing libraries

import numpy as np
import random
#load and save the model
import os
#best library for neural networks since it can handle dynamic graph
import torch
#module contains all tools for the neural networks
import torch.nn as nn
#import all types of functions in the neural networks
import torch.nn.functional as F
#Optimizer library to performing gradient descent
import torch.optim as optim
#...
import torch.autograd as autograd
from torch.autograd import Variable

#Creating the architecture of the neural network

#Child class of a class nn.Module (using inheritance in this case)
class Network(nn.Module):
    
    def __init__(self, inputSize, nbAction):
        #this function inherits from the nn module, uses all the tool from the nn module
        super(Network, self).__init__()
        self.inputSize = inputSize
        self.nbAction = nbAction
        
        #Now we need to specify the connections between the layers
        #We have 2 full connections. Input Layer to Hidden Layer and Hidden Layer to Output Layer
        
        #This function fully connects the neurons from input layer to neurons in hidden layer
        #Note that the second param is mainly based on testing, but now we are putting 30 since it gave
        #the best results; however, feel free to change or whatsoever
        self.fc1 = nn.Linear(inputSize, 30)
        #First param here is 30 since its between hidden layer to ouput layer.
        #Hidden layer again has 30 hidden neurons because of many testing
        #Output neurons are based on how many actions the object has
        self.fc2 = nn.Linear(30, nbAction)
     
    #function to activate the neurons, return the Q value
    def forward(self, state):
        #activate the hidden neuron
        #state in this case is represented as a hidden state
        #Input neuron
        x = F.relu(self.fc1(state))
        #Q-Value for the output neuron
        q_values = self.fc2(x)
        return q_values
    
#Implementing Experience Replay
class ReplayMemory(object):
    
    def __init__(self, capacity):
        #capacity is the maximum number of transitions we want to have in our memory of events
        self.capacity  = capacity
        self.memory = []
        
        
        
        
        
        
        
        
        
        
        
        
        
        