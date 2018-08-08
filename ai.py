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
        
    #event contains four elements. First one is the last state, second one new state, third one
    #last action, last one is the last reward
    #We need to push event to the memory up there
   def push(self, event):     
        self.memory.append(event)
        #if the capacity of the mem go over the capacity that allows, we delete the first element,
        #which is the last state to balance out the number
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batchSize):
        #for the zip function, it restructures the list
        #Example, we have list((1,2,3), (4,5,6)), then zip(*list) = ((1,4), (2,3), (5,6))
        #In our structure, we want each element in each of their own batch
        #One batch for reward, one batch for state and another batch for actions
        #Then we put each of the batch into Pytorch variable and each variable would have it's
        #own gradient
        samples = zip(*random.sample(self.memory, batchSize))
        
        #lambda function will take the samples, concatinate them with the first dimension 0 to align
        #all the element of the sample together. Then we convert the tensor into som torch variables that
        #contain both tensors and the gradient
        #So that later when we applying the gradient descent, we can differentiate to update the weights
        return map(lambda x: Variable(cat(x, 0)), samples)
    
#Implementing Deep Q Learning
class Dqn():
    #gamma is a delay coiefficient
    def __init__(self, inputSize, nbAction, gamma):
        self.gamma = gamma
        #Sliding window of the mean of the last 100 reward which we use to evaluate the evolution
        #of the AI performance
        #Initialize with the empty list
        self.rewardWindow = []
        self.model = Network(inputSize, nbAction)
        self.memory = ReolayMemory(100000)
        
        #There are many and many optimizer we can choose from optim
        #we are connecting the Adam optimizer with the neural network class
        #then we have a learning rate for the AI
        #Again the number is based on several of training time and we came up with this number
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        
        #A tensor for the input state with a number of dimemsion the input state needs
        #We also need a fake dimension that corresponding to the batch
        #We did this because in pyTorch it needs more than a vector, it needs to be a torch tensor
        #and a fake dimension. We need this dimension since the last state will be the input of the
        #neural network and nural network only accepts batch type.
        #Long story short, we create a torch tensor and a dimension responding to a batch in a
        #first dimension
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        
        #We have 3 actions to begin with and they are the changing of the car's angle
        #Its either 0, 20 or -20; so, we can put it 0 as an initialization
        self.last_action = 0
        self.last_reward = 0
        
        
        
        
        
        
        
        
        
        