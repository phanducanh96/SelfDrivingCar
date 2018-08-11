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
        #all the element of the sample together. Then we convert the tensor into some torch variables that
        #contain both tensors and the gradient
        #So that later when we applying the gradient descent, we can differentiate to update the weights
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
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
        self.memory = ReplayMemory(100000)
        
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
        self.lastState = torch.Tensor(inputSize).unsqueeze(0)
        
        #We have 3 actions to begin with and they are the changing of the car's angle
        #Its either 0, 20 or -20; so, we can put it 0 as an initialization
        self.lastAction = 0
        self.lastReward = 0
        
    def selectAction(self, state):
        #Soft max function (best action to play but at the same time, explore other actions)
        #We convert the torch tensor state to a tensor variable
        #We dont want all the gradient and graph of all computation of the nn module; so we add volitile = true
        #Save memory and performance
        #If we set the temperature to 0, the AI would not activate
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) 
        
        #Temperature = 100
        #Temperature variable represents the certainty of which action is going to play
        #Example, we have softmax([1,2,3]*1) = [0.04, 0.11, 0.850]. Then increasing temp to 3 we have
        #softmax([1,2,3]*3) = [0, 0.02, 0.98]
        
        #We get random draw from the probability
        action = probs.multinomial()
        #Do this to select the correct dimension, not the fake dimension that corressponding to
        #the batch
        return action.data[0,0]
    
    def learn(self, batchState, batchNextState, batchReward, batchAction):
        #We only want 1 action, the chosen one so we use the function gather
        #Batch Action doesnt have a fake dimension; so we want to add into it so it can match with
        #the batch state
        #but then we want the output not being in a batch but a simple form, a variable tensor
        #we have to kill the fake dimension by squeeze function
        
        #output = previous state
        outputs = self.model(batchState).gather(1, batchAction.unsqueeze(1)).squeeze(1)
        
        #next output = current state
        #We use detach all of the outputs due to the states and transitions to take the max of the
        #Q value. The action is represented in index 1, we use 1 and the state is represendted in
        #index 0, we use 0 to get the max of the q values of the enxt state
        nextOutput = self.model(batchNextState).detach().max(1)[0]
        
        #Equation in the AI handbook
        target = self.gamma*nextOutput + batchReward
        
        #TD stands for temperal different
        TdLoss = F.smooth_l1_loss(outputs, target)
        
        #Use optimizer to perform gradient descent and update the weights
        #we need to initialize each time of the loop for the optimizer
        self.optimizer.zero_grad()
        
        #Then we perform backward propagation
        #retaun variables is to free some of the memory
        #better in performance
        TdLoss.backward(retain_variables = True)
        
        #Then we update the weights
        self.optimizer.step()
        
    def update(self, reward, newSignal):
        #Creating new state by converting the new signal into a torch tensor
        #Again, add a fake dimension responding to the batch
        newState = torch.Tensor(newSignal).float().unsqueeze(0)
        #We have to convert the lastAction and lastReward variables to torch tensors
        #For the lastAction, its an int so we have to cast it; but for the last reward, its a float,
        #so just leave it
        self.memory.push((self.lastState, newState, torch.LongTensor([int(self.lastAction)]), torch.Tensor([self.lastReward])))
        #Qw place the new action after reaching the new state
        action = self.selectAction(newState)
        
        if len(self.memory.memory) > 100:
            #Return all the variables
            batchState, batchNextState, batchAction, batchReward  = self.memory.sample(100)
            self.learn(batchState, batchNextState, batchReward, batchAction)
        #update last state and last action since we are now moving on to the new state and the new
        #action
        self.lastAction = action
        self.lastState = newState
        #reward is calculated in map.py (conditional if else)
        self.lastReward = reward
        #update the reward window by appending the reward to the window
        self.rewardWindow.append(reward)
        
        #By doing this, we have a fixed size of the window
        #save memory and time
        if len(self.rewardWindow) > 1000:
            del self.rewardWindow[0]
        
        
        return action
    
    def score(self):
        #compute the mean of all the reward in the reward window
        #Get the sum then devided by the size of the window
        #We have a + 1 so that the donominator won't be 0 at any time
        return sum(self.rewardWindow) / (len(self.rewardWindow)+1)
    
    #For this function, we want to save the weight of the model, which is the memory
    #and the optimizer
    def save(self):
        torch.save({'stateDict': self.model.state_dict, 
                    'optimizer': self.optimizer.state_dict,}
                    , 'lastBrain.pth')
        
    #Load what we saved before
    def load(self):
        if os.path.isfile('lastBrain.pth'):
            print("=> loading checkpoint...")
            checkpoint = torch.load('lastBrain.pth')
            self.model.load_state_dict(checkpoint['stateDict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done!")
        else:
            print("No checkpoint b!tch")
        
        
        