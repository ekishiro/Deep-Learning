# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 11:25:04 2019

@author: Ahmed Alhag
"""

# Autoencoder - Movie Recommender System
# Dataset from Grouplens 100k and 1m dataset

# Check that the gpu/cpu is being used properly
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Import Default Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Importing the Keras libraries and packages
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import tensorflow as tf

# List of file paths
BinPath = os.path.abspath(os.path.join(os.path.dirname( '__file__' ), '..', 'bin'))
MovesData = BinPath + '/ml-1m/movies.dat'
UsersData = BinPath + '/ml-1m/users.dat'
RatingsData = BinPath + '/ml-1m/ratings.dat'

TrainingData = BinPath + '/ml-100k/u1.base'
TestingData = BinPath + '/ml-100k/u1.test'

# Import datasets
movies = pd.read_csv(MovesData, sep ='::', header = None, engine = 'python', encoding ='latin-1')
users = pd.read_csv(UsersData, sep ='::', header = None, engine = 'python', encoding ='latin-1')
ratings = pd.read_csv(RatingsData, sep ='::', header = None, engine = 'python', encoding ='latin-1')

# Preparing the training and test sets
training_set = pd.read_csv(TrainingData, delimiter='\t')
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv(TestingData, delimiter='\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Auto Encoder
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

# Defining the Model
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the Model
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) >0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.item()*mean_corrector)
            s +=1.
            optimizer.step()
    print('epoch: ' + str(epoch) + '- loss: ' + str(train_loss/s))

# Testing the Model
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) >0:
        output = sae(input)
        target.require_grad = False
        output[target.view(1,-1) == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.item()*mean_corrector) 
        s+=1.
print('loss: ' + str(test_loss/s))
            














