# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:43:31 2019

@author: Ahmed Alhag
"""


# Momentum Gradiant Descent
# Dataset from https://www.kaggle.com/c/digit-recognizer/data

from __future__ import print_function, division
from builtins import range

# Check that the gpu/cpu is being used properly
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Import Default Libraries
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Impoting ML Libraries
from util import get_normalized_data, error_rate, cost, y2indicator
from mlp import forward, derivative_w2, derivative_w1, derivative_b2, derivative_b1

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

# Initializing variables
max_iter = 10
print_period = 10

X, Y = get_normalized_data()
reg = 0.01

X_train = X[:-1000,]
Y_train = Y[:-1000,]
X_test = X[-1000:,]
Y_test = Y[-1000:,]

Y_train_ind = y2indicator(Y_train)
Y_test_ind = y2indicator(Y_test)

N, D = X_train.shape
batch_sz = 500
n_batch = N // batch_sz

M = 300
K = 10

# Setting up the Moments
W1_0 = np.random.randn(D, M) / np.sqrt(D)
W2_0 = np.random.randn(M, K) / np.sqrt(M)

b1_0 = np.zeros(M)
b2_0 = np.zeros(K)

W1, W2, b1, b2 = W1_0.copy(), W2_0.copy(), b1_0.copy(), b2_0.copy()

# 1st moment
mW1, mW2, mb1, mb2 = 0, 0, 0, 0

# 2nd moment
vW1, vW2, vb1, vb2 = 0, 0, 0, 0

# Hyperparams
lr0 = 0.001
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

# 1. Adam optimizer
loss_adam = []
err_adam = []
t = 1

for i in range(max_iter):
    for j in range(n_batch):
        X_batch = X_train[j*batch_sz:(j*batch_sz + batch_sz), ]
        Y_batch = Y_train_ind[j*batch_sz:(j*batch_sz + batch_sz),]
        pY_batch, Z = forward(X_batch, W1, b1, W2, b2)
        
        # Update the gradiant
        gW2 = derivative_w2(Z, Y_batch, pY_batch) + reg*W2
        gb2 = derivative_b2(Y_batch, pY_batch) + reg*b2
        gW1 = derivative_w1(X_batch, Z, Y_batch, pY_batch, W2) + reg*W1
        gb1 = derivative_b1(Z, Y_batch, pY_batch, W2) + reg*b1
        
        # Update new Moments
        mW1 = beta1 * mW1 + (1 - beta1) * gW1
        mb1 = beta1 * mb1 + (1 - beta1) * gb1
        mW2 = beta1 * mW2 + (1 - beta1) * gW2
        mb2 = beta1 * mb2 + (1 - beta1) * gb2
        
        # Update new Velocity
        vW1 = beta2 * vW1 + (1 - beta2) * gW1 * gW1
        vb1 = beta2 * vb1 + (1 - beta2) * gb1 * gb1
        vW2 = beta2 * vW2 + (1 - beta2) * gW2 * gW2
        vb2 = beta2 * vb2 + (1 - beta2) * gb2 * gb2
        
        # Bias Correction Update
        correction1 = 1 - beta1**t
        hat_mW1 = mW1/correction1
        hat_mb1 = mb1/correction1
        hat_mW2 = mW2/correction1
        hat_mb2 = mb2/correction1
        
        correction2 = 1 - beta2**t
        hat_vW1 = vW1/correction2
        hat_vb1 = vb1/correction2
        hat_vW2 = vW2/correction2
        hat_vb2 = vb2/correction2
        
        # Update T
        t += 1
        
        # Apply Update to parameters
        W1 = W1 - lr0 * hat_mW1 / np.sqrt(hat_vW1 + eps)
        b1 = b1 - lr0 * hat_mb1 / np.sqrt(hat_vb1 + eps)
        W2 = W2 - lr0 * hat_mW2 / np.sqrt(hat_vW2 + eps)
        b1 = b2 - lr0 * hat_mb2 / np.sqrt(hat_vb2 + eps)
        
        if j % print_period == 0:
            pY,_ = forward(X_test, W1, b1, W2, b2)
            l = cost(pY, Y_test_ind)
            print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, l))
            
            err = error_rate(pY, Y_test)
            err_adam.append(err)
            print("Eror rate: ", err)
            
pY,_ = forward(X_test, W1, b1, W2, b2)
print("Final error rate: ", error_rate(pY, Y_test))

# 2. RMSprop with momentum
W1 = W1_0.copy()
b1 = b1_0.copy()
W2 = W2_0.copy()
b2 = b2_0.copy()

loss_rms = []
err_rms = []

# Comparable hyperparameters for fair comparison
lr0 = 0.001
mu = 0.9
decay_rate = 0.999
eps = 1e-8

# rmsprop cache
cache_W1, cache_W2, cache_b1, cache_b2 = 1, 1, 1, 1

# Reinitialize Momentum
dW1, dW2, db1, db2 = 0, 0, 0, 0

for i in range(max_iter):
    for j in range(n_batch):
        X_batch = X_train[j*batch_sz:(j*batch_sz + batch_sz),]
        Y_batch = Y_train_ind[j*batch_sz:(j*batch_sz + batch_sz), ]
        pY_batch, Z = forward(X_batch, W1, b1, W2, b2)
        
        # Update Derivatives
        gW2 = derivative_w2(Z, Y_batch, pY_batch) + reg*W2
        cache_W2 = decay_rate * cache_W2 + (1 - decay_rate)*gW2*gW2
        dW2 = mu * dW2 + (1 - mu) * lr0 * gW2/ (np.sqrt(cache_W2) + eps)
        W2 -= dW2
        
        gb2 = derivative_w2(Z, Y_batch, pY_batch) + reg*W2
        cache_b2 = decay_rate*cache_b2 + (1 - decay_rate)*gb2*gb2
        db2 = mu * db2 + (1 - mu) * lr0 * gb2 / (np.sqrt(cache_b2) + eps)
        b2 -= db2
        
        gW1 = derivative_w1(X_batch, pY_batch) + reg*b2
        cache_W1 = decay_rate * cache_W1 + (1 - decay_rate)*gb2*gb2
        db2 = mu * dW1 + (1 - mu) * lr0 * gW1 / (np.sqrt(cache_W1) + eps)
        W1 -= dW1
        
        gb1 = derivative_b1(Z, Y_batch, pY_batch, W2) + reg*b1
        cache_b1 = decay_rate * cache_b1 + (1 - decay_rate) * gb1 * gb1
        b1 -= db1
        
        if j % print_period == 0:
            pY,_ = forward(X_test, W1, b1, W2, b2)
            l = cost(pY, Y_test_ind)
            print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, l))
            
            err = error_rate(pY, Y_test)
            err_rms.append(err)
            print("Eror rate: ", err)
            
pY,_ = forward(X_test, W1, b1, W2, b2)
print("Final error rate: ", error_rate(pY, Y_test))

plt.plot(loss_adam, label = 'adam')
plt.plot(loss_rms, label = 'rmsprop')
plt.legend()
plt.show()

















