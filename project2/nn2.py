#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:45:14 2017

@author: huanwenxu
"""

import numpy as np
from keras.datasets import mnist
import matplotlib.image as mpimg # mpimg 用于读取图片
import os

# 配置信息
class Config:
    nn_input_dim = 784  # 输入维度
    nn_output_dim = 10  # 输出维度
    
    epsilon = 0.01  # 学习率
    reg_lambda = 0.01  # 正则力度

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
def inference(model, x, Path='0'):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    if Path != "0":
        for dir in os.listdir(Path):
            if dir.split('.')[1] !='png':
                continue
            img = mpimg.imread(Path+dir)
            img_gray = rgb2gray(img)
            img_gray = img_gray.reshape(1,784)/255.0
            net1 = img_gray.dot(W1) + b1
            h1 = np.tanh(net1)
            net2 = h1.dot(W2) + b2
            exp_scores = np.exp(net2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            print(np.argmax(probs, axis=1));
    else:        
        net1 = x.dot(W1) + b1
        h1 = np.tanh(net1)
        net2 = h1.dot(W2) + b2
        exp_scores = np.exp(net2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

def train_model(X, y, nn_hdim=30, num_passes=40,  model={}):
    num_examples = len(X)
    np.random.seed(0)
    if(model=={}):  # 随机初始化提高表现
        W1 = np.random.randn(Config.nn_input_dim, nn_hdim)
        b1 = np.zeros((1, nn_hdim))
        W2 = np.random.randn(nn_hdim, Config.nn_output_dim)
        b2 = np.zeros((1, Config.nn_output_dim))
    else:
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']


    for i in range(0, num_passes):

        # 前向传播
        net1 = X.dot(W1) + b1
        h1 = np.tanh(net1)
        net2 = h1.dot(W2) + b2
        exp_scores = np.exp(net2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # 反向传播
        delta3 = probs
        delta3[range(num_examples), y] -= 1 # 输出和理想结果的差值
        dW2 = (h1.T).dot(delta3)    # W2的梯度
        db2 = np.sum(delta3, axis=0, keepdims=True) # b2的梯度
        delta2 = delta3.dot(W2.T) * (1 - np.power(h1, 2))   
        dW1 = np.dot(X.T, delta2) # W1的梯度
        db1 = np.sum(delta2, axis=0)    # b1的梯度

        # 增加正则项
        dW2 += Config.reg_lambda * W2
        dW1 += Config.reg_lambda * W1

        # 梯度下降更新各个参数
        W1 += -Config.epsilon * dW1
        b1 += -Config.epsilon * db1
        W2 += -Config.epsilon * dW2
        b2 += -Config.epsilon * db2

        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model   

def test_accuracy(model, X_test,y_test):
    y_=inference(model,X_test)
    num_examples = len(X_test)
    test = y_-y_test
    count=0
    for i in range(num_examples):
        if test[i]==0:
            count +=1
    return count
    
if __name__ == "__main__":
    '''
    用了keras里的mnist数据库
    '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255  # 归一化
    X_test /= 255   # 归一化
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    t_model={}
    for i in range(0,600):
        t_model = train_model(X_train[100*i:100*i+100,:], y_train[100*i:100*i+100], 40,  model=t_model)
    print(test_accuracy(t_model, X_test, y_test),'correct predicts out of 10000 samples')