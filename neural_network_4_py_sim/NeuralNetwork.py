# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

import numpy as np
import scipy.special as sp
import matplotlib.pyplot
#%matplotlib inline
# make sure the plots are inside this notebook, not an external window
# scipy,special for the sigmoid function expit()


# neural network class definiton
class NeuralNetwork:
    
    # initialise the neural network
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
    
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # set learing rate
        self.lr = learningrate

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
	# 以下是一种以正态分布的方式初始化权重，中心值、标准方差、numpy数组的大小，常用
        self.wih = np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))

	# 以下是一种随机分布的方式初始化权重，普通
        # self.wih = np.random.rand(self.hnodes,self.inodes)
        # self.who = np.random.rand(self.onodes,self.hnodes)

        # activation function is the sigmoid function
        # 初始化调用激活函数
        self.activation_function = lambda x: sp.expit(x)
    
        pass

    # train the neural network
    def train(self,inputs_list,targets_list):
     
        # convert inputs list to 2d array ndmin是设置为2维的参数
        inputs = np.array(inputs_list,ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T
        
        # calculate signals into hidden layer (input for hidden layer)
        hidden_inputs = np.dot(self.wih,inputs)
        # calculate the signals emerging from hidden0 layer (经激活函数输出)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who,hidden_outputs)      
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the ( target - actual )
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T,output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors*final_outputs*(1.0-final_outputs)),np.transpose(hidden_outputs))
        # update the weights for the links between the hidden and input layers
        self.wih += self.lr * np.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),np.transpose(inputs))
        pass

    # query the neural network
    def query(self,inputs_list):
        
        # convert inputs list to 2d array
        inputs = np.array(inputs_list,ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih,inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who,hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs






































