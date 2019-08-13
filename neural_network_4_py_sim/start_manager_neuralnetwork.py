# -*- coding: utf-8 -*-

from NeuralNetwork import NeuralNetwork
import numpy as np
import scipy.special as sp
import matplotlib.pyplot
#%matplotlib inline
import time


def NN4py():
    
    # load the mnist training data CSV file into a list
    training_data_file = open("./code_from_book/makeyourownneuralnetwork/mnist_dataset/mnist_train.csv","r")
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # load the mnist test data CSV file into a list
    testing_data_file = open("./code_from_book/makeyourownneuralnetwork/mnist_dataset/mnist_test.csv","r")
    testing_data_list = testing_data_file.readlines()
    testing_data_file.close()

    # initialise the neural network base
    # number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 300
    output_nodes = 10
    # learning rate
    learning_rate = 0.1
    # create instance of neural network    
    nw = NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

    # epochs is the number of times the training data set is used for training
    epochs = 5
    time_train_start = time.time()
    print("strat training now !!!!")
    for e in range(epochs):
        # go throgh all records in the training data set
        for record in training_data_list:
	    # split the record by the ',' commas
	    all_values = record.split(',')
            # scale and shift the inputs
	    '''
	    数据是像素值，范围是0-255，下面的代码是让输入数据偏移到0.01-1.0的范围，原因如下：
	    选择0.01作为范围最低点是为了避免0值输入最终人为造成权重更新失败
	    选择1.0作为上限值是因为不需要避免输入，只需要避免输出值是1.0
	    我们使用的逻辑函数，即激活函数，输出范围是0.0-1.0，但事实上不能达到这两个边界值，这是逻辑函数的极限，逻辑函数仅接近这两个极限
	    训练的目标也一样，用0.01代表0，用0.99代表1
	    '''

            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
	    # create the target output values (all 0.01, except the desired label which is 0.99)
	    targets = np.zeros(output_nodes) + 0.01
	    # all_values[0] is the target label for this record
	    # 下标是目标数字，也是输出层10个节点对应10个数字中想要激活的那个节点
	    targets[int(all_values[0])] = 0.99
	    nw.train(inputs, targets)
	    pass
	pass
    print("training done !!!!")
    time_train_end = time.time()
    time_train_final = time_train_end - time_train_start
    print'training time is ',time_train_final,'s'
    

    # scorecard for how well the network performs, initially empty
    scorecard = []
    time_test_start = time.time()
    print("start testing now !!!!")
    # go through all the records in the test data set
    for record in testing_data_list:
	# split the record by the ',' commas
	all_values = record.split(',')
	# correct answer is first value
	correct_label = int(all_values[0])
	# print correct_label, 'correct label '
	# scale and shift the inputs
	inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
	# query the network
	outputs = nw.query(inputs)
	# the index of the highest value corresponds to the label
	label = np.argmax(outputs)
	# np.argmax找到数组中的最大值并返回它的位置
	# print label,"network's answer"
	# append correct or incorrect to list
	if (label == correct_label ):
	    # network's answer matches correct answer, add 1 to scorecard
	    scorecard.append(1)
	else:
	    # network's answer doesn't match correct answer, add 0 to scorecard
	    scorecard.append(0)
	    pass
	pass
    print("testing done !!!!")
    time_test_end = time.time()
    time_test_final = time_test_end - time_test_start
    print"testing time is ",time_test_final,'s'
    # calculate the performance score, the fraction of correct answers
    scorecard_array = np.asarray(scorecard)
    print(scorecard_array.sum())
    print(scorecard_array.size)
    zhengquede = float(scorecard_array.sum())
    zongshu = float(scorecard_array.size)
    zhunquelv = zhengquede / zongshu
    print'testing accurancy is ',zhunquelv * 100,'%'

if __name__ == '__main__':
    
    print("begin to run !!!!")
    time_start = time.time()
    NN4py()
    time_end = time.time()
    time_final = time_end - time_start
    print"the neural network works ",time_final,"s"
    
