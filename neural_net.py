# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:46:00 2017

@author: visak
"""

import numpy as np
import pandas as pd
from scipy.io import arff

def read_data(file_to_read):
    dataset = arff.loadarff(file_to_read)
    data = pd.DataFrame(dataset[0])
    for i in range(0,len(data['Class'])):
        data.loc[i,'Class'] = data.loc[i,'Class'].decode()
    return data

def create_train_test(data,num_folds):
    data=data.sample(frac=1).reset_index(drop=True)
    train_data_size = int((num_folds - 1)*data.shape[0]/num_folds)
    train_data = data[0:train_data_size]
    test_data = data[train_data_size+1:data.shape[0]]
    return train_data,test_data

def prepare(data):
    Xdata = data.drop(labels = 'Class',axis = 1)
    Y_vals = data['Class'].values
    Y_vals[Y_vals=='Mine'] = 1
    Y_vals[Y_vals=='Rock'] = 0
    X_norm = (Xdata - Xdata.mean())/(Xdata.max()-Xdata.min())
    X =  X_norm.as_matrix()
    Y = np.row_stack(Y_vals)
    return X,Y

def sigmoid(x):
    x= x.astype(float)
    return 1/(1+np.exp(-x))

def deriv_sigmoid(x):
    return x*(1-x)
 
def batches(X,Y,size):
    #for i in np.arange(0,X.shape[0],size):
        #yield X[i:i+size,],Y[i:i+size]
        yield X,Y

def neural_net_train(train_data,epochs,folds,learn_rate):
    
    X,Y = prepare(train_data)
    
    weight_hidden = np.random.uniform(size=(X.shape[1],X.shape[0])) 
    bias_hidden = np.random.uniform(size=(1,X.shape[0]))
    weight_output = np.random.uniform(size=(X.shape[0],1))
    bias_output = np.random.uniform(size=(1,1))
    
    batch_size = 32
    batches(X,Y,batch_size)
    
    for i in range(0,epochs):
        for (xbatch,ybatch) in batches(X,Y,batch_size):
            #Forward propogation
            hidden_layer_inputs = xbatch.dot(weight_hidden)
            hidden_layer_inputs_with_bias = hidden_layer_inputs + bias_hidden
            hidden_layer_values = sigmoid(hidden_layer_inputs_with_bias)
            
            output_inputs = hidden_layer_values.dot(weight_output)
            output_inputs_with_bias = output_inputs + bias_output
            output_values = sigmoid(output_inputs_with_bias)

            
            #backward propogation
            error = ybatch - output_values

            slope_of_output = deriv_sigmoid(output_values)
            slope_of_hidden = deriv_sigmoid(hidden_layer_values)
             
            delta_of_output = error*slope_of_output
            
            error_of_hidden  = weight_output.T.dot(delta_of_output)             
            delta_of_hidden = error_of_hidden * slope_of_hidden
             
 
            weight_output_increment = (hidden_layer_values.T.dot(delta_of_output))*learn_rate 
            weight_output =  weight_output + weight_output_increment                         
            bias_output_increment = np.sum(delta_of_output,axis=0,keepdims= True)*learn_rate
            bias_output = bias_output_increment + bias_output
        
            weight_hidden_increment = (xbatch.T.dot(delta_of_hidden))*learn_rate
            weight_hidden = weight_hidden + weight_hidden_increment
            bias_hidden_increment = np.sum(delta_of_hidden,axis=0,keepdims= True)*learn_rate
            bias_hidden = bias_hidden + bias_hidden_increment


    return weight_hidden,bias_hidden,weight_output,bias_output                     
            
            
def prediction(test_data,weight_hidden,bias_hidden,weight_output,bias_output): 
    accuracy_count = 0          
    X,Y = prepare(test_data)
    hidden_layer_input = X.dot(weight_hidden) + bias_hidden
    hidden_layer = sigmoid(hidden_layer_input)

    output_layer_input = hidden_layer.dot(weight_output)+ bias_output
    output_layer =  sigmoid(output_layer_input)
   
    for i in range(0,Y.shape[0]):
        if(output_layer[i][0] > 0.5):
            output_layer[i][0] =1
        else:
            output_layer[i][0] =0
        if(Y[i][0] == output_layer[i][0]):
            accuracy_count += 1
    accuracy = accuracy_count/Y.shape[0]
    print("correct",accuracy_count)
    print("total",Y.shape[0])
    print("accuracy",accuracy)
    
    


if __name__ == '__main__':
    #train_file = str(sys.argv[1])
    #num_folds = int(sys.argv[2])
    #learning_rate = int(sys.argv[3])
    #num_epochs = int(sys.argv[4])
    num_folds = 5
    num_epochs = 1000
    learning_rate = 0.1
    train_file='sonar.arff'
    data = read_data(train_file)
    train_data,test_data = create_train_test(data,num_folds)
    weight_hidden,bias_hidden,weight_output,bias_output = neural_net_train(train_data,num_epochs,num_folds,learning_rate)
    prediction(test_data,weight_hidden,bias_hidden,weight_output,bias_output)
    