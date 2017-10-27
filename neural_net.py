# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:46:00 2017

@author: visakh
"""

import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt

def read_data(file_to_read):
    dataset = arff.loadarff(file_to_read)
    data = pd.DataFrame(dataset[0])
    for i in range(0,len(data['Class'])):
        data.loc[i,'Class'] = data.loc[i,'Class'].decode()
    return data

def create_train_test(data,num_folds,learning_rate,num_epochs):
    accuracy_list=[]
    data=data.sample(frac=1).reset_index(drop=True)
    data_first = data[data['Class']=='Mine']
    data_second = data[data['Class']=='Rock']
    test_data_size_first = data_first.shape[0]/num_folds
    test_data_size_second = data_second.shape[0]/num_folds
    for i in np.arange(0,num_folds):
        
        if ((i+1)*test_data_size_first) <= data_first.shape[0]: 
            test_data_first = data_first[int(i*test_data_size_first):int((i+1)*test_data_size_first)]
            if i !=0:                
                train_data_first = data_first[0:int(i*test_data_size_first)]
                train_data_first.append(data_first[int(((i+1)*test_data_size_first)+1):])
            else:
                train_data_first = data_first[int(((i+1)*test_data_size_first)+1):]
        else:
            test_data_first = data_first[int(i*test_data_size_first):]
            if i!=0:
                train_data_first = data_first[0:int(i*test_data_size_first)]
            
        if ((i+1)*test_data_size_second) <= data_second.shape[0]: 
            test_data_second = data_second[int(i*test_data_size_second):int((i+1)*test_data_size_second)]
            if i !=0:
                train_data_second = data_second[0:int(i*test_data_size_second)]
                train_data_second.append(data_second[int(((i+1)*test_data_size_second)+1):])
            else:
                train_data_second = data_second[int(((i+1)*test_data_size_second)+1):]
        else:
            test_data_second = data_second[int(i*test_data_size_second):]
            if i !=0:
                train_data_second = data_second[0:int(i*test_data_size_second)]        

        test_datas = [test_data_first,test_data_second]
        test_data = pd.concat(test_datas)
        train_datas = [train_data_first,train_data_second]
        train_data = pd.concat(train_datas)
        train_data=train_data.sample(frac=1).reset_index(drop=True)
        test_data=test_data.sample(frac=1).reset_index(drop=True)
        weight_hidden,weight_output = neural_net_train(train_data,num_epochs,num_folds,learning_rate)
        accuracy = prediction(test_data,weight_hidden,weight_output,i)
        accuracy_list.append(accuracy)

    return accuracy_list

def prepare(data):
    Xdata = data.drop(labels = 'Class',axis = 1) 
    Y_vals = data['Class'].values
    Y_vals[Y_vals=='Mine'] = 1
    Y_vals[Y_vals=='Rock'] = 0
    X_norm = (Xdata - Xdata.mean())/(Xdata.max()-Xdata.min())
    X =  X_norm.as_matrix()
    X = np.c_[np.ones(X.shape[0]),X]
    Y = np.row_stack(Y_vals)
    return X,Y

def sigmoid(x):
    x= x.astype(float)
    return 1/(1+np.exp(-x))

def deriv_sigmoid(x):
#    return x*(1-x)
    return sigmoid(x)*(1-sigmoid(x))
 
def batches(X,Y,size):
    for i in np.arange(0,X.shape[0],size):
        yield X[i:i+size,],Y[i:i+size]
        #yield X,Y

def neural_net_train(train_data,epochs,folds,learn_rate):
    
    X,Y = prepare(train_data)
    
    weight_hidden = np.random.uniform(low=-1,high=1,size=(X.shape[1],X.shape[1])) 
    weight_output = np.random.uniform(low=-1,high=1,size=(X.shape[1],1))

    
    batch_size = 32
    batches(X,Y,batch_size)
    
    for i in range(0,epochs):
        for (xbatch,ybatch) in batches(X,Y,batch_size):
            #Forward propogation
            
            hidden_layer_inputs = xbatch.dot(weight_hidden)
            hidden_layer_values = sigmoid(hidden_layer_inputs)
            
            output_inputs = hidden_layer_values.dot(weight_output)
            output_values = sigmoid(output_inputs)

            
            #backward propogation
            error =  ybatch - output_values

            slope_of_output = error
            slope_of_hidden = deriv_sigmoid(hidden_layer_values)
            

            delta_of_output = slope_of_output
            error_of_hidden  = delta_of_output.dot(weight_output.T)             
            delta_of_hidden = error_of_hidden * slope_of_hidden
             
 
            weight_output_increment = (hidden_layer_values.T.dot(delta_of_output))*learn_rate 
            weight_output =  weight_output + weight_output_increment                         
        
            weight_hidden_increment = (xbatch.T.dot(delta_of_hidden))*learn_rate
            weight_hidden = weight_hidden + weight_hidden_increment


    return weight_hidden,weight_output                     
            
            
def prediction(test_data,weight_hidden,weight_output,fold): 
    global predictions
    accuracy_count = 0          
    X,Y = prepare(test_data)
    hidden_layer_input = X.dot(weight_hidden)
    hidden_layer = sigmoid(hidden_layer_input)

    output_layer_input = hidden_layer.dot(weight_output)
    output_layer =  sigmoid(output_layer_input)
    for i in range(0,Y.shape[0]):
        if(output_layer[i][0] > 0.5):
            predicted_class =1
        else:
            predicted_class =0
        if(Y[i][0] == predicted_class):
            accuracy_count += 1
        predictions = predictions + "\n" + str(fold+1) + " " + str(1 if output_layer[i][0]>0.5 else 0)  + " " + str(Y[i][0]) + " " + str(output_layer[i][0])
    accuracy = accuracy_count/Y.shape[0]
    return accuracy
    
def epoch_accuracy(data):
    num_folds = 10
    learning_rate = 0.1
    epoch_accuracy_list = []
    k = [25,50,75,100]
    for i in k:
        accuracy_list = create_train_test(data,num_folds,learning_rate,i)
        epoch_accuracy_list.append(sum(accuracy_list)/len(accuracy_list))
    plt.figure(1)
    plt.plot(k,epoch_accuracy_list)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Average accuracy")
    plt.title("Accuracy variation with number of epochs")
    plt.savefig("Accuracy variation with number of epochs.png")
    
def fold_accuracy(data):
    num_epochs = 50
    learning_rate = 0.1
    epoch_accuracy_list = []
    k = [5,10,15,20,25]
    for i in k:
        accuracy_list = create_train_test(data,num_folds,i,num_epochs)
        epoch_accuracy_list.append(sum(accuracy_list)/len(accuracy_list))
    plt.figure(2)
    plt.plot(k,epoch_accuracy_list)
    plt.xlabel("Number of folds")
    plt.ylabel("Average accuracy")
    plt.title("Accuracy variation with number of folds")
    plt.savefig("Accuracy variation with number of folds.png")


if __name__ == '__main__':
    #train_file = str(sys.argv[1])
    #num_folds = int(sys.argv[2])
    #learning_rate = int(sys.argv[3])
    #num_epochs = int(sys.argv[4]) 
    predictions = ""
    num_folds = 10
    num_epochs = 1000
    learning_rate = 0.1
    train_file='sonar.arff'
    data = read_data(train_file)
    predictions = str("**Fold**Predicted class**Actual class**Confidence of prediction**")
    accuracy_list = create_train_test(data,num_folds,learning_rate,num_epochs)
    print(accuracy_list)
    print("Total accuracy across folds: " , sum(accuracy_list)/len(accuracy_list))
    print(predictions)
    epoch_accuracy(data)
    fold_accuracy(data)
    