
import torch
import torchvision
import numpy as np
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import csv as cv


class ConvertDNN2logic:
    
    def __init__(self, image_data=False):
        with open('param_dict.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.paramDict = dict(reader) 
        if(self.paramDict['multi_label'] == 'True'):
            self.dnn = torch.load('Model/dnn_model_multi') 
        else:
            self.dnn = torch.load('Model/dnn_model')
        self.df = pd.read_csv('OracleData.csv')
        self.no_of_params = int(self.paramDict['no_of_params'])    
        self.no_of_hidden_layer = int(self.paramDict['no_of_layers'])
        self.no_of_layer = len(self.dnn.linears)
        self.no_of_class = self.dnn.linears[len(self.dnn.linears)-1].out_features
        self.image_data = image_data
        
    def remove_exponent(self, value):
        if('e-' in value):
            decial = value.split('e')
            ret_val = format(((float(decial[0]))*(10**int(decial[1]))), '.5f')
            return ret_val
        else: 
            return value
   
    
    def funcDNN2logic(self):
        f=open('DNNSmt.smt2', 'w')
        f.write(';Input layer neurons \n')
        for i in range(0, self.no_of_params):
            f.write(';-----------'+str(i)+'th parameter----------------- \n')
            #Initializing input features
            for j in range(0, self.dnn.linears[0].in_features):
                if(self.image_data == False):
                    f.write('(declare-fun '+self.df.columns.values[j]+str(i)+' ()')
                    if('int' in str(self.df.dtypes[j])):
                        f.write(' Int) \n')
                    else:
                        f.write(' Real) \n')
                else:
                    f.write('(declare-fun pixel'+str(j)+str(i)+' () Real) \n')
                            
            #Initializing hidden layer neurons
            for j in range(0, self.no_of_layer-1):
                for k in range(0, self.dnn.linears[j].out_features):
                    f.write('(declare-fun nron'+str(j)+str(k)+str(i)+' () Real) \n')
                    f.write('(declare-fun tmp'+str(j)+str(k)+str(i)+' () Real) \n')
            #Initializing output neurons
            for j in range(0, self.no_of_class):
                f.write('(declare-fun y'+str(j)+str(i)+' () Real) \n')
                f.write('(declare-fun tmp'+str(self.no_of_layer-1)+str(j)+str(i)+' () Real) \n')
            #Initializing extra variables needed for encoding argmax 
            for j in range(0, self.no_of_class):
                for k in range(0, self.no_of_class):
                    f.write('(declare-fun d'+str(j)+str(k)+str(i)+' () Int) \n')
            
            if(self.paramDict['multi_label'] == 'False'):
                f.write('(declare-fun Class'+str(i)+' () Int) \n')
            else:
                for j in range(0, self.no_of_class):
                    class_name = self.df.columns.values[self.df.shape[1]-self.no_of_class+j]
                    f.write('(declare-fun '+class_name+str(i)+' () Int) \n')
                    f.write('(assert (and (>= '+class_name+str(i)+' 0) (<= '+class_name+str(i)+' 1))) \n')
    
        f.write('(define-fun absoluteInt ((x Int)) Int \n')
        f.write('  (ite (>= x 0) x (- x))) \n')
        f.write('(define-fun absoluteReal ((x Real)) Real \n')
        f.write('  (ite (>= x 0) x (- x))) \n')  
        
                    
        for i in range(0, self.no_of_params):
            f.write(';-----------'+str(i)+'th parameter----------------- \n')
            f.write('\n ;Encoding the hidden layer neuron \n')
            for j in range(0, self.no_of_layer-1):
                for k in range(0, self.dnn.linears[j].out_features):
                    f.write('(assert (= ')
                    f.write('tmp'+str(j)+str(k)+str(i)+' (+')
                    if(j == 0):
                        for l in range(0, self.dnn.linears[j].in_features):
                            temp_val = round(float(self.dnn.linears[j].weight[k][l]), 2)
                            #print('temp_val before',temp_val)
                            if('e' in str(temp_val)):
                                temp_val = self.remove_exponent(str(temp_val))
                            #print('temp_val after',temp_val)    
                            if(self.image_data == False):
                                f.write('(* '+self.df.columns.values[l]+str(i)+' '+str(temp_val)+') ')
                            else:
                                f.write('(* pixel'+str(l)+str(i)+' '+str(temp_val)+') \n')
                        temp_bias = round(float(self.dnn.linears[j].bias[k]), 2)  
                        if('e' in str(temp_bias)):
                            temp_bias = self.remove_exponent(str(temp_bias))
                        f.write(str(temp_bias)+'))) \n')
                    else:
                        for l in range(0, self.dnn.linears[j].in_features):
                            temp_val = round(float(self.dnn.linears[j].weight[k][l]), 2)
                            if('e' in str(temp_val)):
                                temp_val = self.remove_exponent(str(temp_val))
                            f.write('(* nron'+str(j-1)+str(l)+str(i)+' '+str(temp_val)+')')
                        temp_bias = round(float(self.dnn.linears[j].bias[k]), 2)
                        if('e' in str(temp_bias)):
                            temp_bias = self.remove_exponent(str(temp_bias))
                        f.write(str(temp_bias)+'))) \n')

                    f.write('(assert (=> (> tmp'+str(j)+str(k)+str(i)+' 0) (= nron'+str(j)+str(k)+str(i)+
                           ' tmp'+str(j)+str(k)+str(i)+'))) \n')
                    f.write('(assert (=> (<= tmp'+str(j)+str(k)+str(i)+' 0) (= nron'+str(j)+str(k)+str(i)+
                           ' 0))) \n')
            
            f.write('\n ;Encoding the output layer neuron \n')           
            for j in range(0, self.dnn.linears[self.no_of_layer-1].out_features):
                
                f.write('(assert (= tmp'+str(self.no_of_layer-1)+str(j)+str(i)+' (+ ')
                for k in range(0, self.dnn.linears[self.no_of_layer-1].in_features):
                    temp_val = round(float( self.dnn.linears[self.no_of_layer-1].weight[j][k]), 2)
                    if('e' in str(temp_val)):
                        temp_val = self.remove_exponent(str(temp_val))
                    f.write('(* nron'+str(self.no_of_layer-2)+str(k)+str(i)+' '+str(temp_val)+')')
                temp_bias = round(float(self.dnn.linears[self.no_of_layer-1].bias[j]), 2)
                if('e' in str(temp_bias)):
                    temp_bias = self.remove_exponent(str(temp_bias))
                f.write(' '+str(temp_bias)+'))) \n')
                f.write('(assert (=> (> tmp'+str(self.no_of_layer-1)+str(j)+str(i)+' 0) (= y'+str(j)+str(i)+
                           ' tmp'+str(self.no_of_layer-1)+str(j)+str(i)+'))) \n')
                f.write('(assert (=> (<= tmp'+str(self.no_of_layer-1)+str(j)+str(i)+' 0) (= y'+str(j)+str(i)+
                           ' 0))) \n')
            f.write('\n ;Encoding argmax constraint \n')
            if(self.paramDict['multi_label'] == 'False'): 
                for j in range(0, self.no_of_class):
                    for k in range(0, self.no_of_class):
                        if(j == k):
                            f.write('(assert (= d'+str(j)+str(k)+str(i)+' 1)) \n')
                        else:
                            f.write('(assert (=> (>= y'+str(j)+str(i)+' y'+str(k)+str(i)+') (= d'
                                +str(j)+str(k)+str(i)+' 1))) \n')
                            f.write('(assert (=> (< y'+str(j)+str(i)+' y'+str(k)+str(i)+') (= d'
                                +str(j)+str(k)+str(i)+' 0))) \n')
                    
                for j in range(0, self.no_of_class):
                    f.write('(assert (=> (= (+ ')
                    for k in range(0, self.no_of_class):
                        f.write('d'+str(j)+str(k)+str(i)+' ')
                    f.write(') '+str(self.no_of_class)+') (= Class'+str(i)+' '+str(j)+'))) \n')
            else:
                for j in range(0, self.no_of_class):
                    class_name = self.df.columns.values[self.df.shape[1]-self.no_of_class+j]
                    
                    f.write('(assert (=> (> y'+str(j)+str(i)+' 0.5) (= '+class_name+str(i)+' 1))) \n')
            
            

