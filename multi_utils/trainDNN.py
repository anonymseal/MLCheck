#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import numpy as np
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import csv as cv
import sys
sys.path.append("../")


# In[55]:


class LinearNet(nn.Module):
    def __init__(self, input_size):
        super(LinearNet, self).__init__()
        with open('param_dict.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.mydict = dict(reader)   
        self.num_layers = int(self.mydict['no_of_layers'])
        self.layers_size = int(self.mydict['layer_size'])
        self.output_size = int(self.mydict['no_of_class'])
        self.linears = nn.ModuleList([nn.Linear(input_size, self.layers_size)])
        self.linears.extend([nn.Linear(self.layers_size, self.layers_size) for i in range(1, self.num_layers-1)])
        self.linears.append(nn.Linear(self.layers_size, self.output_size))
    
    def forward(self, x):
        for i in range(0, self.num_layers-1):
            x = F.relu(self.linears[i](x))
        x = self.linears[self.num_layers-1](x)    
        return F.log_softmax(x, dim=1)  


# In[ ]:


class weightConstraint(object):
    def __init__(self):
        pass
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            w=module.weight.data
            w=torch.clamp(w, min=-10, max=10)
            module.weight.data=w


# In[58]:


def functrainDNN():
    df = pd.read_csv('OracleData.csv')
    data = df.values
    with open('param_dict.csv') as csv_file:
        reader = cv.reader(csv_file)
        mydict = dict(reader) 
    
    no_of_class = int(mydict['no_of_class'])
    X = data[:,:-no_of_class]
    y = data[:, -no_of_class:]  
    EPOCH = int(mydict['no_EPOCHS'])
    X_train = torch.from_numpy(X).float()
    y_train = torch.tensor(y)
    
    net = LinearNet(input_size=df.shape[1]-no_of_class)
    constraints=weightConstraint()
    net(X_train)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-1)

    for epoch in range(0, EPOCH):
        optimizer.zero_grad()
        output = net(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        for i in range(0, len(net.linears)):
            net.linears[i].apply(constraints)
    
    MODEL_PATH = 'Model/dnn_model_multi'
    torch.save(net, MODEL_PATH)   
    


# In[ ]:





# In[ ]:





# In[ ]:




