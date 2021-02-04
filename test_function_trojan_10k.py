from mlCheck import Assume, Assert, propCheck
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import pandas as pd
from ART_MainFiles import artGen


MAX_SAMPLES = 1000
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


model = Net()
os.system('python Dataframe2XML.py Datasets/mnist_resized.csv')

iteration_no = int(input('How many times would you want each test case to execute: \n'))

trojaned_model_arch1 = ['TrojanedModel/dnn_model_MNIST_ARCH1_10K_EPOCH1_C4_T1', 'TrojanedModel/dnn_model_MNIST_ARCH1_10K_EPOCH1_C5_T1',
                  'TrojanedModel/dnn_model_MNIST_ARCH1_10K_EPOCH1_C4_T2', 'TrojanedModel/dnn_model_MNIST_ARCH1_10K_EPOCH1_C5_T2',
                  'TrojanedModel/dnn_model_MNIST_ARCH1_10K_EPOCH1_C4_T3', 'TrojanedModel/dnn_model_MNIST_ARCH1_10K_EPOCH1_C5_T3']

white_box = ['Decision tree', 'DNN']
cont = 0
f = open('Output/trojan_attack_results_table8.txt', 'w')
t = []
f.write('§§§§----Results of NN1 is -----§§§§\n \n')
for cont in range(0, len(trojaned_model_arch1)):
    cex_count = 0
    model = trojaned_model_arch1[cont]
    if cont == 0:
        t.append(673.723)
        t.append(43.693)
        art_class = 4
        art_trigger = 'T1'
    elif cont == 1:
        t.append(673.723)
        t.append(43.693)
        art_class = 5
        art_trigger = 'T1'
    elif cont == 2:
        t.append(673.723)
        t.append(43.693)
        t.append(83.484)
        art_class = 4
        art_trigger = 'T2'
    elif cont == 3:
        t.append(673.723)
        t.append(43.693)
        t.append(83.484)
        art_class = 5
        art_trigger = 'T2'
    elif cont == 4:
        t.append(673.723)
        t.append(43.693)
        t.append(83.484)
        t.append(2137.505)
        t.append(2137.505)
        art_class = 4
        art_trigger = 'T3'
    elif cont == 5:
        t.append(673.723)
        t.append(43.693)
        t.append(83.484)
        t.append(2137.505)
        t.append(2137.505)
        art_class = 5
        art_trigger = 'T3'

    for box in white_box:
        cex_count = 0
        if box == 'Decision tree':
            f.write('--Results of MLC_DT ---\n')
        else:
            f.write('--Results of MLC_NN ---\n')

        for i in range(0, iteration_no):
            propCheck(no_of_params=1, max_samples=1500, model_type='Pytorch', model_path=model, mul_cex=False,
                xml_file='dataInput.xml', no_of_class=10, white_box_model=box, no_of_layers=2, layer_size=10, no_EPOCHS=1)

            for j in range(0, len(t)):
                Assume('x[i] = t[i]', j, t)
            if 'C4' in model:
                Assert('model.predict(x) == 4')
            else:
                Assert('model.predict(x) == 5')
            dfCexSet = pd.read_csv('CexSet.csv')
            if dfCexSet.shape[0] == 1:
                cex_count += 1

        if cont == 0:
            f.write('$$$---Probability of detected violations for T1-4:'+str(cex_count/iteration_no)+'\n')
        elif cont == 1:
            f.write('$$$---Probability of detected violations for T1-5:'+str(cex_count/iteration_no)+'\n')
        elif cont == 2:
            f.write('$$$---Probability of detected violations for T2-4:'+str(cex_count/iteration_no)+'\n')
        elif cont == 3:
            f.write('$$$---Probability of detected violations for T2-5:'+str(cex_count/iteration_no)+'\n')
        elif cont == 4:
            f.write('$$$---Probability of detected violations for T3-4:'+str(cex_count/iteration_no)+'\n')
        elif cont == 5:
            f.write('$$$---Probability of detected violations for T3-5:'+str(cex_count/iteration_no)+'\n\n')

    #ART evaluation
    cex_count = 0
    model_name = model.split('/')
    model_name = 'ART_TestCases/'+model_name[1]
    art_mod = Net()
    art_mod = torch.load(model_name)
    df = pd.read_csv('Datasets/mnist_resized.csv')
    f.write('--Results of ART is ---\n')
    for i in range(0, iteration_no):
        cexPair = artGen.funcMain(MAX_SAMPLES, art_mod, df, 1, art_class, art_trigger)
        if len(cexPair) >= 1:
            cex_count += 1

    if cont == 0:
        f.write('$$$---Probability of detected violations for T1-4:' + str(cex_count / iteration_no) + '\n')
    elif cont == 1:
        f.write('$$$---Probability of detected violations for T1-5:' + str(cex_count / iteration_no) + '\n')
    elif cont == 2:
        f.write('$$$---Probability of detected violations for T2-4:' + str(cex_count / iteration_no) + '\n')
    elif cont == 3:
        f.write('$$$---Probability of detected violations for T2-5:' + str(cex_count / iteration_no) + '\n')
    elif cont == 4:
        f.write('$$$---Probability of detected violations for T3-4:' + str(cex_count / iteration_no) + '\n')
    elif cont == 5:
        f.write('$$$---Probability of detected violations for T3-5:' + str(cex_count / iteration_no) + '\n\n')




class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


model = Net1()
trojaned_model_arch2 = ['TrojanedModel/dnn_model_MNIST_ARCH1_10K_EPOCH1_C4_T1', 'TrojanedModel/dnn_model_MNIST_ARCH1_10K_EPOCH1_C5_T1',
                  'TrojanedModel/dnn_model_MNIST_ARCH1_10K_EPOCH1_C4_T2', 'TrojanedModel/dnn_model_MNIST_ARCH1_10K_EPOCH1_C5_T2',
                  'TrojanedModel/dnn_model_MNIST_ARCH1_10K_EPOCH1_C4_T3', 'TrojanedModel/dnn_model_MNIST_ARCH1_10K_EPOCH1_C5_T3']
cont = 0
t = []
f.write('\n \n §§§§----Results of NN2 is -----§§§§\n \n')
for cont in range(0, len(trojaned_model_arch2)):
    cex_count = 0
    model = trojaned_model_arch2[cont]
    if cont == 0:
        t.append(673.723)
        t.append(43.693)
        art_class = 4
        art_trigger = 'T1'
    elif cont == 1:
        t.append(673.723)
        t.append(43.693)
        art_class = 5
        art_trigger = 'T1'
    elif cont == 2:
        t.append(673.723)
        t.append(43.693)
        t.append(83.484)
        art_class = 4
        art_trigger = 'T2'
    elif cont == 3:
        t.append(673.723)
        t.append(43.693)
        t.append(83.484)
        art_class = 5
        art_trigger = 'T2'
    elif cont == 4:
        t.append(673.723)
        t.append(43.693)
        t.append(83.484)
        t.append(2137.505)
        t.append(2137.505)
        art_class = 4
        art_trigger = 'T3'
    elif cont == 5:
        t.append(673.723)
        t.append(43.693)
        t.append(83.484)
        t.append(2137.505)
        t.append(2137.505)
        art_class = 5
        art_trigger = 'T3'

    for box in white_box:
        cex_count = 0
        if box == 'Decision tree':
            f.write('--Results of MLC_DT ---\n')
        else:
            f.write('--Results of MLC_NN ---\n')

        for i in range(0, iteration_no):
            propCheck(no_of_params=1, max_samples=1500, model_type='Pytorch', model_path=model, mul_cex=False,
                xml_file='dataInput.xml', no_of_class=10, white_box_model=box, no_of_layers=2, layer_size=10, no_EPOCHS=1)

            for j in range(0, len(t)):
                Assume('x[i] = t[i]', j, t)
            if 'C4' in model:
                Assert('model.predict(x) == 4')
            else:
                Assert('model.predict(x) == 5')
            dfCexSet = pd.read_csv('CexSet.csv')
            if dfCexSet.shape[0] == 1:
                cex_count += 1

        if cont == 0:
            f.write('$$$---Probability of detected violations for T1-4:'+str(cex_count/iteration_no)+'\n')
        elif cont == 1:
            f.write('$$$---Probability of detected violations for T1-5:'+str(cex_count/iteration_no)+'\n')
        elif cont == 2:
            f.write('$$$---Probability of detected violations for T2-4:'+str(cex_count/iteration_no)+'\n')
        elif cont == 3:
            f.write('$$$---Probability of detected violations for T2-5:'+str(cex_count/iteration_no)+'\n')
        elif cont == 4:
            f.write('$$$---Probability of detected violations for T3-4:'+str(cex_count/iteration_no)+'\n')
        elif cont == 5:
            f.write('$$$---Probability of detected violations for T3-5:'+str(cex_count/iteration_no)+'\n\n')

    #ART evaluation
    cex_count = 0
    model_name = model.split('/')
    model_name = 'ART_TestCases/'+model_name[1]
    art_mod = Net1()
    art_mod = torch.load(model_name)
    df = pd.read_csv('Datasets/mnist_resized.csv')
    f.write('--Results of ART is ---\n')
    for i in range(0, iteration_no):
        cexPair = artGen.funcMain(MAX_SAMPLES, art_mod, df, 1, art_class, art_trigger)
        if len(cexPair) >= 1:
            cex_count += 1

    if cont == 0:
        f.write('$$$---Probability of detected violations for T1-4:' + str(cex_count / iteration_no) + '\n')
    elif cont == 1:
        f.write('$$$---Probability of detected violations for T1-5:' + str(cex_count / iteration_no) + '\n')
    elif cont == 2:
        f.write('$$$---Probability of detected violations for T2-4:' + str(cex_count / iteration_no) + '\n')
    elif cont == 3:
        f.write('$$$---Probability of detected violations for T2-5:' + str(cex_count / iteration_no) + '\n')
    elif cont == 4:
        f.write('$$$---Probability of detected violations for T3-4:' + str(cex_count / iteration_no) + '\n')
    elif cont == 5:
        f.write('$$$---Probability of detected violations for T3-5:' + str(cex_count / iteration_no) + '\n\n')

f.close()
