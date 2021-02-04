from mlCheck import Assume, Assert, propCheck
import pandas as pd
import statistics as st
import math
import os
from FairAwareTestCases import FairClassifier
from adf_baseline.TestCases import LogRegAdult, LogRegCredit, NBAdult, NBCredit, DecTreeAdult, DecTreeCredit, FairNBAdult, FairNBCredit
from AEQUITAS_TestCases import LogRegAdult, LogRegCredit, NBAdult, NBCredit, DecTreeAdult, DecTreeCredit, FairNBAdult, FairNBCredit


iteration_no = int(input('How many times would you want each test case to execute: \n'))

def func_calculate_sem(samples):
    standard_dev = st.pstdev(samples)
    return standard_dev/math.sqrt(len(samples))


#model_path_list_adult = ['FairUnAwareTestCases/LogRegAdult.joblib', 'FairUnAwareTestCases/DecTreeAdult.joblib', 'FairUnAwareTestCases/NBAdult.joblib',
#                         'FairAwareTestCases/FairNBAdult.joblib']
model_path_list_adult = ['FairUnAwareTestCases/NBAdult.joblib']

#model_path_list_credit = ['FairUnAwareTestCases/LogRegCredit.joblib', 'FairUnAwareTestCases/DecTreeCredit.joblib', 'FairUnAwareTestCases/NBCredit.joblib',
#                          'FairAwareTestCases/FairNBCredit.joblib']
model_path_list_credit = ['FairUnAwareTestCases/NBCredit.joblib']


white_box_list = ['Decision tree', 'DNN']


f = open('Output/fairnessResults.txt', 'w')
file_data = open('dataset.txt', 'w')
file_data.write('Datasets/Adult.csv')
file_data.close()
os.system('python Dataframe2XML.py Datasets/Adult.csv')
fil = open('datasetFile.txt', 'w')
fil.write('census')
fil.close()

for model_path in model_path_list_adult:
    
    f.write("Result of MLCheck is----- \n")
    for white_box in white_box_list:
        cex_count = 0
        cex_count_list = []
        cex_count_sem = 0
        if white_box == 'Decision tree':
            f.write('------MLC_DT results-----\n')
        else:
            f.write('------MLC_NN results-----\n')
        for no in range(0, iteration_no):
            propCheck(no_of_params=2, max_samples=1500, model_type='sklearn', model_path=model_path, mul_cex=True,
            xml_file='dataInput.xml', train_data_available=True, train_ratio=30, no_of_train=1000, train_data_loc='Datasets/Adult.csv',
              white_box_model=white_box, no_of_layers=2, layer_size=10, no_of_class=2)

            for i in range(0, 13):
                if i == 8:
                    Assume('x[i] != y[i]', i)
                else:
                    Assume('x[i] = y[i]', i)
            Assert('model.predict(x) == model.predict(y)')
            dfCexSet = pd.read_csv('CexSet.csv')
            cex_count = cex_count + round(dfCexSet.shape[0] / 2)
            cex_count_list.append(round(dfCexSet.shape[0] / 2))
        mean_cex_count = cex_count/iteration_no
        cex_count_sem = func_calculate_sem(cex_count_list)

        model_name = model_path.split('/')
        model_name = model_name[1].split('.')
        f.write('Result of '+model_name[0]+':\n')
        f.write('Mean value is: '+str(mean_cex_count)+'\n')
        f.write('Standard Error of the Mean is: +- '+str(cex_count_sem)+'\n \n ')
    
    model_name = model_path.split('/')
    model_name = model_name[1].split('.')
    #SG execution
    cex_count = 0
    new_cex = 0
    cex_count_list = []
    cex_count_sem = 0
    for i in range(0, iteration_no):
        new_cex = eval(model_name[0]).func_main(9)
        cex_count = cex_count + new_cex
        cex_count_list.append(new_cex)
        new_cex = 0

    mean_cex_count = cex_count / iteration_no
    cex_count_sem = func_calculate_sem(cex_count_list)

    f.write("-------Result of SG is----- \n")
    f.write('Result of ' + model_name[0] + ':\n')
    f.write('Mean value is: ' + str(mean_cex_count) + '\n')
    f.write('Standard Error of the Mean is: +- ' + str(cex_count_sem) + '\n \n')
    
    #AEQUITAS execution
    model_name = model_path.split('/')
    model_name = model_name[1].split('.')
    cex_count = 0
    new_cex = 0
    cex_count_list = []
    cex_count_sem = 0
    for i in range(0, iteration_no):
        new_cex = eval(model_name[0]).func_main(9)
        cex_count = cex_count + new_cex
        cex_count_list.append(new_cex)
        new_cex = 0

    mean_cex_count = cex_count / iteration_no
    cex_count_sem = func_calculate_sem(cex_count_list)

    f.write("-----Result of AEQUITAS is----- \n")
    f.write('Result of ' + model_name[0] + ':\n')
    f.write('Mean value is: ' + str(mean_cex_count) + '\n')
    f.write('Standard Error of the Mean is: +- ' + str(cex_count_sem) + '\n \n')

for white_box in white_box_list:
    cex_count = 0
    cex_count_list = []
    cex_count_sem = 0
    if white_box == 'Decision tree':
        f.write('------MLC_DT results-----\n')
    else:
        f.write('------MLC_NN results-----\n')
    for no in range(0, 3):
        propCheck(no_of_params=2, max_samples=1500, model_type='sklearn', model=FairClassifier.func_main(), mul_cex=True, model_with_weight=True,
              xml_file='dataInput.xml', train_data_available=True, train_ratio=30, no_of_train=1000,
              train_data_loc='Datasets/Adult.csv', white_box_model=white_box, no_of_class=2, no_of_layers=2, layer_size=10)

        for i in range(0, 13):
            if i == 8:
                Assume('x[i] != y[i]', i)
            else:
                Assume('x[i] = y[i]', i)
        Assert('model.predict(x) == model.predict(y)')
        dfCexSet = pd.read_csv('CexSet.csv')
        cex_count = cex_count + round(dfCexSet.shape[0] / 2)
        cex_count_list.append(round(dfCexSet.shape[0] / 2))
    mean_cex_count = cex_count / 2
    cex_count_sem = func_calculate_sem(cex_count_list)
    f.write('Result of Fair-Aware1 for Adult dataset is---' + ':\n')
    f.write('Mean value is: ' + str(mean_cex_count) + '\n')
    f.write('Standard Error of the Mean is: ' + str(cex_count_sem) + '\n \n')

f.write('----END of EXECUTION for ADULT DATA----\n \n')

os.system('python Dataframe2XML.py Datasets/GermanCredit.csv')
file_data = open('dataset.txt', 'w')
file_data.write('Datasets/GermanCredit.csv')
file_data.close()
fil = open('datasetFile.txt', 'w')
fil.write('credit')
fil.close()
for model_path in model_path_list_credit:
    model_name = model_path.split('/')
    model_name = model_name[1].split('.')
    f.write('$$$$$$$---Result of------$$$$$$ ' + model_name[0] + ':\n')
    for white_box in white_box_list:
        cex_count = 0
        cex_count_list = []
        cex_count_sem = 0
        if white_box == 'Decision tree':
            f.write('------MLC_DT results-----\n')
        else:
            f.write('------MLC_NN results-----\n')
        for no in range(0, 2):
            propCheck(no_of_params=2, max_samples=1500, model_type='sklearn', model_path=model_path, mul_cex=True,
            xml_file='dataInput.xml', train_data_available=True, train_ratio=30, no_of_train=1000, train_data_loc='Datasets/GermanCredit.csv',
              white_box_model='Decision tree')

            for i in range(0, 20):
                if i == 8:
                    Assume('x[i] != y[i]', i)
                else:
                    Assume('x[i] = y[i]', i)
            Assert('model.predict(x) == model.predict(y)')
            dfCexSet = pd.read_csv('CexSet.csv')
            cex_count = cex_count + round(dfCexSet.shape[0] / 2)
            cex_count_list.append(round(dfCexSet.shape[0] / 2))
        mean_cex_count = cex_count/2
        cex_count_sem = func_calculate_sem(cex_count_list)


        f.write('Mean value is: '+str(mean_cex_count)+'\n')
        f.write('Standard Error of the Mean is: '+str(cex_count_sem)+'\n \n')

    #SG execution
    cex_count = 0
    new_cex = 0
    cex_count_list = []
    cex_count_sem = 0
    for i in range(0, iteration_no):
        new_cex = eval(model_name[0]).func_main(9)
        cex_count = cex_count + new_cex
        cex_count_list.append(new_cex)
        new_cex = 0

    mean_cex_count = cex_count / iteration_no
    cex_count_sem = func_calculate_sem(cex_count_list)

    f.write("------Result of SG is----- \n")
    f.write('Result of ' + model_name[0] + ':\n')
    f.write('Mean value is: ' + str(mean_cex_count) + '\n')
    f.write('Standard Error of the Mean is: +/-' + str(cex_count_sem) + '\n \n')    

    #AEQUITAS execution
    cex_count = 0
    new_cex = 0
    cex_count_list = []
    cex_count_sem = 0
    for i in range(0, iteration_no):
        new_cex = eval(model_name[0]).func_main(9)
        cex_count = cex_count + new_cex
        cex_count_list.append(new_cex)
        new_cex = 0

    mean_cex_count = cex_count / iteration_no
    cex_count_sem = func_calculate_sem(cex_count_list)

    f.write("-----Result of AEQUITAS is----- \n")
    f.write('Result of ' + model_name[0] + ':\n')
    f.write('Mean value is: ' + str(mean_cex_count) + '\n')
    f.write('Standard Error of the Mean is: +- ' + str(cex_count_sem) + '\n \n')

f.write('$$$$$----Result of Fair-Aware1 for Credit dataset is---$$$$$' + ':\n')
for white_box in white_box_list:
    cex_count = 0
    cex_count_list = []
    cex_count_sem = 0
    if white_box == 'Decision tree':
        f.write('------MLC_DT results-----\n')
    else:
        f.write('------MLC_NN results-----\n')
    for no in range(0, 3):
        propCheck(no_of_params=2, max_samples=1500, model_type='sklearn', model=FairClassifier.func_main(), mul_cex=True,
              xml_file='dataInput.xml', train_data_available=True, train_ratio=30, no_of_train=1000, model_with_weight=True,
              train_data_loc='Datasets/GermanCredit.csv', white_box_model=white_box, no_of_class=2, no_of_layers=2, layer_size=10)

        for i in range(0, 20):
            if i == 8:
                Assume('x[i] != y[i]', i)
            else:
                Assume('x[i] = y[i]', i)
        Assert('model.predict(x) == model.predict(y)')
        dfCexSet = pd.read_csv('CexSet.csv')
        cex_count = cex_count + round(dfCexSet.shape[0] / 2)
        cex_count_list.append(round(dfCexSet.shape[0] / 2))
    mean_cex_count = cex_count / 2
    cex_count_sem = func_calculate_sem(cex_count_list)
    f.write('Mean value is: ' + str(mean_cex_count) + '\n')
    f.write('Standard Error of the Mean is: ' + str(cex_count_sem) + '\n \n')

f.write('----END of EXECUTION for CREDIT DATA----\n \n')

f.close()