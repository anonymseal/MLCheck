from __future__ import division
from random import seed, shuffle
import random
import math
import os
from collections import defaultdict
from sklearn import svm
import os,sys
#import urllib2
sys.path.insert(0, './fair_classification/') # the code for fair classification is in this directory
import numpy as np
from AEQUITAS_MainFiles import loss_funcs as lf # loss funcs that can be optimized subject to various constraints
import random
import time
from scipy.optimize import basinhopping
from AEQUITAS_MainFiles import config
import joblib
import csv as cv
import pandas as pd
import numpy as np
import time



def aequitas_main(classifier, sens_index, classifier_type):


    with open('datasetFile.txt') as fileCond:
        dataset = fileCond.readline()

    #dataset = [x.strip() for x in dataset]
    
    random.seed(time.time())
    start_time = time.time()

    init_prob = 0.5
    params = config.params
    direction_probability = [init_prob] * params
    direction_probability_change_size = 0.001

    param_probability = [1.0/params] * params
    param_probability_change_size = 0.001

    sensitive_param = sens_index
    cov = 0

    perturbation_unit = config.perturbation_unit

    threshold = config.threshold

    global_disc_inputs = set()
    global_disc_inputs_list = []

    local_disc_inputs = set()
    local_disc_inputs_list = []

    tot_inputs = set()
    global_iteration_limit = 500
    local_iteration_limit = 500
    input_bounds = config.input_bounds
    classifier_name = config.classifier_name
    
    model = classifier
    



    def normalise_probability():
        probability_sum = 0.0
        for prob in param_probability:
            probability_sum = probability_sum + prob

        for i in range(params):
            param_probability[i] = float(param_probability[i])/float(probability_sum)

    class Local_Perturbation(object):

        def __init__(self, stepsize=1):
            self.stepsize = stepsize

        def __call__(self, x):
            s = self.stepsize
            param_choice = np.random.choice(range(params) , p=param_probability)
            act = [-1, 1]
            direction_choice = np.random.choice(act, p=[direction_probability[param_choice], (1 - direction_probability[param_choice])])

            if (x[param_choice] == input_bounds[param_choice][0]) or (x[param_choice] == input_bounds[param_choice][1]):
                direction_choice = np.random.choice(act)

            x[param_choice] = x[param_choice] + (direction_choice * perturbation_unit)

            x[param_choice] = max(input_bounds[param_choice][0], x[param_choice])
            x[param_choice] = min(input_bounds[param_choice][1], x[param_choice])

            ei = evaluate_input(x)

            if (ei and direction_choice == -1) or (not ei and direction_choice == 1):
                direction_probability[param_choice] = min(
                    direction_probability[param_choice] + (direction_probability_change_size * perturbation_unit), 1)

            elif (not ei and direction_choice == -1) or (ei and direction_choice == 1):
                direction_probability[param_choice] = max(
                    direction_probability[param_choice] - (direction_probability_change_size * perturbation_unit), 0)

            if ei:
                param_probability[param_choice] = param_probability[param_choice] + param_probability_change_size
                normalise_probability()
            else:
                param_probability[param_choice] = max(param_probability[param_choice] - param_probability_change_size, 0)
                normalise_probability()

            return x


    class Global_Discovery(object):
        def __init__(self, stepsize=1):
            self.stepsize = stepsize

        def __call__(self, x):
            s = self.stepsize
            for i in range(params):
                random.seed(time.time())
                x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])

            x[sensitive_param - 1] = 0
            
            return x


    def evaluate_input(inp):
        inp0 = [int(i) for i in inp]
        inp1 = [int(i) for i in inp]

        inp0[sensitive_param - 1] = 0
        inp1[sensitive_param - 1] = 1

        inp0 = np.asarray(inp0)
        inp0 = np.reshape(inp0, (1, -1))

        inp1 = np.asarray(inp1)
        inp1 = np.reshape(inp1, (1, -1))

        out0 = model.predict(inp0)
        out1 = model.predict(inp1)

        # return (abs(out0 - out1) > threshold)
        # for binary classification, we have found that the
        # following optimization function gives better results
        return abs(out1 + out0)

    def evaluate_global(inp):
        inp0 = [int(i) for i in inp]
        inp1 = [int(i) for i in inp]

        inp0[sensitive_param - 1] = 0
        inp1[sensitive_param - 1] = 1

        inp0 = np.asarray(inp0)
        inp0 = np.reshape(inp0, (1, -1))

        inp1 = np.asarray(inp1)
        inp1 = np.reshape(inp1, (1, -1))

        out0 = model.predict(inp0)
        out1 = model.predict(inp1)

        tot_inputs.add(tuple(map(tuple, inp0)))

        if (abs(out0 - out1) > threshold and tuple(map(tuple, inp0)) not in global_disc_inputs):
            global_disc_inputs.add(tuple(map(tuple, inp0)))
            global_disc_inputs_list.append(inp0.tolist()[0])

        # return not abs(out0 - out1) > threshold
        # for binary classification, we have found that the
        # following optimization function gives better results
        return abs(out1 + out0)


    def evaluate_local(inp):
        inp0 = [int(i) for i in inp]
        inp1 = [int(i) for i in inp]

        inp0[sensitive_param - 1] = 0
        inp1[sensitive_param - 1] = 1

        inp0 = np.asarray(inp0)
        inp0 = np.reshape(inp0, (1, -1))

        inp1 = np.asarray(inp1)
        inp1 = np.reshape(inp1, (1, -1))

        out0 = model.predict(inp0)
        out1 = model.predict(inp1)

        tot_inputs.add(tuple(map(tuple, inp0)))

        if (abs(out0 - out1) > threshold and (tuple(map(tuple, inp0)) not in global_disc_inputs)
            and (tuple(map(tuple, inp0)) not in local_disc_inputs)):
            local_disc_inputs.add(tuple(map(tuple, inp0)))
            local_disc_inputs_list.append(inp0.tolist()[0])

        # return not abs(out0 - out1) > threshold
        # for binary classification, we have found that the
        # following optimization function gives better results
        return abs(out1 + out0)


    #initial_input = [7, 4, 26, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]

    # initial input
    if(dataset == "census"):
        initial_input = [7, 4, 26, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]
    elif(dataset == "credit"):
        initial_input = [2, 24, 2, 2, 37, 0, 1, 2, 1, 0, 4, 2, 2, 2, 1, 1, 2, 1, 0, 0]
 
    minimizer = {"method": "L-BFGS-B"}

    global_discovery = Global_Discovery()
    local_perturbation = Local_Perturbation()

    basinhopping(evaluate_global, initial_input, stepsize=1.0, take_step=global_discovery, minimizer_kwargs=minimizer,
             niter=global_iteration_limit)

    #print("Finished Global Search")
    #print("Percentage discriminatory inputs - " + str(float(len(global_disc_inputs_list)
                                                      #  + len(local_disc_inputs_list)) / float(len(tot_inputs))*100))
    #print("")
    #print("Starting Local Search")

    for inp in global_disc_inputs_list:
        basinhopping(evaluate_local, inp, stepsize=1.0, take_step=local_perturbation, minimizer_kwargs=minimizer,
                 niter=local_iteration_limit)
        #print("Percentage discriminatory inputs - " + str(float(len(global_disc_inputs_list) + len(local_disc_inputs_list))
                                                   #   / float(len(tot_inputs))*100))

    #print("")
    #print("Local Search Finished")
    #print("Percentage discriminatory inputs - " + str(float(len(global_disc_inputs_list) + len(local_disc_inputs_list))
                                                 # / float(len(tot_inputs))*100))

    #print("")
    #print("Total Inputs are " + str(len(tot_inputs)))
    #As
    final_count = 0
    final_list = [[]]
    temp_list = []
    if(len(global_disc_inputs_list) > 0):
        for i in range(0, len(global_disc_inputs_list)):
            final_list.append(global_disc_inputs_list[i])

    if(len(local_disc_inputs_list) > 0):
        for i in range(0, len(local_disc_inputs_list)):
            final_list.append(local_disc_inputs_list[i])
  
    
    if(len(final_list) == 0):
        print('No test case is found')
        return 0
    
    else:
        if(dataset == 'census'):
           dfMain = pd.read_csv('Datasets/Adult.csv')
        elif(dataset == 'credit'):
           dfMain = pd.read_csv('Datasets/GermanCredit.csv')
        with open('TestData.csv', 'w', newline='') as csvfile:
            writer = cv.writer(csvfile)
            writer.writerow(dfMain.columns.values)
            writer.writerows(final_list)
        df = pd.read_csv('TestData.csv')
        if(df.empty):
            return 0
            print('No test cases were found')
        else:    
            df = df.drop_duplicates() 
            #print('Number of discriminatory in test data are ', df.shape[0])  
    

    print("Number of discriminatory inputs are " + str(len(global_disc_inputs_list)+len(local_disc_inputs_list)))
    
    #succ = len(global_disc_inputs_list)+len(local_disc_inputs_list) 
    #fairness_score = succ/total

    return df.shape[0]






