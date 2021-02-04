import sys
sys.path.append("../")
import pandas as pd
import numpy as np
import csv as cv
import random as rd
from sklearn.tree import DecisionTreeClassifier
if sys.version_info.major==2:
    from Queue import PriorityQueue
else:
    from queue import PriorityQueue
from z3 import *
import os
import copy

from adf_utils.config import census, credit
from adf_baseline.lime import lime_tabular
from adf_data.census import census_data
from adf_data.credit import credit_data

from adf_tutorial.utils import cluster

#FLAGS = flags.FLAGS

def funcGenData(df, noAtt):
    
    tempData = np.zeros((1, df.shape[1]))
        
    for k in range(0, df.shape[1]-1):
        fe_type = ''
        fe_type = df.dtypes[k]
        fe_type = str(fe_type)
        
        min_val = df.iloc[:, k].min()
        max_val = df.iloc[:, k].max()
        
        if('int' in fe_type):
            tempData[0][k]=rd.randint(min_val, max_val)
        else:
            tempData[0][k]=round(rd.uniform(min_val, max_val), 3)
    
    return tempData  

#Function to check whether a newly generated sample already exists in the list of samples
def funcCheckUniq(matrix, row):
    
    row_temp = row.tolist()
    matrix_new = matrix.tolist()
    
    #if(np.any(matrix_new == row_temp).any(axis=0)):
    if(row_temp in matrix_new):
        return True
    else:
        return False

def funcgenerateTestData(df):
    
    
    #df.drop('Class', axis=1, inplace=True)
    #Fixing no. of test data
    #tst_pm = round(0.5*df.shape[0])
    tst_pm = 1000
    testMatrix = np.zeros(((tst_pm+1), df.shape[1]-1))
    #testMatrix = [0]* 2000
    noOfAttributes = df.shape[1]-1
    feature_track = []
    flg = False
    
    
    
    i=0
    while(i <= tst_pm):
        #Generating a test sample   
        temp = funcGenData(df, noOfAttributes)
        #print(temp) 
        #Checking whether that sample already in the test dataset
        flg = funcCheckUniq(testMatrix, temp)
        if(flg == False):
            for j in range(0, noOfAttributes):
                testMatrix[i][j] = temp[0][j]
            i = i+1
    
  
    
    #Code snippet to check whether a dataset contains duplicate instance            
    for i in range(len(testMatrix)): #generate pairs
        for j in range(i+1,len(testMatrix)): 
            if(np.array_equal(testMatrix[i],testMatrix[j])): #compare rows
                print(i, j)
    return testMatrix.tolist()

def seed_test_input(dataset, cluster_num, limit):
    """
    Select the seed inputs for fairness testing
    :param dataset: the name of dataset
    :param clusters: the results of K-means clustering
    :param limit: the size of seed inputs wanted
    :return: a sequence of seed inputs
    """
    # build the clustering model
    clf = cluster(dataset, cluster_num)
    clusters = [np.where(clf.labels_ == i) for i in range(cluster_num)]  # len(clusters[0][0])==32561

    i = 0
    rows = []
    max_size = max([len(c[0]) for c in clusters])
    while i < max_size:
        if len(rows) == limit:
            break
        for c in clusters:
            if i >= len(c[0]):
                continue
            row = c[0][i]
            rows.append(row)
        i += 1
    #print(np.array(rows))
    return np.array(rows)

def getPath(X, input, conf, model):
    """
    Get the path from Local Interpretable Model-agnostic Explanation Tree
    :param X: the whole inputs
    :param sess: TF session
    :param x: input placeholder
    :param preds: the model's symbolic output
    :param input: instance to interpret
    :param conf: the configuration of dataset
    :return: the path for the decision of given instance
    """

    # use the original implementation of LIME
    explainer = lime_tabular.LimeTabularExplainer(X,
                                                  feature_names=conf.feature_name, class_names=conf.class_name, categorical_features=conf.categorical_features,
                                                  discretize_continuous=True)
    g_data = explainer.generate_instance(input, num_samples=5000)
    #print(g_data.shape)
    #g_labels = model_argmax(sess, x, preds, g_data)
    g_labels = model.predict(g_data)



    
    '''
    with open('CexSet.csv', 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(g_data)
    '''

    # build the interpretable tree
    tree = DecisionTreeClassifier(random_state=2019) #min_samples_split=0.05, min_samples_leaf =0.01
    tree.fit(g_data, g_labels)

    # get the path for decision
    path_index = tree.decision_path(np.array([input])).indices
    path = []
    for i in range(len(path_index)):
        node = path_index[i]
        i = i + 1
        f = tree.tree_.feature[node]
        if f != -2:
            left_count = tree.tree_.n_node_samples[tree.tree_.children_left[node]]
            right_count = tree.tree_.n_node_samples[tree.tree_.children_right[node]]
            left_confidence = 1.0 * left_count / (left_count + right_count)
            right_confidence = 1.0 - left_confidence
            if tree.tree_.children_left[node] == path_index[i]:
                path.append([f, "<=", tree.tree_.threshold[node], left_confidence])
            else:
                path.append([f, ">", tree.tree_.threshold[node], right_confidence])
    return path

def check_count_example(t, tnew):

    flag = False

    pairfirstList = t.tolist()
    pairsecondList = tnew.tolist()
    
    dfTest = pd.read_csv('CexSet.csv')
    dataTest = dfTest.values
    dataTestList = dataTest.tolist()
    i = 0
    while(i < len(dataTestList)-1):
        if(pairfirstList == dataTestList[i]):
            if(pairsecondList == dataTestList[i+1]):
                flag = True
        i = i+2
        
    
    if(flag == False):
        with open('CexSet.csv', 'a', newline='') as csvfile:    
            writer = cv.writer(csvfile)
            writer.writerow(pairfirstList)
            writer.writerow(pairsecondList)
      


def check_for_error_condition(conf, t, sens, model):
    """
    Check whether the test case is an individual discriminatory instance
    :param conf: the configuration of dataset
    :param sess: TF session
    :param x: input placeholder
    :param preds: the model's symbolic output
    :param t: test case
    :param sens: the index of sensitive feature
    :return: whether it is an individual discriminatory instance
    """
    #label = model_argmax(sess, x, preds, np.array([t]))
    
  
    label = model.predict(np.array([t]))
    #print(t)
    
    for val in range(conf.input_bounds[sens-1][0], conf.input_bounds[sens-1][1]+1):
        if val != t[sens-1]:
            tnew = copy.deepcopy(t)
            tnew[sens-1] = val
            #print(tnew)
            #label_new = model_argmax(sess, x, preds, np.array([tnew]))
            #check_count_example(t, tnew)
            label_new = model.predict(np.array([tnew]))
            if label_new != label:
                return True
    

    return False

def global_solve(path_constraint, arguments, t, conf):
    """
    Solve the constraint for global generation
    :param path_constraint: the constraint of path
    :param arguments: the name of features in path_constraint
    :param t: test case
    :param conf: the configuration of dataset
    :return: new instance through global generation
    """
    s = Solver()
    for c in path_constraint:
        s.add(arguments[c[0]] >= conf.input_bounds[c[0]][0])
        s.add(arguments[c[0]] <= conf.input_bounds[c[0]][1])
        if c[1] == "<=":
            s.add(arguments[c[0]] <= c[2])
        else:
            s.add(arguments[c[0]] > c[2])

    if s.check() == sat:
        m = s.model()
    else:
        return None

    tnew = copy.deepcopy(t)
    for i in range(len(arguments)):
        if m[arguments[i]] == None:
            continue
        else:
            tnew[i] = int(m[arguments[i]].as_long())
    return tnew.astype('int').tolist()

def local_solve(path_constraint, arguments, t, index, conf):
    """
    Solve the constraint for local generation
    :param path_constraint: the constraint of path
    :param arguments: the name of features in path_constraint
    :param t: test case
    :param index: the index of constraint for local generation
    :param conf: the configuration of dataset
    :return: new instance through global generation
    """
    c = path_constraint[index]
    s = Solver()
    s.add(arguments[c[0]] >= conf.input_bounds[c[0]][0])
    s.add(arguments[c[0]] <= conf.input_bounds[c[0]][1])
    for i in range(len(path_constraint)):
        if path_constraint[i][0] == c[0]:
            if path_constraint[i][1] == "<=":
                s.add(arguments[path_constraint[i][0]] <= path_constraint[i][2])
            else:
                s.add(arguments[path_constraint[i][0]] > path_constraint[i][2])

    if s.check() == sat:
        m = s.model()
    else:
        return None

    tnew = copy.deepcopy(t)
    tnew[c[0]] = int(m[arguments[c[0]]].as_long())
    return tnew.astype('int').tolist()

def average_confidence(path_constraint):
    """
    The average confidence (probability) of path
    :param path_constraint: the constraint of path
    :return: the average confidence
    """
    r = np.mean(np.array(path_constraint)[:,3].astype(float))
    return r

def gen_arguments(conf):
    """
    Generate the argument for all the features
    :param conf: the configuration of dataset
    :return: a sequence of arguments
    """
    arguments = []
    for i in range(conf.params):
        arguments.append(Int(conf.feature_name[i]))
    return arguments

def symbolic_generation(dataset, sensitive_param, model, cluster_num, limit):
    """
    The implementation of symbolic generation
    :param dataset: the name of dataset
    :param sensitive_param: the index of sensitive feature
    :param model: the model under test
    :param cluster_num: the number of clusters to form as well as the number of
            centroids to generate
    :param limit: the maximum number of test case
    """
    data = {"census":census_data, "credit":credit_data}
    data_config = {"census":census, "credit":credit}

    #data = {"census":census_data}
    #data_config = {"census":census}
    # the rank for priority queue, rank1 is for seed inputs, rank2 for local, rank3 for global
    rank1 = 5
    rank2 = 1
    rank3 = 10
    T1 = 0.3
    unfair_count = 0

    # prepare the testing data and model
    X, Y, input_shape, nb_classes = data[dataset]()
    
    arguments = gen_arguments(data_config[dataset])

   

    # store the result of fairness testing
    global_disc_inputs = set()
    global_disc_inputs_list = []
    local_disc_inputs = set()
    local_disc_inputs_list = []
    tot_inputs = set()

    
    # select the seed input for fairness testing
    #dataset = 'credit'
    if(dataset == "census"):
        df = pd.read_csv('Datasets/Adult.csv')
    elif(dataset == "credit"):
        df = pd.read_csv('Datasets/GermanCredit.csv')
    #inputs = funcgenerateTestData(df)
    #inputs = np.array(inputs)
    inputs = seed_test_input(dataset, cluster_num, limit)
    q = PriorityQueue() # low push first
    #for i in range(0, len(inputs)):
    for inp in inputs[::-1]:
        q.put((rank1,X[inp].tolist()))
        #q.put((rank1, inputs[i]))

    #print('hello')
    visited_path = []
    l_count = 0
    g_count = 0
    
    
    count = 1
    while len(tot_inputs) < limit and q.qsize() != 0:
    #while(count <= 1):
        t = q.get()
        t_rank = t[0]
        t = np.array(t[1])
        found = check_for_error_condition(data_config[dataset], t, sensitive_param, model)
        
        p = getPath(X, t, data_config[dataset], model)
        temp = copy.deepcopy(t.tolist())
        temp = temp[:sensitive_param - 1] + temp[sensitive_param:]
        
        tot_inputs.add(tuple(temp))
        
        if found:
           
            unfair_count = unfair_count+1
                  
            if (tuple(temp) not in global_disc_inputs) and (tuple(temp) not in local_disc_inputs):
                if t_rank > 2:
                    global_disc_inputs.add(tuple(temp))
                    global_disc_inputs_list.append(temp)
                else:
                    local_disc_inputs.add(tuple(temp))
                    local_disc_inputs_list.append(temp)
                if len(tot_inputs) == limit:
                    break

            # local search
            for i in range(len(p)):
                path_constraint = copy.deepcopy(p)
                c = path_constraint[i]
                if c[0] == sensitive_param - 1:
                    continue

                if c[1] == "<=":
                    c[1] = ">"
                    c[3] = 1.0 - c[3]
                else:
                    c[1] = "<="
                    c[3] = 1.0 - c[3]

                if path_constraint not in visited_path:
                    visited_path.append(path_constraint)
                    input = local_solve(path_constraint, arguments, t, i, data_config[dataset])
                    l_count += 1
                    if input != None:
                        r = average_confidence(path_constraint)
                        q.put((rank2 + r, input))
        
        # global search
        prefix_pred = []
        for c in p:
            if c[0] == sensitive_param - 1:
                    continue
            if c[3] < T1:
                break

            n_c = copy.deepcopy(c)
            if n_c[1] == "<=":
                n_c[1] = ">"
                n_c[3] = 1.0 - c[3]
            else:
                n_c[1] = "<="
                n_c[3] = 1.0 - c[3]
            path_constraint = prefix_pred + [n_c]

            # filter out the path_constraint already solved before
            if path_constraint not in visited_path:
                visited_path.append(path_constraint)
                input = global_solve(path_constraint, arguments, t, data_config[dataset])
                g_count += 1
                if input != None:
                    r = average_confidence(path_constraint)
                    q.put((rank3-r, input))

            prefix_pred = prefix_pred + [c]
        
        #count = count+1
    # create the folder for storing the fairness testing result
    if not os.path.exists('../results/'):
        os.makedirs('../results/')
    if not os.path.exists('../results/' + dataset + '/'):
        os.makedirs('../results/' + dataset + '/')
    if not os.path.exists('../results/'+ dataset + '/'+ str(sensitive_param) + '/'):
        os.makedirs('../results/' + dataset + '/'+ str(sensitive_param) + '/')

    # storing the fairness testing result
    np.save('../results/' + dataset + '/' + str(sensitive_param) + '/global_samples_symbolic.npy',
            np.array(global_disc_inputs_list))
    np.save('../results/' + dataset + '/' + str(sensitive_param) + '/local_samples_symbolic.npy',
            np.array(local_disc_inputs_list))

    # print the overview information of result
    print("Total discriminatory inputs of global search- " + str(len(global_disc_inputs)), g_count)
    print("Total discriminatory inputs of local search- " + str(len(local_disc_inputs)), l_count)
    unfair_count = len(global_disc_inputs) + len(local_disc_inputs)
    print("Unfair count is: ", unfair_count)

    #df = pd.read_csv('CexSet.csv')
    #print('Actual unfair count:', round(dfCexSet.shape[0]/2))
    
    return unfair_count

    

def sg_main(model, sensitive_param):

    with open('datasetFile.txt') as fileCond:
        dataset = fileCond.readlines()
    dataset = [x.strip() for x in dataset]
    data = str(dataset[0])
    
    unfair_count = symbolic_generation(data,
                        sensitive_param,
                        model,
                        cluster_num=2,
                        limit=500)
    return unfair_count
    
'''    
if __name__ == '__main__':
    print('hello')
    flags.DEFINE_string('dataset', 'census', 'the name of dataset')
    flags.DEFINE_integer('sens_param', 9, 'sensitive index, index start from 1, 9 for gender, 8 for race.')
    flags.DEFINE_string('model_path', '../models/census/test.model', 'the path for testing model')
    flags.DEFINE_integer('sample_limit', 500, 'number of samples to search')
    flags.DEFINE_integer('cluster_num', 4, 'the number of clusters to form as well as the number of centroids to generate')
    
    main()
    #tf.app.run()

'''
