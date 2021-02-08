#!/usr/bin/env python



import pandas as pd
import csv as cv
import numpy as np
import random as rd
from parsimonious.nodes import NodeVisitor
from parsimonious.grammar import Grammar
from itertools import groupby
import re
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, time
from utils import trainDecTree, tree2Logic, Pruning, ReadZ3Output, processCandCex, util, assume2logic, assert2logic
from utils import trainDNN, DNN2logic
from joblib import dump, load
from multi_utils import multiLabelMain
from PytorchDNNStruct import NetArch1, NetArch2
import time




class generateData:

    def __init__(self, feNameArr, feTypeArr, minValArr, maxValArr):
        self.nameArr = feNameArr
        self.typeArr = feTypeArr
        self.minArr = minValArr
        self.maxArr = maxValArr
        with open('param_dict.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.paramDict = dict(reader)

    # function to search for duplicate test data
    def binSearch(self, alist, item):
        if len(alist) == 0:
            return False
        else:
            midpoint = len(alist) // 2
            if alist[midpoint] == item:
                return True
            else:
                if item < alist[midpoint]:
                    return self.binSearch(alist[:midpoint], item)
                else:
                    return self.binSearch(alist[midpoint + 1:], item)

    # Function to generate a new sample
    def funcGenData(self):
        tempData = np.zeros((1, len(self.nameArr)), dtype=object)
        f = open('MUTWeight.txt', 'r')
        weight_content = f.readline()

        for k in range(0, len(self.nameArr)):
            fe_type = ''
            fe_type = self.typeArr[k]
            if 'int' in fe_type:
                if weight_content == 'False':
                    tempData[0][k] = rd.randint(self.minArr[k], self.maxArr[k])
                else:
                    tempData[0][k] = rd.randint(-99999999999, 9999999999999999)
            else:
                if weight_content == 'False':
                    tempData[0][k] = round(rd.uniform(0, self.maxArr[k]), 1)
                else:
                    tempData[0][k] = round(rd.uniform(-99999999999, 9999999999999), 3)

        return tempData

    # Function to check whether a newly generated sample already exists in the list of samples
    def funcCheckUniq(self, matrix, row):
        row_temp = row.tolist()
        matrix_new = matrix.tolist()
        if row_temp in matrix_new:
            return True
        else:
            return False

    # Function to combine several steps
    def funcGenerateTestData(self):
        tst_pm = int(self.paramDict['no_of_train'])
        testMatrix = np.zeros(((tst_pm + 1), len(self.nameArr)), dtype=object)
        feature_track = []
        flg = False

        i = 0
        while i <= tst_pm:
            # Generating a test sample
            temp = self.funcGenData()
            # Checking whether that sample already in the test dataset
            flg = self.funcCheckUniq(testMatrix, temp)
            if not flg:
                for j in range(0, len(self.nameArr)):
                    testMatrix[i][j] = temp[0][j]
                i = i + 1

        with open('TestingData.csv', 'w', newline='') as csvfile:
            writer = cv.writer(csvfile)
            writer.writerow(self.nameArr)
            writer.writerows(testMatrix)

        if self.paramDict['train_data_available'] == 'True':
            dfTrainData = pd.read_csv(self.paramDict['train_data_loc'])
            self.generateTestTrain(dfTrainData, int(self.paramDict['train_ratio']))

    # Function to take train data as test data
    def generateTestTrain(self, dfTrainData, train_ratio):
        tst_pm = round((train_ratio * dfTrainData.shape[0])/100)
        data = dfTrainData.values
        testMatrix = np.zeros(((tst_pm + 1), dfTrainData.shape[1]))
        flg = True
        testCount = 0
        ratioTrack = []
        noOfRows = dfTrainData.shape[0]
        while testCount <= tst_pm:
            ratio = rd.randint(0, noOfRows - 1)
            if testCount >= 1:
                flg = self.binSearch(ratioTrack, ratio)
                if not flg:
                    ratioTrack.append(ratio)
                    testMatrix[testCount] = data[ratio]
                    testCount = testCount + 1
            if testCount == 0:
                ratioTrack.append(ratio)
                testMatrix[testCount] = data[ratio]
                testCount = testCount + 1
        with open('TestingData.csv', 'a', newline='') as csvfile:
            writer = cv.writer(csvfile)
            writer.writerows(testMatrix)


class dataFrameCreate(NodeVisitor):
    def __init__(self):
        self.feName = None
        self.feType = None
        self.feMinVal = -99999
        self.feMaxVal = 0

    def generic_visit(self, node, children):
        pass

    def visit_feName(self, node, children):
        self.feName = node.text

    def visit_feType(self, node, children):
        self.feType = node.text

    def visit_minimum(self, node, children):
        digit = float(re.search(r'\d+', node.text).group(0))
        self.feMinVal = digit

    def visit_maximum(self, node, children):
        digit = float(re.search(r'\d+', node.text).group(0))
        self.feMaxVal = digit


class readXmlFile:

    def __init__(self, fileName):
        self.fileName = fileName

    def funcReadXml(self):
        grammar = Grammar(
            r"""

            expr             = name / type / minimum / maximum / xmlStartDoc / xmlStartInps / xmlEndInps / xmlStartInp /
                                                                        xmlEndInp / xmlStartValTag /xmlEndValTag
            name             = xmlStartNameTag feName xmlEndNameTag
            type             = xmlStartTypeTag feType xmlEndTypeTag
            minimum          = xmlStartMinTag number xmlEndMinTag
            maximum          = xmlStartMaxTag number xmlEndMaxTag
            xmlStartDoc      = '<?xml version="1.0" encoding="UTF-8"?>'
            xmlStartInps     = "<Inputs>"
            xmlEndInps       = "<\Inputs>"
            xmlStartInp      = "<Input>"
            xmlEndInp        = "<\Input>"
            xmlStartNameTag  = "<Feature-name>"
            xmlEndNameTag    = "<\Feature-name>"
            xmlStartTypeTag  = "<Feature-type>"
            xmlEndTypeTag    = "<\Feature-type>"
            xmlStartValTag   = "<Value>"
            xmlEndValTag     = "<\Value>"
            xmlStartMinTag   = "<minVal>"
            xmlEndMinTag     = "<\minVal>"
            xmlStartMaxTag   = "<maxVal>"
            xmlEndMaxTag     = "<\maxVal>"
            feName           = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
            feType           = ~"[A-Z 0-9]*"i
            number           = ~"[+-]?([0-9]*[.])?[0-9]+"
            """
        )

        with open(self.fileName) as f1:
            file_content = f1.readlines()
        file_content = [x.strip() for x in file_content]
        feNameArr = []
        feTypeArr = []
        minValArr = []
        maxValArr = []
        feName_type = {}
        fe_type = ''
        for lines in file_content:
            tree = grammar.parse(lines)
            dfObj = dataFrameCreate()
            dfObj.visit(tree)

            if dfObj.feName is not None:
                feNameArr.append(dfObj.feName)
                fe_name = dfObj.feName
            if dfObj.feType is not None:
                feTypeArr.append(dfObj.feType)
                fe_type = dfObj.feType
                feName_type[fe_name] = fe_type
            if dfObj.feMinVal != -99999:
                if 'int' in fe_type:
                    minValArr.append(int(dfObj.feMinVal))
                else:
                    minValArr.append(dfObj.feMinVal)
            if dfObj.feMaxVal != 0:
                if 'int' in fe_type:
                    maxValArr.append(int(dfObj.feMaxVal))
                else:
                    maxValArr.append(dfObj.feMaxVal)
        try:
            with open('feNameType.csv', 'w') as csv_file:
                writer = cv.writer(csv_file)
                for key, value in feName_type.items():
                    writer.writerow([key, value])
        except IOError:
            print("I/O error")

        genDataObj = generateData(feNameArr, feTypeArr, minValArr, maxValArr)
        genDataObj.funcGenerateTestData()


class makeOracleData:

    def __init__(self, model):
        self.model = model
        with open('param_dict.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.paramDict = dict(reader)

    def funcGenOracle(self):
        dfTest = pd.read_csv('TestingData.csv')
        dataTest = dfTest.values
        predict_list = np.zeros((1, dfTest.shape[0]))
        X = dataTest[:, :-1]

        if 'numpy.ndarray' in str(type(self.model)):
            for i in range(0, X.shape[0]):
                predict_list[0][i] = np.sign(np.dot(self.model, X[i]))
                dfTest.loc[i, 'Class'] = int(predict_list[0][i])

        else:
            if self.paramDict['model_type'] == 'Pytorch':
                X = torch.tensor(X, dtype=torch.float32)
                predict_class = []
                for i in range(0, X.shape[0]):
                    predict_prob = self.model(X[i].view(-1, X.shape[1]))
                    predict_class.append(int(torch.argmax(predict_prob)))
                for i in range(0, X.shape[0]):
                    dfTest.loc[i, 'Class'] = predict_class[i]
            else:
                data_new = np.zeros((dataTest.shape[0], dataTest.shape[1]), dtype=object)
                predict_class = self.model.predict(X)
                for i in range(0, X.shape[0]):
                    dfTest.loc[i, 'Class'] = int(predict_class[i])
        dfTest.to_csv('OracleData.csv', index=False, header=True)


class propCheck:
    
    def __init__(self, max_samples=None, deadline=None, model=None, no_of_params=None, xml_file='', mul_cex=False,
                white_box_model=None, no_of_layers=None, layer_size=None, no_of_class=None,
                no_EPOCHS=None, model_with_weight=False, train_data_available=False, train_data_loc='',
                multi_label=False, model_type=None, model_path='', no_of_train=None, train_ratio=None):
        
        self.paramDict = {}

        if white_box_model == 'DNN' or multi_label:
            if no_of_class is None:
                raise Exception('Please provide the number of classes the dataset contain')
            else:
                self.paramDict['no_of_class'] = no_of_class

        if multi_label:
            multiLabelMain.multiLabelPropCheck(no_of_params=no_of_params, max_samples=max_samples, deadline=deadline, model=model,
                                               xml_file=xml_file, no_of_class=no_of_class, mul_cex=mul_cex,
                                               white_box_model=white_box_model, no_of_layers=no_of_layers,
                                               layer_size=layer_size, no_EPOCHS=no_EPOCHS, model_path=model_path, no_of_train=None,
                                               train_ratio=None, model_type=model_type)
        else:
            if max_samples is None:
                self.max_samples = 1000
            else:
                self.max_samples = max_samples
            self.paramDict['max_samples'] = self.max_samples
        
            if deadline is None:
                self.deadline = 500000
            else:
                self.deadline = deadline
            self.paramDict['deadlines'] = self.deadline
        
            if white_box_model is None:
                self.white_box_model = 'Decision tree'
            else:
                self.white_box_model = white_box_model
            self.paramDict['white_box_model'] = self.white_box_model    

            if self.white_box_model == 'DNN':
                if (no_of_layers is None) and (layer_size is None):
                    self.no_of_layers = 2
                    self.layer_size = 64
                elif no_of_layers is None:
                    self.no_of_layers = 2
                    self.layer_size = layer_size
                elif layer_size is None:
                    self.no_of_layers = no_of_layers
                    self.layer_size = 64
                elif (layer_size > 100) or(no_of_layers > 5):
                    raise Exception("White-box model is too big to translate")
                    sys.exit(1)    
                else:
                    self.no_of_layers = no_of_layers
                    self.layer_size = layer_size
                self.paramDict['no_of_layers'] = self.no_of_layers
                self.paramDict['layer_size'] = self.layer_size 
           
            if no_EPOCHS is None:
                self.paramDict['no_EPOCHS'] = 20
            else:
                self.paramDict['no_EPOCHS'] = no_EPOCHS
            
            if (no_of_params is None) or (no_of_params > 3):
                raise Exception("Please provide a value for no_of_params or the value of it is too big")
            else:
                self.no_of_params = no_of_params
            self.paramDict['no_of_params'] = self.no_of_params   
            self.paramDict['mul_cex_opt'] = mul_cex
            self.paramDict['multi_label'] = False

            if xml_file == '':
                raise Exception("Please provide a file name")
            else:
                try:
                    self.xml_file = xml_file
                except Exception as e:
                    raise Exception("File does not exist")
              
            f = open('MUTWeight.txt', 'w')
            if not model_with_weight:
                f.write(str(False))
                if model_type == 'sklearn':
                    if model is None:
                        if model_path == '':
                            raise Exception("Please provide a classifier to check")
                        else:
                            self.model = load(model_path)
                            self.paramDict['model_path'] = model_path
                            self.paramDict['model_type'] = 'sklearn'
                    
                    else:
                        self.paramDict['model_type'] = 'sklearn'
                        self.model = model
                        dump(self.model, 'Model/MUT.joblib')
                        
                elif model_type == 'Pytorch':
                    if model is None:
                        if model_path == '':
                            raise Exception("Please provide a classifier to check")
                        else:
                            self.paramDict['model_type'] = 'Pytorch'
                            self.paramDict['model_path'] = model_path
                            if 'ARCH1' in model_path:
                                self.model = NetArch1()
                            else:
                                self.model = NetArch2()
                            self.model = torch.load(model_path)
                            self.model.eval()
                    else:
                        self.paramDict['model_type'] = 'Pytorch'
                        self.model = model
                        self.model.eval()
                else:
                    raise Exception("Please provide the type of the model (Pytorch/sklearn)")

            #Adjusting for fairness aware test cases          
            else:
                dfWeight = pd.read_csv('MUTWeight.csv')
                pred_weight = dfWeight.values
                pred_weight = pred_weight[:, :-1]
                self.model = pred_weight
                f.write(str(True))
            
            f.close()

            if no_of_train is None:
                self.no_of_train = 1000
            else:
                self.no_of_train = no_of_train
            if train_data_available:
                if train_data_loc == '':
                    raise Exception('Please provide the training data location')
                    sys.exit(1)
                else:
                    if train_ratio is None:
                        self.paramDict['train_ratio'] = 100
                    else:
                        self.paramDict['train_ratio'] = train_ratio
            self.paramDict['no_of_train'] = self.no_of_train
            self.paramDict['train_data_available'] = train_data_available
            self.paramDict['train_data_loc'] = train_data_loc

            try:
                with open('param_dict.csv', 'w') as csv_file:
                    writer = cv.writer(csv_file)
                    for key, value in self.paramDict.items():
                        writer.writerow([key, value])
            except IOError:
                print("I/O error")

            genData = readXmlFile(self.xml_file)
            genData.funcReadXml()
            gen_oracle = makeOracleData(self.model)
            gen_oracle.funcGenOracle()
        
        

class runChecker:
    
    def __init__(self):
        self.df = pd.read_csv('OracleData.csv')
        f = open('MUTWeight.txt', 'r')
        self.MUTcontent = f.readline()
        f.close()
        with open('param_dict.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.paramDict = dict(reader)
        
        if self.MUTcontent == 'False':
            self.model_type = self.paramDict['model_type'] 
            if 'model_path' in self.paramDict:
                model_path = self.paramDict['model_path']
                if self.model_type == 'Pytorch':
                    self.model = NetArch1()
                    self.model = torch.load(model_path)
                    self.model.eval()
                else:    
                    self.model = load(model_path) 
            else:
                #if self.model_type == 'Pytorch':

                self.model = load('Model/MUT.joblib')
        else:
            dfWeight = pd.read_csv('MUTWeight.csv')
            pred_weight = dfWeight.values
            pred_weight = pred_weight[:, :-1]
            self.model = pred_weight
        with open('TestSet.csv', 'w', newline='') as csvfile:
            fieldnames = self.df.columns.values  
            writer = cv.writer(csvfile)
            writer.writerow(fieldnames)
        with open('CexSet.csv', 'w', newline='') as csvfile:
            fieldnames = self.df.columns.values  
            writer = cv.writer(csvfile)
            writer.writerow(fieldnames)  

    def funcCreateOracle(self):
        dfTest = pd.read_csv('TestingData.csv')
        data = dfTest.values
        X = data[:, :-1]
        if self.MUTcontent == 'False':
            if self.paramDict['model_type'] == 'Pytorch':
                X = torch.tensor(X, dtype=torch.float32)
                predict_class = []
                for i in range(0, X.shape[0]):
                    predict_prob = self.model(X[i].view(-1, X.shape[1]))
                    predict_class.append(int(torch.argmax(predict_prob)))
                for i in range(0, X.shape[0]):
                    dfTest.loc[i, 'Class'] = predict_class[i]   
            else:
                predict_class = self.model.predict(X)
                for i in range(0, X.shape[0]):
                    dfTest.loc[i, 'Class'] = predict_class[i]
            dfTest.to_csv('OracleData.csv', index = False, header = True)   
        else:
            predict_list = np.zeros((1, dfTest.shape[0]))
            for i in range(0, X.shape[0]):
                predict_list[0][i] = np.sign(np.dot(self.model, X[i]))
                dfTest.loc[i, 'Class'] = int(predict_list[0][i])
            dfTest.to_csv('OracleData.csv', index = False, header = True)

    def chkPairBel(self, tempMatrix, noAttr):
        firstTest = np.zeros((noAttr,))
        secTest = np.zeros((noAttr,))
        dfT = pd.read_csv('TestingSet.csv')
        tstMatrix = dfT.values

        for i in range(0, noAttr):
            firstTest[i] = tempMatrix[0][i]

        firstTestList = firstTest.tolist()
        secTestList = secTest.tolist()
        testMatrixList = tstMatrix.tolist()
        for i in range(0, len(testMatrixList) - 1):
            if firstTestList == testMatrixList[i]:
                if secTestList == testMatrixList[i + 1]:
                    return True
        return False

    def chkAttack(self, target_class):
        cexPair = ()
        dfTest = pd.read_csv('TestingSet.csv')
        dataTest = dfTest.values
        i = 0
        X = torch.tensor(dataTest, dtype=torch.float32)
        while i < dfTest.shape[0] - 1:
            # for j in range(0, dfTest.shape[1]):
            # firstTest[0][j] = dataTest[i][j]
            predict_prob = self.model(X[i].view(-1, X.shape[1]))
            pred_class = int(torch.argmax(predict_prob))
            if pred_class != target_class:
                cexPair = (X[i])
                print('A counter example is found \n')
                print(cexPair)
                return cexPair, True
            i = i + 1
        return cexPair, False

    def checkWithOracle(self):
        assume_dict = []
        f = open('assumeStmnt.txt', 'r')
        p = f.readlines()
        dfTr = pd.read_csv('Datasets/mnist_resized.csv')
        noOfAttr = dfTr.shape[1] - 1
        i = 0
        for lines in p:
            if p == '\n':
                pass
            else:
                for col in dfTr.columns.values:
                    if col in lines:
                        num = float(re.search(r'[+-]?([0-9]*[.])[0-9]+', lines).group(0))
                        assume_dict.append(num)
                        i += 1
        f1 = open('assertStmnt.txt', 'r')
        p1 = f1.readlines()
        num1 = float(re.search(r'[+-]?([0-9]*[.])[0-9]+', p1[0]).group(0))
        with open('TestingSet.csv', 'w', newline='') as csvfile:
            fieldnames = dfTr.columns.values
            writer = cv.writer(csvfile)
            writer.writerow(fieldnames)
        dfAg = pd.read_csv('TestingSet.csv')
        dfAg.drop('Class', axis=1, inplace=True)
        dfAg.to_csv('TestingSet.csv', index=False, header=True)
        inst_count = 0
        i = 0
        while inst_count < 1000:
            tempMatrix = np.zeros((1, noOfAttr))
            for i in range(0, len(assume_dict)):
                tempMatrix[0][i] = assume_dict[i]
            for i in range(len(assume_dict), noOfAttr):
                fe_type = dfTr.dtypes[i]
                fe_type = str(fe_type)
                if 'int' in fe_type:
                    tempMatrix[0][i] = rd.randint(dfTr.iloc[:, i].min(), dfTr.iloc[:, i].max())
                else:
                    tempMatrix[0][i] = rd.uniform(dfTr.iloc[:, i].min(), dfTr.iloc[:, i].max())
            if not self.chkPairBel(tempMatrix, noOfAttr):
                with open('TestingSet.csv', 'a', newline='') as csvfile:
                    writer = cv.writer(csvfile)
                    writer.writerows(tempMatrix)
            inst_count = inst_count + 1
        cexPair, flag = self.chkAttack(num1)
        if flag:
            return cexPair, True
        return cexPair, False

    def funcPrediction(self, X, dfCand, testIndx):
        if self.MUTcontent == 'False':
            if self.model_type == 'Pytorch':
                X_pred = torch.tensor(X[testIndx], dtype=torch.float32)

                predict_prob = self.model(X_pred.view(-1, X.shape[1]))
                return int(torch.argmax(predict_prob))
            else:
                if self.MUTcontent == 'False':
                    return self.model.predict(util.convDataInst(X, dfCand, testIndx, 1))
        else:
            temp_class = np.sign(np.dot(self.model, X[testIndx]))
            if temp_class < 0:
                return 0
            else:
                return temp_class

    def addModelPred(self):
        dfCexSet = pd.read_csv('CexSet.csv')
        dataCex = dfCexSet.values
        if self.MUTcontent == 'False':
            if self.model_type == 'Pytorch':
                X = dataCex[:, :-1]
                X = torch.tensor(X, dtype=torch.float32)
                predict_class=[]
                for i in range(0, X.shape[0]):
                    predict_prob = self.model(X[i].view(-1, X.shape[1]))
                    predict_class.append(int(torch.argmax(predict_prob)))
            else:        
                predict_class = self.model.predict(dataCex[:,:-1])
            for i in range(0, dfCexSet.shape[0]):
                dfCexSet.loc[i, 'Class'] = predict_class[i]
        else:        
            X = dataCex[:, :-1]
            predict_list = np.zeros((1, dfCexSet.shape[0]))
            for i in range(0, X.shape[0]):
                predict_list[0][i] = np.sign(np.dot(self.model, X[i]))
                if predict_list[0][i] < 0:
                    predict_list[0][i] = 0
                dfCexSet.loc[i, 'Class'] = predict_list[0][i]
        dfCexSet.to_csv('CexSet.csv', index = False, header = True)
    
    def runWithDNN(self):
        self.no_of_params = int(self.paramDict['no_of_params'])
        retrain_flag = False
        MAX_CAND_ZERO = 10
        count_cand_zero = 0
        count = 0
        satFlag = False
        start_time = time.time()
        ret_flag = False
        if self.no_of_params == 1:
            cex, ret_flag = self.checkWithOracle()
            if ret_flag:
                with open('CexSet.csv', 'a', newline='') as csvfile:
                    writer = cv.writer(csvfile)
                    writer.writerows(np.reshape(np.array(cex), (1, self.df.shape[1]-1)))
                self.addModelPred()
                return 0

        while count < self.max_samples:
            trainDNN.functrainDNN()
            print('count is:', count)
            obj_dnl = DNN2logic.ConvertDNN2logic()
            obj_dnl.funcDNN2logic()
            util.storeAssumeAssert('DNNSmt.smt2')
            util.addSatOpt('DNNSmt.smt2')
            os.system(r"z3 DNNSmt.smt2 > FinalOutput.txt")
            satFlag = ReadZ3Output.funcConvZ3OutToData(self.df)
            if not satFlag:
                if count == 0:
                    print('No CEX is found by the checker in the first trial')
                    return 0
                elif (count != 0) and (self.mul_cex == 'True'):
                    dfCexSet = pd.read_csv('CexSet.csv')
                    if round(dfCexSet.shape[0]/self.no_of_params) == 0:
                        print('No CEX is found')
                        return 0
                    print('Total number of cex found is:', round(dfCexSet.shape[0]/self.no_of_params))
                    self.addModelPred()
                    return round(dfCexSet.shape[0]/self.no_of_params)
                elif (count != 0) and (self.mul_cex == 'False'):
                    print('No Cex is found after '+str(count)+' no. of trials')
                    return 0
            else:
                processCandCex.funcAddCex2CandidateSet()
                processCandCex.funcAddCexPruneCandidateSet4DNN()
                processCandCex.funcCheckCex()
                #Increase the count if no further candidate cex has been found
                dfCand = pd.read_csv('Cand-set.csv')
                if round(dfCand.shape[0]/self.no_of_params) == 0:
                    count_cand_zero += 1
                    if count_cand_zero == MAX_CAND_ZERO:
                        if self.mul_cex == 'True':
                            dfCexSet = pd.read_csv('CexSet.csv')
                            print('Total number of cex found is:', round(dfCexSet.shape[0]/self.no_of_params))
                            if round(dfCexSet.shape[0]/self.no_of_params) > 0:
                                self.addModelPred()
                            return round(dfCexSet.shape[0]/self.no_of_params) + 1
                        else:
                            print('No CEX is found by the checker')
                            return 0
                else:
                    count = count + round(dfCand.shape[0]/self.no_of_params)

                data = dfCand.values
                X = data[:, :-1]
                y = data[:, -1]
                if dfCand.shape[0] % self.no_of_params == 0:
                    arr_length = dfCand.shape[0]
                else:
                    arr_length = dfCand.shape[0]-1
                testIndx = 0

                while testIndx < arr_length:
                    temp_count = 0
                    temp_store = []
                    temp_add_oracle = []
                    for i in range(0, self.no_of_params):
                        if self.funcPrediction(X, dfCand, testIndx) == y[testIndx]:
                            temp_store.append(X[testIndx])
                            temp_count += 1
                            testIndx += 1
                        else:
                            retrain_flag = True
                            temp_add_oracle.append(X[testIndx])
                            testIndx += 1
                    if temp_count == self.no_of_params:
                        if self.mul_cex == 'True':
                            with open('CexSet.csv', 'a', newline='') as csvfile:
                                writer = cv.writer(csvfile)
                                writer.writerows(temp_store)
                        else:
                            print('A counter example is found, check it in CexSet.csv file: ', temp_store)
                            with open('CexSet.csv', 'a', newline='') as csvfile:
                                writer = cv.writer(csvfile)
                                writer.writerows(temp_store)
                            self.addModelPred()
                            return 1
                    else:
                        util.funcAdd2Oracle(temp_add_oracle)
                        
                    if retrain_flag:
                        self.funcCreateOracle()
                    
                if (time.time() - start_time) > self.deadline:
                    print("Time out")
                    break

        dfCexSet = pd.read_csv('CexSet.csv')
        if (round(dfCexSet.shape[0]/self.no_of_params) > 0) and (count >= self.max_samples):
            self.addModelPred()
            print('Total number of cex found is:', round(dfCexSet.shape[0]/self.no_of_params))
            print('No. of Samples looked for counter example has exceeded the max_samples limit')
        else:
            print('No counter example has been found')

    def runPropCheck(self):
        retrain_flag = False
        MAX_CAND_ZERO = 5
        count_cand_zero = 0
        count = 0
        satFlag = False
        self.max_samples = int(self.paramDict['max_samples'])
        self.no_of_params = int(self.paramDict['no_of_params'])
        self.mul_cex = self.paramDict['mul_cex_opt']
        self.deadline = int(self.paramDict['deadlines'])
        white_box = self.paramDict['white_box_model']
        start_time = time.time()
        
        if white_box == 'DNN':
            self.runWithDNN()
        else:    
            while count < self.max_samples:
                print('count is:', count)
                tree = trainDecTree.functrainDecTree()
                tree2Logic.functree2LogicMain(tree, self.no_of_params)
                util.storeAssumeAssert('DecSmt.smt2')
                util.addSatOpt('DecSmt.smt2')
                os.system(r"z3 DecSmt.smt2 > FinalOutput.txt")
                satFlag = ReadZ3Output.funcConvZ3OutToData(self.df)
                if  not satFlag:
                    if count == 0:
                        print('No CEX is found by the checker at the first trial')
                        return 0
                    elif (count != 0) and (self.mul_cex == 'True'):
                        dfCexSet = pd.read_csv('CexSet.csv')
                        if round(dfCexSet.shape[0]/self.no_of_params) == 0:
                            print('No CEX is found')
                            return 0
                        print('Total number of cex found is:', round(dfCexSet.shape[0]/self.no_of_params))
                        self.addModelPred()
                        return round(dfCexSet.shape[0]/self.no_of_params) 
                    elif (count != 0) and (self.mul_cex == 'False'):
                        print('No Cex is found after '+str(count)+' no. of trials')
                        return 0
                else:
                    processCandCex.funcAddCex2CandidateSet()
                    processCandCex.funcAddCexPruneCandidateSet(tree)
                    processCandCex.funcCheckCex()
                    #Increase the count if no further candidate cex has been found
                    dfCand = pd.read_csv('Cand-set.csv')
                    if round(dfCand.shape[0]/self.no_of_params) == 0:
                        count_cand_zero += 1
                        if count_cand_zero == MAX_CAND_ZERO:
                            if self.mul_cex == 'True':
                                dfCexSet = pd.read_csv('CexSet.csv')
                                print('Total number of cex found is:', round(dfCexSet.shape[0]/self.no_of_params))
                                if round(dfCexSet.shape[0]/self.no_of_params) > 0:
                                    self.addModelPred()
                                return round(dfCexSet.shape[0]/self.no_of_params) + 1
                            else:
                                print('No CEX is found by the checker')
                                return 0
                    else:
                        count = count + round(dfCand.shape[0]/self.no_of_params)
                    
                    data = dfCand.values
                    X = data[:, :-1]
                    y = data[:, -1]    
                    if dfCand.shape[0] % self.no_of_params == 0:
                        arr_length = dfCand.shape[0]
                    else:
                        arr_length = dfCand.shape[0]-1    
                    testIndx = 0
                    while testIndx < arr_length:
                        temp_count = 0
                        temp_store = []
                        temp_add_oracle = []
                        for i in range(0, self.no_of_params):
                            if self.funcPrediction(X, dfCand, testIndx) == y[testIndx]:
                                temp_store.append(X[testIndx])
                                temp_count += 1
                                testIndx += 1
                            else:
                                retrain_flag = True
                                temp_add_oracle.append(X[testIndx])
                                testIndx += 1
                        if temp_count == self.no_of_params:
                            if self.mul_cex == 'True':
                                with open('CexSet.csv', 'a', newline='') as csvfile:    
                                    writer = cv.writer(csvfile)
                                    writer.writerows(temp_store)
                            else:
                                print('A counter example is found, check it in CexSet.csv file: ', temp_store)
                                with open('CexSet.csv', 'a', newline='') as csvfile:    
                                    writer = cv.writer(csvfile)
                                    writer.writerows(temp_store)
                                self.addModelPred()    
                                return 1
                        else:
                            util.funcAdd2Oracle(temp_add_oracle)
                        
                    if retrain_flag:
                        self.funcCreateOracle()
                    
                    if (time.time() - start_time) > self.deadline:
                        print("Time out")
                        break
             
            dfCexSet = pd.read_csv('CexSet.csv')
            if (round(dfCexSet.shape[0]/self.no_of_params) > 0) and (count >= self.max_samples):
                self.addModelPred()
                print('Total number of cex found is:', round(dfCexSet.shape[0]/self.no_of_params))
                print('No. of Samples looked for counter example has exceeded the max_samples limit')
            else:
                print('No counter example has been found')


def Assume(*args):    
    grammar = Grammar(
            r"""
    
        expr        = expr1 / expr2 / expr3 /expr4 /expr5 / expr6 /expr7
        expr1       = expr_dist1 logic_op num_log
        expr2       = expr_dist2 logic_op num_log
        expr3       = classVar ws logic_op ws value
        expr4       = classVarArr ws logic_op ws value
        expr5       = classVar ws logic_op ws classVar
        expr6       = classVarArr ws logic_op ws classVarArr
        expr7       = "True"
        expr_dist1  = op_beg?abs?para_open classVar ws arith_op ws classVar para_close op_end?
        expr_dist2  = op_beg?abs?para_open classVarArr ws arith_op ws classVarArr para_close op_end?
        classVar    = variable brack_open number brack_close
        classVarArr = variable brack_open variable brack_close
        para_open   = "("
        para_close  = ")"
        brack_open  = "["
        brack_close = "]"
        variable    = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
        logic_op    = ws (geq / leq / eq / neq / and / lt / gt) ws
        op_beg      = number arith_op
        op_end      = arith_op number
        arith_op    = (add/sub/div/mul)
        abs         = "abs"
        add         = "+"
        sub         = "-"
        div         = "/"
        mul         = "*"
        lt          = "<"
        gt          = ">"
        geq         = ">="
        leq         = "<="
        eq          = "="
        neq         = "!="
        and         = "&"
        ws          = ~"\s*"
        value       = ~"\d+"
        num_log     = ~"[+-]?([0-9]*[.])?[0-9]+"
        number      = ~"[+-]?([0-9]*[.])?[0-9]+"
        """
        )
        
    tree = grammar.parse(args[0])
    assumeVisitObj = assume2logic.AssumptionVisitor()
    if len(args) == 3:
        assumeVisitObj.storeInd(args[1])
        assumeVisitObj.storeArr(args[2])
        assumeVisitObj.visit(tree)
    elif len(args) == 2:
        assumeVisitObj.storeInd(args[1])
        assumeVisitObj.visit(tree)
    elif len(args) == 1:
        assumeVisitObj.visit(tree)    
            

def Assert(*args):    
    grammar = Grammar(
    r"""
    expr        = expr1 / expr2/ expr3
    expr1       = classVar ws operator ws number
    expr2       = classVar ws operator ws classVar
    expr3       = classVar mul_cl_var ws operator ws neg? classVar mul_cl_var
    classVar    = class_pred brack_open variable brack_close
    model_name  = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
    class_pred  = model_name classSymbol
    classSymbol = ~".predict"
    brack_open  = "("
    brack_close = ")"
    variable    = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
    brack3open  = "["
    brack3close = "]"
    class_name  = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
    mul_cl_var  = brack3open class_name brack3close
    operator    = ws (gt/ lt/ geq / leq / eq / neq / and/ implies) ws
    lt          = "<"
    gt          = ">"
    geq         = ">="
    implies     = "=>"
    neg         = "~"
    leq         = "<="
    eq          = "=="
    neq         = "!="
    and         = "&"
    ws          = ~"\s*"
    number      = ~"[+-]?([0-9]*[.])?[0-9]+"
    """
    )
      
    tree = grammar.parse(args[0])
    assertVisitObj = assert2logic.AssertionVisitor()
    assertVisitObj.visit(tree)
    
    with open('param_dict.csv') as csv_file:
        reader = cv.reader(csv_file)
        paramDict = dict(reader)
    if paramDict['multi_label'] == 'True':
        start_time = time.time()
        obj_multi = multiLabelMain.runChecker()
        obj_multi.runPropCheck()
        print('time required is', time.time()-start_time)
    else:
        obj_faircheck = runChecker()
        start_time = time.time()
        obj_faircheck.runPropCheck()
        print('time required is', time.time()-start_time)

    if os.path.exists('assumeStmnt.txt'):
        os.remove('assumeStmnt.txt')
    if os.path.exists('assertStmnt.txt'):
        os.remove('assertStmnt.txt')

    if os.path.exists('Cand-set.csv'):
        os.remove('Cand-set.csv')
    if os.path.exists('CandidateSet.csv'):
        os.remove('CandidateSet.csv')
    if os.path.exists('CandidateSetInst.csv'):
        os.remove('CandidateSetInst.csv')
    if os.path.exists('CandidateSetBranch.csv'):
        os.remove('CandidateSetBranch.csv')
   
    if os.path.exists('TestDataSMT.csv'):
        os.remove('TestDataSMT.csv')
    if os.path.exists('TestDataSMTMain.csv'):
        os.remove('TestDataSMTMain.csv')
   
    if os.path.exists('DecSmt.smt2'):
        os.remove('DecSmt.smt2')
    if os.path.exists('ToggleBranchSmt.smt2'):
        os.remove('ToggleBranchSmt.smt2')
    if os.path.exists('ToggleFeatureSmt.smt2'):
        os.remove('ToggleFeatureSmt.smt2')
    if os.path.exists('TreeOutput.txt'):
        os.remove('TreeOutput.txt')

    if os.path.exists('SampleFile.txt'):
        os.remove('SampleFile.txt')
    if os.path.exists('FinalOutput.txt'):
        os.remove('FinalOutput.txt')
    if os.path.exists('MUTWeight.txt'):
        os.remove('MUTWeight.txt')
    if os.path.exists('ConditionFile.txt'):
        os.remove('ConditionFile.txt')
    

    if os.path.exists('MUTWeight.csv'):
        os.remove('MUTWeight.csv')
    if os.path.exists('MUTWeight.txt'):
        os.remove('MUTWeight.txt')
    if os.path.exists('DNNSmt.smt2'):
        os.remove('DNNSmt.smt2')

    if os.path.exists('TestData.csv'):
        os.remove('TestData.csv')
    if os.path.exists('TestDataSet.csv'):
        os.remove('TestDataSet.csv')
    if os.path.exists('CandTestDataSet.csv'):
        os.remove('CandTestDataSet.csv')
   









