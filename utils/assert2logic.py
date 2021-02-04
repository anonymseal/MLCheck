#!/usr/bin/env python
# coding: utf-8

# In[1]:


from parsimonious.nodes import NodeVisitor
from parsimonious.grammar import Grammar
from itertools import groupby
import csv as cv
import re, sys
import pandas as pd


# In[2]:


class AssertionVisitor(NodeVisitor):
    
    def __init__(self):
        self.currentClass = []
        self.modelVarList = []
        self.classNameList = []
        self.currentOperator = ""
        self.negOp = ""
        self.varList = []
        self.mydict = {}
        self.varMap = {}
        self.feVal = 0
        self.count = 0
        self.dfOracle = pd.read_csv('OracleData.csv')
        with open('dict.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.mydict = dict(reader)  
        with open('param_dict.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.paramDict = dict(reader)      
    
    def generic_visit(self, node, children):
        pass
    
    def visit_classVar(self, node, children):
        if(self.mydict['no_mapping'] == 'True'):
            pass
        else:
            for el in self.varList:
                if(el in node.text):
                    if(self.mydict['no_assumption'] == 'False'):
                        className = 'Class'+str(self.mydict[el])
                    else:
                        className = 'Class'+str(self.count)
            self.currentClass.append(className)
        
    def visit_neg(self, node, children):
        self.negOp = node.text
        
    def visit_model_name(self, node, children):
        self.modelVarList.append(node.text)
        
    def visit_class_name(self, node, children):
        if(node.text in self.dfOracle.columns.values):
            self.classNameList.append(node.text)
        else:
            raise Exception('Class name '+str(node.text)+' do not exist')
        
    def visit_variable(self, node, children):
        if(self.mydict['no_mapping'] == 'True'):
            pass
        else:
            self.varList.append(node.text)
            if(self.mydict['no_assumption'] == 'False'):
                num = str(int(re.search(r'\d+', self.mydict[node.text]).group(0)))
                self.mydict[node.text] = num[len(num)-1]
            else:
                if(node.text in self.varMap):
                    pass
                else:
                    self.varMap[node.text] = self.count
                    self.count += 1
    
    def visit_operator(self, node, children):
        if('!=' in node.text):
            self.currentOperator = 'not(= '
        elif('==' in node.text):
            self.currentOperator = '= '
        else:
            self.currentOperator = node.text
    
    def visit_number(self, node, children):
        self.feVal = float(node.text)

    def visit_expr1(self, node, children):
        if(self.mydict['no_mapping'] == 'True'):
            assertStmnt = ('(assert(not (', self.currentOperator,' Class', str(0), ' ', str(self.feVal), ')))')
        else:    
            assertStmnt = ('(assert(not (', self.currentOperator,self.currentClass[0], ' ', str(self.feVal), ')))')
        f = open('assertStmnt.txt', 'a')
        for x in assertStmnt:
            f.write(x)
        if(self.currentOperator == 'not(= '):
            f.write(')')
        f.close()    
        
    def checkModelName(self):
        if(self.modelVarList[0] != self.modelVarList[1]):
            raise Exception('Model names do not match')
            sys.exit(1)
    
    def visit_expr2(self, node, children):
        self.checkFeConsist()
        self.checkModelName()
        assertStmnt = ('(assert(not (', self.currentOperator,self.currentClass[0], ' ', self.currentClass[1], ')))')
        f = open('assertStmnt.txt', 'a')
        f.write('\n')
        for x in assertStmnt:
            f.write(x)
        if(self.currentOperator == 'not(= '):
            f.write(')')
        f.close() 
        
    def visit_expr3(self, node, children):
        if(self.count > int(self.paramDict['no_of_params'])):
            raise Exception('The no. of parameters mentioned exceeded in assert statement')
            sys.exit(1)
        self.checkModelName()
        if(self.negOp == '~'):
            if(self.paramDict['white_box_model'] == 'DNN'):
                assertStmnt = ('(assert(not (', self.currentOperator,' (= ', self.classNameList[0],str(self.count-1),
                               ' 1)', ' (not ', ' (= ', self.classNameList[1],str(self.count-1),' 1)','))))')
            else:
                assertStmnt = ('(assert(not (', self.currentOperator,' (= ', self.classNameList[0],' 1)', 
                           ' (not ', ' (= ', self.classNameList[1],' 1)','))))')
        else:
            if(self.paramDict['white_box_model'] == 'DNN'):
                assertStmnt = ('(assert(not (', self.currentOperator,' (= ', self.classNameList[0],str(self.count-1),' 1)', ' ', 
                                                                ' (= ', self.classNameList[1],str(self.count-1),' 1)', ')))')
            else:    
                assertStmnt = ('(assert(not (', self.currentOperator,' (= ', self.classNameList[0],' 1)', ' ', 
                                                                ' (= ', self.classNameList[1],' 1)', ')))')
        f = open('assertStmnt.txt', 'a')
        f.write('\n')
        for x in assertStmnt:
            f.write(x)
        f.close()    
   
        
    def checkFeConsist(self):
        if(len(self.varList) == len(self.mydict)-2):
            for el in self.varList:
                if(el not in self.mydict.keys()):
                    raise Exception("Unknown feature vector")
                    sys.exit(1)
        
        else:
            raise Exception("No. of feature vectors do not match with the assumption")
            sys.exit(1)
               
            


# In[ ]:



    
    

