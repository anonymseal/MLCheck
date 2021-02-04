#!/usr/bin/env python


import pandas as pd
import csv as cv
import sys



def funcWriteXml(df):
    
    f = open('dataInput.xml', 'w')
    f.write('<?xml version="1.0" encoding="UTF-8"?> \n <Inputs> \n')
    
    for i in range(0, df.shape[1]):
        f.write('<Input> \n <Feature-name>')
        f.write(df.columns.values[i])
        f.write('<\Feature-name> \n <Feature-type>')
        f.write(str(df.dtypes[i]))
        f.write('<\Feature-type> \n <Value> \n <minVal>')
        f.write(str(df.iloc[:, i].min()))
        f.write('<\minVal> \n <maxVal>')
        f.write(str(df.iloc[:, i].max()))
        f.write('<\maxVal> \n <\Value> \n <\Input>\n')

    f.write('<\Inputs>') 
    f.close()


df = pd.read_csv(str(sys.argv[1]))
funcWriteXml(df)


fe_dict = {}
for i in range(0, df.shape[1]):
    fe_dict[df.columns.values[i]] = str(df.dtypes[i])

try:
    with open('feNameType.csv', 'w') as csv_file:
        writer = cv.writer(csv_file)
        for key, value in fe_dict.items():
            writer.writerow([key, value])
except IOError:
    print("I/O error")


