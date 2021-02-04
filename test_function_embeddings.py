from mlCheck import Assume, Assert, propCheck
import pandas as pd
import os
import sys
sys.path.append("../")
import csv as cv
from MainFiles import quickTestMlAt
from tqdm import tqdm

MAX_SAMPLES = 1000
iteration_no = int(input('How many times would you want each test case to execute: \n'))
white_box = ['DNN', 'Decision tree']
f = open('Output/Subsumptions-disjointResults.txt', 'w')

model_list = ['Multi_label_classifiers/act_per_cel.joblib', 'Multi_label_classifiers/act_per_cel_RF.joblib']
os.system('python Dataframe2XML.py Datasets/dbo_Actor_dbo_CelestialBody_dbo_Person_Multi.csv')
for job in model_list:
    if 'RF' in job:
        f.write('\n----Results for CR1(RF)-----\n\n')
    else:
        f.write('\n----Results for CR1(NN)-----\n\n')
    for box in white_box:
        if box == 'Decision tree':
            f.write('MLC_NN results-\n')
        else:
            f.write('MLC_DT results-\n')
        cex_count_S1 = 0
        cex_count_D1 = 0
        cex_count_D2 = 0
        for no in range(0, iteration_no):
            propCheck(no_of_params=1, max_samples=1000, model_type='sklearn', model_path=job, mul_cex=False,
            xml_file='dataInput.xml', white_box_model=box, multi_label=True, no_of_class=3, no_of_layers=2, layer_size=10)

            for i in range(0, 3):

                if i == 0:
                    Assume('True')
                    Assert('model.predict(x)[Actor] => model.predict(x)[Person]')
                    dfCexSet = pd.read_csv('CexSet.csv')
                    if dfCexSet.shape[0] == 1:
                        cex_count_S1 += 1

                elif i == 1:
                    Assume('True')
                    Assert('model.predict(x)[Person] => ~model.predict(x)[CelestialBody]')
                    dfCexSet = pd.read_csv('CexSet.csv')
                    if dfCexSet.shape[0] == 1:
                        cex_count_D1 += 1
                else:
                    Assume('True')
                    Assert('model.predict(x)[Actor] => ~model.predict(x)[CelestialBody]')
                    dfCexSet = pd.read_csv('CexSet.csv')
                    if dfCexSet.shape[0] == 1:
                        cex_count_D2 += 1

        f.write('Probability value of S1 is: '+str(cex_count_S1/iteration_no)+'\n')
        f.write('Probability value of D1 is: ' + str(cex_count_D1 / iteration_no) + '\n')
        f.write('Probability value of D2 is: ' + str(cex_count_D2 / iteration_no) + '\n\n')

    paramDict = {}
    f.write('PBT results are---\n')
    for j in range(0, 3):
        paramDict['classification_type'] = 'multi_label'
        paramDict['no_of_param'] = 1
        paramDict['no_of_class'] = 3
        cex_count_S1 = 0
        cex_count_D1 = 0
        cex_count_D2 = 0
        if j == 0:
            paramDict['property_type'] = 'subsumption'
        elif j == 1:
            paramDict['property_type'] = 'disjunction_artOrAct'
        else:
            paramDict['property_type'] = 'disjunction_person'
        try:
            with open('param_dict.csv', 'w', newline='') as csv_file:
                writer = cv.writer(csv_file)
                for key, value in paramDict.items():
                    writer.writerow([key, value])
        except IOError:
            print("I/O error")
        #actor_person_celestial model evaluation
        cexFlag = False
        execTime = 0
        failed_trials = 0
        #Reading the dataset
        df = pd.read_csv('Datasets/dbo_Actor_dbo_CelestialBody_dbo_Person_Multi.csv')
        for i in tqdm(range(iteration_no)):
            cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, job, df)
            if (len(cexPair) >= 1) & (cexFlag == False):
                if j == 0:
                    cex_count_S1 += 1
                elif j == 1:
                    cex_count_D1 += 1
                elif j == 2:
                    cex_count_D2 += 1
        if j == 0:
            f.write('Probability value of S1 is: ' + str(cex_count_S1 / iteration_no) + '\n')
        elif j == 1:
            f.write('Probability value of D1 is: ' + str(cex_count_D1 / iteration_no) + '\n')
        elif j == 2:
            f.write('Probability value of D2 is: ' + str(cex_count_D2 / iteration_no) + '\n\n')


os.system('python Dataframe2XML.py Datasets/dbo_Actor_dbo_Planet_dbo_Person_Multi.csv')
model_list = ['Multi_label_classifiers/act_per_planet.joblib', 'Multi_label_classifiers/act_per_planet_RF.joblib']
for job in model_list:
    if 'RF' in job:
        f.write('\n----Results for CR2(RF)-----\n\n')
    else:
        f.write('\n----Results for CR2(NN)-----\n\n')
    for box in white_box:
        if box == 'Decision tree':
            f.write('MLC_DT results-\n')
        else:
            f.write('MLC_NN results-\n')
        cex_count_S1 = 0
        cex_count_D1 = 0
        cex_count_D2 = 0
        for no in range(0, iteration_no):
            propCheck(no_of_params=1, max_samples=1000, model_type='sklearn', model_path=job, mul_cex=False,
            xml_file='dataInput.xml', white_box_model=box, multi_label=True, no_of_class=3, no_of_layers=2, layer_size=10)

            for i in range(0, 3):
                if i == 0:
                    Assume('True')
                    Assert('model.predict(x)[Actor] => model.predict(x)[Person]')
                    dfCexSet = pd.read_csv('CexSet.csv')
                    if dfCexSet.shape[0] == 1:
                        cex_count_S1 += 1

                elif i == 1:
                    Assume('True')
                    Assert('model.predict(x)[Person] => ~model.predict(x)[Planet]')
                    dfCexSet = pd.read_csv('CexSet.csv')
                    if dfCexSet.shape[0] == 1:
                        cex_count_D1 += 1
                else:
                    Assume('True')
                    Assert('model.predict(x)[Actor] => ~model.predict(x)[Planet]')
                    dfCexSet = pd.read_csv('CexSet.csv')
                    if dfCexSet.shape[0] == 1:
                        cex_count_D2 += 1

        f.write('Probability value of S1 is: '+str(cex_count_S1/iteration_no)+'\n')
        f.write('Probability value of D1 is: ' + str(cex_count_D1 / iteration_no) + '\n')
        f.write('Probability value of D2 is: ' + str(cex_count_D2 / iteration_no) + '\n\n')

    paramDict = {}
    f.write('PBT results are---\n')
    for j in range(0, 3):
        paramDict['classification_type'] = 'multi_label'
        paramDict['no_of_param'] = 1
        paramDict['no_of_class'] = 3
        cex_count_S1 = 0
        cex_count_D1 = 0
        cex_count_D2 = 0
        if j == 0:
            paramDict['property_type'] = 'subsumption'
        elif j == 1:
            paramDict['property_type'] = 'disjunction_artOrAct'
        else:
            paramDict['property_type'] = 'disjunction_person'
        try:
            with open('param_dict.csv', 'w', newline='') as csv_file:
                writer = cv.writer(csv_file)
                for key, value in paramDict.items():
                    writer.writerow([key, value])
        except IOError:
            print("I/O error")
        #actor_person_celestial model evaluation
        cexFlag = False
        execTime = 0
        failed_trials = 0
        #Reading the dataset
        df = pd.read_csv('Datasets/dbo_Actor_dbo_Planet_dbo_Person_Multi.csv')
        for i in tqdm(range(iteration_no)):
            cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, job, df)
            if (len(cexPair) >= 1) & (cexFlag == False):
                if j == 0:
                    cex_count_S1 += 1
                elif j == 1:
                    cex_count_D1 += 1
                elif j == 2:
                    cex_count_D2 += 1
        if j == 0:
            f.write('Probability value of S1 is: ' + str(cex_count_S1 / iteration_no) + '\n')
        elif j == 1:
            f.write('Probability value of D1 is: ' + str(cex_count_D1 / iteration_no) + '\n')
        elif j == 2:
            f.write('Probability value of D2 is: ' + str(cex_count_D2 / iteration_no) + '\n\n')

os.system('python Dataframe2XML.py Datasets/PYKE_embeddings_Actor_Person_Place_Multi_New.csv')
model_list = ['Multi_label_classifiers/act_per_pal.joblib', 'Multi_label_classifiers/act_per_cel_RF.joblib']
for job in model_list:
    if 'RF' in job:
        f.write('\n----Results for CR3(RF)-----\n\n')
    else:
        f.write('\n----Results for CR3(NN)-----\n\n')
    for box in white_box:
        if box == 'Decision tree':
            f.write('MLC_DT results-\n')
        else:
            f.write('MLC_NN results-\n')
        cex_count_S1 = 0
        cex_count_D1 = 0
        cex_count_D2 = 0
        for no in range(0, iteration_no):
            propCheck(no_of_params=1, max_samples=1000, model_type='sklearn', model_path=job, mul_cex=False,
            xml_file='dataInput.xml', white_box_model=box, multi_label=True, no_of_class=3, no_of_layers=2, layer_size=10)

            for i in range(0, 3):
                if i == 0:
                    Assume('True')
                    Assert('model.predict(x)[Actor] => model.predict(x)[Person]')
                    dfCexSet = pd.read_csv('CexSet.csv')
                    if dfCexSet.shape[0] == 1:
                        cex_count_S1 += 1

                elif i == 1:
                    Assume('True')
                    Assert('model.predict(x)[Person] => ~model.predict(x)[Place]')
                    dfCexSet = pd.read_csv('CexSet.csv')
                    if dfCexSet.shape[0] == 1:
                        cex_count_D1 += 1
                else:
                    Assume('True')
                    Assert('model.predict(x)[Actor] => ~model.predict(x)[Place]')
                    dfCexSet = pd.read_csv('CexSet.csv')
                    if dfCexSet.shape[0] == 1:
                        cex_count_D2 += 1

        f.write('Probability value of S1 is: '+str(cex_count_S1/iteration_no)+'\n')
        f.write('Probability value of D1 is: ' + str(cex_count_D1 / iteration_no) + '\n')
        f.write('Probability value of D2 is: ' + str(cex_count_D2 / iteration_no) + '\n\n')

    paramDict = {}
    f.write('PBT results are---\n')
    for j in range(0, 3):
        paramDict['classification_type'] = 'multi_label'
        paramDict['no_of_param'] = 1
        paramDict['no_of_class'] = 3
        cex_count_S1 = 0
        cex_count_D1 = 0
        cex_count_D2 = 0
        if j == 0:
            paramDict['property_type'] = 'subsumption'
        elif j == 1:
            paramDict['property_type'] = 'disjunction_artOrAct'
        else:
            paramDict['property_type'] = 'disjunction_person'
        try:
            with open('param_dict.csv', 'w', newline='') as csv_file:
                writer = cv.writer(csv_file)
                for key, value in paramDict.items():
                    writer.writerow([key, value])
        except IOError:
            print("I/O error")
        # actor_person_celestial model evaluation
        cexFlag = False
        execTime = 0
        failed_trials = 0
        # Reading the dataset
        df = pd.read_csv('Datasets/PYKE_embeddings_Actor_Person_Place_Multi_New.csv')
        for i in tqdm(range(iteration_no)):
            cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, job, df)
            if (len(cexPair) >= 1) & (cexFlag == False):
                if j == 0:
                    cex_count_S1 += 1
                elif j == 1:
                    cex_count_D1 += 1
                elif j == 2:
                    cex_count_D2 += 1
        if j == 0:
            f.write('Probability value of S1 is: ' + str(cex_count_S1 / iteration_no) + '\n')
        elif j == 1:
            f.write('Probability value of D1 is: ' + str(cex_count_D1 / iteration_no) + '\n')
        elif j == 2:
            f.write('Probability value of D2 is: ' + str(cex_count_D2 / iteration_no) + '\n\n')


os.system('python Dataframe2XML.py Datasets/dbo_Artist_dbo_CelestialBody_dbo_Person_Multi.csv')
model_list = ['Multi_label_classifiers/art_per_cel.joblib', 'Multi_label_classifiers/art_per_cel_RF.joblib']
for job in model_list:
    if 'RF' in job:
        f.write('\n----Results for CR4(RF)-----\n\n')
    else:
        f.write('\n----Results for CR4(NN)-----\n\n')
    for box in white_box:
        if box == 'Decision tree':
            f.write('MLC_NN results-\n')
        else:
            f.write('MLC_DT results-\n')
        cex_count_S1 = 0
        cex_count_D1 = 0
        cex_count_D2 = 0
        for no in range(0, iteration_no):
            propCheck(no_of_params=1, max_samples=1000, model_type='sklearn', model_path=job, mul_cex=False,
            xml_file='dataInput.xml', white_box_model=box, multi_label=True, no_of_class=3, no_of_layers=2, layer_size=10)

            for i in range(0, 3):
                if i == 0:
                    Assume('True')
                    Assert('model.predict(x)[Artist] => model.predict(x)[Person]')
                    dfCexSet = pd.read_csv('CexSet.csv')
                    if dfCexSet.shape[0] == 1:
                        cex_count_S1 += 1

                elif i == 1:
                    Assume('True')
                    Assert('model.predict(x)[Person] => ~model.predict(x)[CelestialBody]')
                    dfCexSet = pd.read_csv('CexSet.csv')
                    if dfCexSet.shape[0] == 1:
                        cex_count_D1 += 1
                else:
                    Assume('True')
                    Assert('model.predict(x)[Artist] => ~model.predict(x)[CelestialBody]')
                    dfCexSet = pd.read_csv('CexSet.csv')
                    if dfCexSet.shape[0] == 1:
                        cex_count_D2 += 1

        f.write('Probability value of S1 is: '+str(cex_count_S1/iteration_no)+'\n')
        f.write('Probability value of D1 is: ' + str(cex_count_D1 / iteration_no) + '\n')
        f.write('Probability value of D2 is: ' + str(cex_count_D2 / iteration_no) + '\n\n')

    paramDict = {}
    f.write('PBT results are---\n')
    for j in range(0, 3):
        paramDict['classification_type'] = 'multi_label'
        paramDict['no_of_param'] = 1
        paramDict['no_of_class'] = 3
        cex_count_S1 = 0
        cex_count_D1 = 0
        cex_count_D2 = 0
        if j == 0:
            paramDict['property_type'] = 'subsumption'
        elif j == 1:
            paramDict['property_type'] = 'disjunction_artOrAct'
        else:
            paramDict['property_type'] = 'disjunction_person'
        try:
            with open('param_dict.csv', 'w', newline='') as csv_file:
                writer = cv.writer(csv_file)
                for key, value in paramDict.items():
                    writer.writerow([key, value])
        except IOError:
            print("I/O error")
        # actor_person_celestial model evaluation
        cexFlag = False
        execTime = 0
        failed_trials = 0
        # Reading the dataset
        df = pd.read_csv('Datasets/dbo_Artist_dbo_CelestialBody_dbo_Person_Multi.csv')
        for i in tqdm(range(iteration_no)):
            cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, job, df)
            if (len(cexPair) >= 1) & (cexFlag == False):
                if j == 0:
                    cex_count_S1 += 1
                elif j == 1:
                    cex_count_D1 += 1
                elif j == 2:
                    cex_count_D2 += 1
        if j == 0:
            f.write('Probability value of S1 is: ' + str(cex_count_S1 / iteration_no) + '\n')
        elif j == 1:
            f.write('Probability value of D1 is: ' + str(cex_count_D1 / iteration_no) + '\n')
        elif j == 2:
            f.write('Probability value of D2 is: ' + str(cex_count_D2 / iteration_no) + '\n\n')

'''
os.system('python Dataframe2XML.py Datasets/dbo_Artist_dbo_Place_dbo_Person_Multi.csv')
model_list = ['Multi_label_classifiers/art_per_pal.joblib', 'Multi_label_classifiers/art_per_pal_RF.joblib']

for job in model_list:
    if 'RF' in job:
        f.write('\n----Results for CR5(RF)-----\n\n')
    else:
        f.write('\n----Results for CR5(NN)-----\n\n')
    for box in white_box:
        if box == 'Decision tree':
            f.write('MLC_DT results-\n')
        else:
            f.write('MLC_NN results-\n')
        cex_count_S1 = 0
        cex_count_D1 = 0
        cex_count_D2 = 0
        for no in range(0, iteration_no):
            propCheck(no_of_params=1, max_samples=1000, model_type='sklearn', model_path=job, mul_cex=False,
            xml_file='dataInput.xml', white_box_model=box, multi_label=True, no_of_class=3, no_of_layers=2, layer_size=10)

            for i in range(0, 3):
                if i == 0:
                    Assume('True')
                    Assert('model.predict(x)[Artist] => model.predict(x)[Person]')
                    dfCexSet = pd.read_csv('CexSet.csv')
                    if dfCexSet.shape[0] == 1:
                        cex_count_S1 += 1

                elif i == 1:
                    Assume('True')
                    Assert('model.predict(x)[Person] => ~model.predict(x)[Place]')
                    dfCexSet = pd.read_csv('CexSet.csv')
                    if dfCexSet.shape[0] == 1:
                        cex_count_D1 += 1
                else:
                    Assume('True')
                    Assert('model.predict(x)[Artist] => ~model.predict(x)[Place]')
                    dfCexSet = pd.read_csv('CexSet.csv')
                    if dfCexSet.shape[0] == 1:
                        cex_count_D2 += 1

        f.write('Probability value of S1 is: '+str(cex_count_S1/iteration_no)+'\n')
        f.write('Probability value of D1 is: ' + str(cex_count_D1 / iteration_no) + '\n')
        f.write('Probability value of D2 is: ' + str(cex_count_D2 / iteration_no) + '\n\n')

    paramDict = {}
    f.write('PBT results are---\n')
    for j in range(0, 3):
        paramDict['classification_type'] = 'multi_label'
        paramDict['no_of_param'] = 1
        paramDict['no_of_class'] = 3
        cex_count_S1 = 0
        cex_count_D1 = 0
        cex_count_D2 = 0
        if j == 0:
            paramDict['property_type'] = 'subsumption'
        elif j == 1:
            paramDict['property_type'] = 'disjunction_artOrAct'
        else:
            paramDict['property_type'] = 'disjunction_person'
        try:
            with open('param_dict.csv', 'w', newline='') as csv_file:
                writer = cv.writer(csv_file)
                for key, value in paramDict.items():
                    writer.writerow([key, value])
        except IOError:
            print("I/O error")
        # actor_person_celestial model evaluation
        cexFlag = False
        execTime = 0
        failed_trials = 0
        # Reading the dataset
        df = pd.read_csv('Datasets/dbo_Artist_dbo_Place_dbo_Person_Multi.csv')
        for i in tqdm(range(iteration_no)):
            cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, job, df)
            if (len(cexPair) >= 1) & (cexFlag == False):
                if j == 0:
                    cex_count_S1 += 1
                elif j == 1:
                    cex_count_D1 += 1
                elif j == 2:
                    cex_count_D2 += 1
        if j == 0:
            f.write('Probability value of S1 is: ' + str(cex_count_S1 / iteration_no) + '\n')
        elif j == 1:
            f.write('Probability value of D1 is: ' + str(cex_count_D1 / iteration_no) + '\n')
        elif j == 2:
            f.write('Probability value of D2 is: ' + str(cex_count_D2 / iteration_no) + '\n\n')
'''
f.close()
