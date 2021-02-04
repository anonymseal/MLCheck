
import os
import numpy as np
import pandas as pd
import csv as cv



def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def funcWrite(MAX_SAMPLES, df, file_name, classification_type, no_of_param, no_of_class, property_type):
                    
    if(classification_type == 'multi_label'):
        noOfFe = df.shape[1]-no_of_class
    else:
        noOfFe = df.shape[1]-1
   
    f = open('quickCheck.py', 'w')
    f.write('from hypothesis.strategies import tuples, floats, lists, integers \n')
    f.write('from hypothesis import settings, seed, HealthCheck, given, assume, Verbosity \n')
    f.write('import time \n')
    f.write('from joblib import dump, load')
    f.write('\n')
    f.write('from MainFiles import monIndexArr')
    f.write('\n \n')

    f.write('@settings(max_examples='+str(MAX_SAMPLES)+', suppress_health_check=HealthCheck.all(), deadline = None)\n')
    #f.write('@settings(deadline = None, verbosity=Verbosity.verbose)\n')
    f.write('@given(')
     
    for i in range(0, no_of_param):
        f.write('tuples(')
        for j in range(0, noOfFe):
            data_type = str(df.dtypes[j])
            min_val = str(df.iloc[:, j].min())
            max_val = str(df.iloc[:, j].max())
            if('int' in data_type):
                f.write('integers(min_value='+min_val+', max_value='+max_val)
            elif('float' in data_type):
                f.write('floats(min_value='+min_val+', max_value='+max_val)
            if(j == noOfFe-1):
                f.write(')')
            else:
                f.write('),')
        if(i == no_of_param-1):
            f.write(')')
        else:
            f.write('),')
                    
    f.write(')')
    f.write('\n\n')
    if(no_of_param == 1):
        f.write('def check_prop(x):\n')
    else:
        f.write('def check_prop(x, y):\n')
    f.write(' model = load('+f'"{file_name}"'+')')
    f.write('\n')
    #f2.write(' start_time = time.time()\n')
    if(classification_type == 'multi_label' and property_type == 'subsumption'):
        f.write(' pred_arr = monIndexArr.predict(model, list(x)) \n')
        f.write(' Actor_Artist = pred_arr[0][1] \n')
        f.write(' Person = pred_arr[0][0] \n')
        #f.write(' print(\'Actor_Artist is\', Actor_Artist) \n')
        #f.write(' print(\'Person is\', Person) \n')
        f.write(' assert((Actor_Artist != 1) or (Person == 1)) \n')
    elif(classification_type == 'multi_label' and property_type == 'disjunction_artOrAct'):
        f.write(' pred_arr = monIndexArr.predict(model, list(x)) \n')
        f.write(' Actor_Artist = pred_arr[0][1] \n')
        f.write(' Place_Planet_Celest = pred_arr[0][2] \n')
        #f.write(' print(\'Actor/Artist is\', Actor_Artist) \n')
        #f.write(' print(\'Place/Planet/Celest is\', Place_Planet_Celest) \n')
        f.write(' assert((Place_Planet_Celest != 1) or (Actor_Artist == 0)) \n')
    elif(classification_type == 'multi_label' and property_type == 'disjunction_person'):
        f.write(' pred_arr = monIndexArr.predict(model, list(x)) \n')
        f.write(' Person = pred_arr[0][0] \n')
        f.write(' Place_Planet_Celest = pred_arr[0][2] \n')
        #f.write(' print(\'Person is\', Person) \n')
        #f.write(' print(\'Place/Planet/Celest is\', Place_Planet_Celest) \n')
        f.write(' assert((Place_Planet_Celest != 1) or (Person == 0)) \n')
    
    else:
        f.write(' index_arr = monIndexArr.mon_indxArr()\n')
        f.write(' for i in range(0, len(index_arr)):\n')
        f.write('  for j in range(0, '+str(df.shape[1]-1)+'):\n')
        f.write('   if(int(index_arr[i]) == j):\n')
        f.write('    assume(x[int(index_arr[i])] != y[int(index_arr[i])])\n')
        f.write('   else:\n')
        f.write('    assume(x[int(index_arr[i])] == y[int(index_arr[i])])\n')
        f.write(' assert(monIndexArr.predict(model, list(x)) == monIndexArr.predict(model, list(y)))\n')

    f.write('check_prop()\n')
    f.close()


# In[14]:


def funcCountEx(total_count):
    
    attempt_count = 0
    with open('Output.txt') as f1:
        file_content = f1.readlines()
    
    file_content = [x.strip() for x in file_content]
    return file_content    


# In[15]:


def funcMain(MAX_SAMPLES, file_name, df):
    with open('param_dict.csv') as csv_file:
        reader = cv.reader(csv_file)
        paramDict = dict(reader)
    funcWrite(MAX_SAMPLES, df, file_name, paramDict['classification_type'], int(paramDict['no_of_param']), int(paramDict['no_of_class']), paramDict['property_type'])
    os.system(r"python quickCheck.py > Output.txt")
    cexPair = funcCountEx(MAX_SAMPLES)
	
	
    return cexPair
        


# In[16]:



