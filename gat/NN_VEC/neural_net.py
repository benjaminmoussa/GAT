
"""
Created on Sat Oct 28 22:53:46 2017

@author: Moussa
"""
import pandas as pd
from pandas import Series, DataFrame
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from ast import literal_eval
import numpy as np
path = '/Users/Moussa/documents/AllSampleVectors.csv'
data = pd.read_pickle(path)


data['c_event']= data['c_event'].apply(lambda x: tuple(x))

#split train test
x_train = DataFrame()
y_train = DataFrame()
x_test = DataFrame()
y_test = DataFrame()
y_set = set(data['c_event'])
data['c_event'] = data['c_event'].astype(object)
data['ex'] = data['ex'].astype(object)




#for x in y_set:
#    temp_df = data[data['c_event'] == x]
#    train, test = train_test_split(temp_df, test_size=0)
#    x_test = pd.concat([x_test, test['ex']], axis=0)
#    y_test = pd.concat([y_test, test['c_event']], axis=0)
#    x_train = pd.concat([x_train, train['ex']], axis=0)
#    y_train = pd.concat([y_train, train['c_event']], axis=0)

train, test = train_test_split(data, test_size=.2)
x_train = DataFrame(train['ex'])
y_train = DataFrame(train['c_event'])
x_test = DataFrame(test['ex'])
y_test = DataFrame(test['c_event'])
x_train.columns = ['data']
y_train.columns = ['data']
x_test.columns = ['data']
y_test.columns = ['data']
#x_train['data'] = x_train['data'].astype(object)
#y_train['data'] = y_train['data'].astype(object)

#x_test['data'] = x_test['data'].astype(object)
#y_test['data'] = y_test['data'].astype(object)

#train
x = np.array(x_train)
y = np.array(y_train)

x_ftr = []
y_ftr = []
clf = MLPClassifier()
for z in range(len(x)):
    x_ftr.append(x[z][0])
    y_ftr.append(y[z][0])
clf.fit(x_ftr,y_ftr)

#test
x1 = np.array(x_test)
y1 = np.array(y_test)

x_ft = []
y_ft = []
for z in range(len(x1)):
    x_ft.append(x1[z][0])
    y_ft.append(y1[z][0])
    
#get predictions for test
lst = []
predict = clf.predict(x_ft)
for x in predict:
    lst.append(tuple(x))
pred = Series(lst)
actual = Series(y_ft)

tests = pd.concat([pred, actual], axis=1)
tests.columns = ['pred', 'actual']
#get result array with binary values
same_diff = []
for index, row in tests.iterrows():
    if row['pred'] == row['actual']:
        same_diff.append(1)
    else:
         same_diff.append(0)
print(same_diff.count(1)/len(same_diff))
        