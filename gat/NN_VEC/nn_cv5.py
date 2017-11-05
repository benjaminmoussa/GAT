#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 13:20:49 2017

@author: Moussa
"""

import pandas as pd
from pandas import Series, DataFrame
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from ast import literal_eval
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import vectorize as vc
import pickle


class neural_net:
    def __init__(self, path):
        self.data = DataFrame(pd.read_pickle(path))
        self.data['c_event'] = data['c_event'].astype(object)
        self.data['ex'] = data['ex'].astype(object)
        self.model =  pickle.load(open("model", "rb"))
        #self.x = np.array(data['ex'])
        #self.y = np.array(data['c_event'])
        self.vecDict = vc.createVectorCameo(self.data)
        self.bag = Series(vc.makeBag('cameo.csv')).tolist()
        y = []
        x = []
        for index, row in self.data.iterrows():
            x.append(row['ex'])
            y.append(row['c_event'])
        x = np.array(x)
        y = np.array(y)
        self.x = x
        self.y = y
    def trainNN(self):
        self.model.fit(self.x,self.y)
        return self.model
    def predict(self, string):
        word_lst = vc.createWord_List(string)
        print(word_lst)
        vector = vc.createVector(self.bag, word_lst)
        #vector = np.array(vector)
        vector = np.array(vector)
        vector = vector.reshape(1,1167)
        pred = self.model.predict(vector.reshape(1,-1))
        top5 = self.model.predict_proba(vector)
        return [string, vector, pred, top5]
    def cross_validate(self, folds =5):
        clf = MLPClassifier()
        acc = cross_val_score(clf, x, y, cv=5)
        return acc
    
        
nn_ob = neural_net('AllSampleVectors.csv')   
#md = nn_ob.trainNN()
#pickle.dump(md, open("model", "wb")) 
#bag = nn_ob.bag
string = "While Mattis did not go that far, his tough talk is in line with Trump, who wants NATO members to adopt a plan to boost military funding to 2 percent of each country's" 
preds = nn_ob.predict(string)
        
       
    
    
        









