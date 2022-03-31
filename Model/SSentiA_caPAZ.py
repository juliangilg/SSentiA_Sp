#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:12:50 2020

@author: russell
"""
from supervisedalgorithm import Logistic_Regression_Classifier, SVM_Classifier, AdaBoost_Classifier, KNN_Classifier, RandomForest_Classifier, MultinomialNB_Classifier

from supervisedalgorithm import  Performance
from supervisedalgorithm  import TF_IDF
from pandas import read_excel
import numpy as np
from deep_translator import GoogleTranslator
import pandas as pd

class sSentiA_Sp(object):
    
    def __init__(self, name):
        self.name = name
        return None
        

    def get_data_n_label_n_predcition(self, data):
    
              
        content = data.values
    
        X = content[:,0]
        Y = content[:,1]
        Z = content[:,2]
    
        
        Y = Y.astype('int')
        Z = Z.astype('int')
        
      
        return X, Y

#------------SsentiA----used in the paper-------  
    def apply_SSSentiA(self, df1, df2, df3, df4, df5): 
     
                           
        # Very-high Confidence Group (Bin 1)
        X_1, Z_1 = self.get_data_n_label_n_predcition(df1)
        
        
        # High Confidence Group (Bin 2)
        X_2, Z_2 = self.get_data_n_label_n_predcition(df2)
        
        # Low Confidence Group (Bin 3)
        X_3, Z_3 = self.get_data_n_label_n_predcition(df3)
        
        # very-Low Confidence Group (Bin 4)
        X_4, Z_4 = self.get_data_n_label_n_predcition(df4)
       
        # Zero Confidence Group  (Bin 5)
        X_5, Z_5 = self.get_data_n_label_n_predcition(df5)
        
                   
        
        ml_classifier = Logistic_Regression_Classifier() 
        ml_classifier = SVM_Classifier()
        ml_classifier = RandomForest_Classifier()
        # # ml_classifier = AdaBoost_Classifier()
        lr_classifier = RandomForest_Classifier()
        ml_classifier = MultinomialNB_Classifier()
        # lr_classifier = ExtraTree_Classifier()
        # ml_classifier = xgboost()
        
        bin_size_1_2 = len(X_1) + len(X_2) # + len(X_3) #+  len(X_01) + len(X_02) + len( X_11) + len(X_12)
        # print("---",bin_size_1_2)
        
        
        data = np.concatenate((X_1,X_2,X_3), axis=None)
        label = np.concatenate((Z_1,Z_2,Z_3), axis=None)
    
      
        
        tf_idf = TF_IDF()
        data = tf_idf.get_tf_idf(data)
        
        X_train = data[:bin_size_1_2]
        Y_train = label[:bin_size_1_2]
        
        X_test = data[bin_size_1_2:]
        Y_test = label[bin_size_1_2:]
        
    
        
        prediction_bin_3 = ml_classifier.predict(X_train, Y_train, X_test)
        
        # print("Bin-3 Results")
        # performance = Performance()
        # _,precision,  recall, f1_score, acc = performance.get_results(Y_test, prediction_bin_3)
        # print("Total: ", round(precision,4),  round(recall,4), round(f1_score,4),round(acc,4) )
        
    
        data = np.concatenate((X_1,X_2,X_3,X_4,X_5), axis=None)
        label = np.concatenate((Z_1,Z_2,prediction_bin_3,Z_4,Z_5), axis=None)
        
        
        tf_idf = TF_IDF()
        data = tf_idf.get_tf_idf(data)
    
        bin_1_2_3_training_data = len(X_1) + len(X_2) + len(X_3)  
        
        X_train = data[:bin_1_2_3_training_data]
        Y_train = label[:bin_1_2_3_training_data]
        
        X_test = data[bin_1_2_3_training_data:]
        Y_test = label[bin_1_2_3_training_data:]
        
     
        # print("Bin-4results")
        prediction_bin_4_5 = ml_classifier.predict(X_train, Y_train, X_test)
        # _,precision,  recall, f1_score, acc = performance.get_results(Y_test[:len(X_4)], prediction_bin_4_5[:len(X_4)])
        # print("F1: ", round(precision,4),  round(recall,4), round(f1_score,4),round(acc,4) )
         
    
        # print("Bin 5 results")
        # _,precision,  recall, f1_score, acc = performance.get_results(Y_test[len(X_4):], prediction_bin_4_5[len(X_4):])
        # print("F1: ", round(precision,4),  round(recall,4), round(f1_score,4),round(acc,4) )
    
        
        
        # true_label = np.concatenate((Y_1,Y_2,Y_3,Y_4,Y_5), axis=None)
        all_prediction = np.concatenate((Z_1,Z_2,prediction_bin_3, prediction_bin_4_5), axis=None)
        
        # print("\nOverall Predcition of SSSentiA")
        #precision,  recall, f1_score, acc = performance.get_results(true_label, all_prediction)
        # M, precision,  recall, f1_score, acc = performance.get_results(true_label, all_prediction)
        # print("Overall: ", round(precision,4),  round(recall,4), round(f1_score,4),round(acc,4) )
        # print(M)
        # D_ = pd.DataFrame(all_prediction, columns=['label'])
        # D_.to_excel(dataset + ".xlsx", header=False)
        print(self.name)
        print(all_prediction.sum()/len(all_prediction))
  
