#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 01:48:29 2019

@author: russell
"""

import csv
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import nltk
import numpy as np
from deep_translator import GoogleTranslator
import pandas as pd

nlp_english = spacy.load('en_core_web_sm')
# nlp_english = spacy.load('en_core_web_trf')
nlp_english.add_pipe('spacytextblob')

from nltk.stem import PorterStemmer

dic = {}

class LexicalAnalyzer(object):
    
    def __init__(self, data_):
        # self.create_polarity_dictionary_opinion_lexicon()
        self.data_ = data_
        

    def preprocess_data(self,text):

        text = text.strip().replace("\n", " ").replace("\r", " ")
        text = text.strip().replace(".", ".").replace(".", ".")
        text = text.lower()
        return text


    def stem_data(self, text):
        ps = PorterStemmer() 
        modified_text = " ".join(ps.stem(w)for w in nltk.wordpunct_tokenize(text))  

        return modified_text

    def split_review_text(self,review):
        import re
        split_sentences = review.split("�")
        split_sentences  = re.split('[.,]', review) # re.split('[^a-zA-Z][]', review)
        sentences = []
   
        for s in split_sentences:
            if len(s) > 1:
                sentences.append(s)
       
        return sentences

    def remove_pronoun(self,tokens):
        pronouns = ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'who','me', 'him', 'her', 'it', 'us', 'you', 'them', 'whom','mine', 'yours', 'his', 'hers', 'ours', 'theirs','this', 'that', 'these', 'those']
    
        for token in tokens:
            for pronoun in pronouns:
                if token == pronoun:
                    #print("^^^^^^^^\:   ", token, pronoun)
                    tokens.remove(token)
                    break
                
                
        return tokens

    #----------- Check whether text contains adjective and adverb --------------                
    def does_contain_adjective(self, tokens):
        
        text = ' '.join(tokens)

        tokens = nlp_english(text)
        for token in tokens:
            if token.pos_ == 'ADJ' or token.pos_ == 'ADV':
                return True
     
        return False

    #----------- Check polarity Shifter due to Negation  --------------  
    def get_negation_score(self,tokens):
      
        text = ' '.join(tokens)
 
        tokens = nlp_english(text)
        for i in range(len(tokens) - 2):
            if tokens[i].dep_ == 'neg' and  (tokens[i + 1].pos_ == 'ADJ' or tokens[i + 2].pos_ == 'ADJ' ):# or token.text == 'should' or token.text == 'could' or token.text == 'must' :
               
               if tokens[i + 1].pos_ == 'ADJ':
                   token = tokens[i + 1].text
               else:
                   token =  tokens[i + 2].text
               
               if token in  dic.keys(): 
                   return  - 2 * dic[token]
               
        #not good, do not like, not terrible    
        for i in range(len(tokens) - 1):
             if tokens[i].dep_ == 'neg' and  (tokens[i + 1].pos_ == 'ADJ'):# or token.text == 'should' or token.text == 'could' or token.text == 'must' :
               
                token = tokens[i + 1].text
              
                if token in  dic.keys(): 
                   #print("^^^^^ ^^^ ^ ^^  ^^ " , token, dic[token])
                   return  - 2 * dic[token]
        return 0
    
    
    #----------- Check presence of Comparison in sentence  --------------  
    def get_comparison_score(self,tokens):
        text = ' '.join(tokens)
        #print(text)
        tokens = nlp_english(text)
        for i in range(len(tokens) - 2 ):
            token = tokens[i]
            next_token = tokens[i + 2]
            #could/should/must be ?,  negate the ?
            if token.text == 'should' or token.text == 'could' or token.text == 'must' :
                if next_token in dic.keys(): 
                    return   dic[next_token] * -1
        return 0
    
    
    #----------- Bing Liu opinion Lexicon --------------  
    def create_polarity_dictionary_opinion_lexicon(self):
        file = open('../SSentiA_1/positive.txt', 'r') 
        for line in file: 
            token = line.split()
            key = ''.join(token)
            dic[key] = 1
            
        file = open('../SSentiA_1/negative.txt', 'r') 
        for line in file: 
            token = line.split()
            key = ''.join(token)
            dic[key] = -1
      
    
    def get_polarity_score(self, aspect_sentence):
        total_sum = 0
        positive_score = 0
        negative_score = 0
        ddd = 0
        for token in aspect_sentence:
            if token in dic.keys(): 
                total_sum += dic[token]
                if dic[token] == 1:
                    positive_score += dic[token]
                elif dic[token] == -1:
                    negative_score  -= dic[token]
                else:
                    ddd = 0
        
        return total_sum, positive_score, negative_score

    
    def remove_text_index(self,text):
        
        text = text.strip()
        index = text.find(":")
        text = text [index + 1:]
        
        return text
    

    def classify_binary_dataset(self, X_data, Y_label):
        
        num_of_detection = 0
        true_prediction = 0
        false_prediction = 0
       
        prediction_confidence_scores = []
        
        predictions = []
        i = 0
       
        Y_label = Y_label.astype('int')
        
       
        for user_review in X_data:
            if len(user_review) > 4990:
                user_review = user_review[:4990]
            
            user_review = GoogleTranslator(source='spanish',target='english').translate(user_review)
            #print(i, user_review)
            if len(str(user_review)) < 5:
                i += 1
                prediction_confidence_scores. append(-1)
                print("\n\n\n\n: Less: ",user_review, "\n\n\n\n" )
                continue;
            
            sentiments = []
            
            user_review = self.preprocess_data(user_review)
            user_review = self.split_review_text(user_review)
            
            
            total_score = 0
            total_aspect_term = 0
            total_positive_score = 0
            total_negative_score = 0
            
            for sentence in user_review:  
                # sentence = GoogleTranslator(source='spanish',target='english').translate(sentence).lower()
                # try:
                #     sentence = GoogleTranslator(source='spanish',target='english').translate(sentence)
                # except:
                #     sentence = " "
                        
                # if sentence is None:
                #     sentence = " "
                    
                score = nlp_english(sentence)._.polarity
            
                if score > 0:
                    total_positive_score += score
                elif score < 0:
                    total_negative_score += abs(score)
                    
                total_score += score
                
            
            
 
            predicted_label = 0
            true_label = int(Y_label[i])
        
            if total_score >= 0:
                predicted_label = 1 
                
        
            total_positive_negative = total_positive_score + total_negative_score
            
            # print(i, total_positive_score, total_negative_score, total_positive_negative, total_score)
            
            if total_score != 0:
                #if total_positive_score >= total_negative_score:
                prediction_confidence_score =  float(abs(total_positive_score - total_negative_score)/total_positive_negative)
            else:
                prediction_confidence_score = 0
            prediction_confidence_scores.append(prediction_confidence_score)
            
           
            if predicted_label == true_label:
                true_prediction += 1
            else:
                false_prediction += 1
                for aspect_sentence in sentiments:
                    score = self.get_polarity_score(aspect_sentence)
                    
            # print(i, total_score, predicted_label, true_label)
            predictions.append(predicted_label)
                
            
            i += 1
     
        #print("\n!!!!!!",len(Y_label), len(predictions))
        
        from supervisedalgorithm import  Performance
        print("Prediction ------")
        performance = Performance() 
        conf_matrix, f1_score, precision,  recall,acc = performance.get_results(Y_label, predictions)
        print("----",round(f1_score,4), round(precision,4),  round(recall,4), round(acc,4) )
    
        #print(conf_matrix)
        
        return predictions, prediction_confidence_scores
       
        
        #write_label_prediction_to_file(X_data,  Y_label , prediction, prediction_confidence_scores, "/Users/russell/Downloads/results_process/" + current_dataset + "/" + str(process_id) + ".txt")
        
        #write_label_prediction_to_file(X_data,  Y_label , prediction, prediction_confidence_scores, "/Users/russell/Downloads/dvd/" + str(process_id) + ".txt")
           
        #write_label_prediction_to_file(X_data,  Y_label , prediction, prediction_confidence_scores, "/Users/russell/Downloads/electronics/" + str(process_id) + ".txt")
           
          

    
    