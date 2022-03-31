#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 01:48:29 2019

@author: russell
"""

import csv
import spacy
import nltk
import numpy as np
import pandas as pd
from collections import Counter
import re

nlp_english = spacy.load('es_core_news_sm')
from nltk.stem import PorterStemmer

dic = {}
list_a = ['debería', 'debe', 'deberia', 'podría', 'podria', 'tiene',
          'sería', 'seria', 'ser']

class LexicalAnalyzer(object):
    
    def __init__(self, data_):
        self.create_polarity_dictionary_opinion_lexicon()
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
        #print(len(sentences))
        return sentences

    def remove_pronoun(self,tokens):
        pronouns1 = ['yo','mi', 'conmigo','tú','contigo','conmigo','vos','él','ella','ello',
                    'usted','el','sí','consigo','nosotros','nosotras','vosotros','vosotras','ellos',
                    'ellas','ustedes','me','nos','te','os','lo','la','le','se','los','las','les',
                    'mío','mio','tuyo','suyo','mía','mia','tuya','suya','míos','mios','tuyos','suyos',
                    'mías','mias','tuyas','suyas','nuestro','vuestro','nuestra','vuestra',
                    'nuestros','vuestros','nuestras','vuestras','este','ese','aquel',
                    'esta','esa','aquella','esto','eso','aquello','estas','esas','aquellas','estos','esos','aquellos']
        pronouns2 = ['yo', 'tú', 'tu' 'el', 'ella', 'esto', 'nosotros', 'ellos', 'lo que', 'quién', 'quien','mio', 'suyo', 'nuestro', 'usted', 'de ellos', 'que','mi', 'los', 'eso', 'esto', 'aquello', 'esos', 'estos', 'aquellos', 'aquel']
        pronouns = pronouns1 + pronouns2
        
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
            if token.pos_ == 'ADJ' or token.pos_ == 'ADV' or token.pos_ == 'NOUN':
                return True
            
            
        return False

    #----------- Check polarity Shifter due to Negation  --------------  
    def get_negation_score(self,tokens):
      
        text = ' '.join(tokens)
 
        tokens = nlp_english(text)
        for i in range(len(tokens) - 2):
            if (tokens[i].dep_ == 'neg' or tokens[i].text =='no' 
                or tokens[i].text in list_a) and  (tokens[i + 1].pos_ == 'ADJ' 
                or tokens[i + 1].pos_ == 'VERB' or tokens[i - 1].pos_ == 'ADJ' 
                or tokens[i - 1].pos_ == 'VERB' or tokens[i + 2].pos_ == 'ADJ' 
                or tokens[i + 2].pos_ == 'NOUN', tokens[i + 2].pos_ == 'ADV'):# or token.text == 'should' or token.text == 'could' or token.text == 'must' :
            # if (tokens[i].dep_ == 'neg' or tokens[i].text in list_a) and  (tokens[i + 1].pos_ == 'ADJ' or tokens[i + 1].pos_ == 'VERB' or tokens[i + 2].pos_ == 'ADJ' or tokens[i + 2].pos_ == 'NOUN'):
               if tokens[i + 1].pos_ == 'ADJ' or  tokens[i + 1].pos_ == 'VERB':
                   token = tokens[i + 1].text
               elif tokens[i - 1].pos_ == 'ADJ' or  tokens[i - 1].pos_ == 'VERB':
                   token = tokens[i - 1].text
               else:
                   token =  tokens[i + 2].text
               
               if token in  dic.keys(): 
                   return  - 2 * dic[token]
               elif tokens[i + 2].pos_ == 'ADJ' or tokens[i + 2].pos_ == 'NOUN':
                   if tokens[i + 2].text in  dic.keys():
                       return  - 2 * dic[tokens[i + 2].text]
               
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
            if token.text == 'debería' or token.text == 'debe' or token.text == 'deberia' or token.text == 'podría' or token.text == 'podria' or token.text == 'tiene' or token.text == 'puede' or token.text == 'sería' or token.text == 'es' or token.text == 'seria':
                if next_token in dic.keys(): 
                    return   dic[next_token] * -1
        return 0
    
    
    #----------- Bing Liu opinion Lexicon --------------  
    def create_polarity_dictionary_opinion_lexicon(self):
        data = pd.read_csv('../Dic/positive_es.txt',names=['Word'])
        data.insert(1, "Polarity", np.ones(data.shape[0]), True)
        data1 = pd.read_csv('../Dic/negative_es.txt',names=['Word'])
        data1.insert(1, "Polarity", -1*np.ones(data1.shape[0]), True)
        frames = [data, data1]
        result = pd.concat(frames)
        for i in range(result.shape[0]):
            dic[result.iloc[i,0]] = result.iloc[i,1]
        # dic = dict(zip(result['Word'].values, result['Polarity'].values))
        
        # file = open('../SSentiA/positive_es.txt', 'r') 
        # for line in file: 
        #     token = line.split()
        #     key = ''.join(token)
        #     dic[key] = 1
            
        # file = open('../SSentiA/negative_es.txt', 'r') 
        # for line in file: 
        #     token = line.split()
        #     key = ''.join(token)
        #     dic[key] = -1
      
    
    def get_polarity_score(self, aspect_sentence):
        total_sum = 0
        positive_score = 0
        negative_score = 0
        ddd = 0
        text = ' '.join(aspect_sentence)
        if re.search('no es',text):
            neg = 1
        else:
            neg = 0
            
        for token in aspect_sentence:
            if token in dic.keys(): 
                total_sum += dic[token]
                if dic[token] == 1:
                    aux = 1
                    positive_score += dic[token]
                elif dic[token] == -1:
                    aux = -1
                    negative_score  -= dic[token]
                else:
                    ddd = 0
        
        # if neg == 1:
        #     if aux == 1:
                
        
        # if re.search('no es',text):
        #     negative_score  += 1
        
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
       
        # Y_label = Y_label.astype('int')
        
       
        for user_review in X_data:
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
                tokens = nlp_english(sentence)
                #print(">> ",tokens)
                
                aspect_sentence = []
               
             
                for token in tokens:   
                    # if not token.is_stop:
                    #if  token.dep_ == 'nsubj' or  token.dep_ == 'amod' or token.pos_ == 'ADJ':
                    if token.dep_ == 'nsubj' or token.dep_ == 'neg'  or token.dep_ == 'advmod' or token.dep_ == 'ROOT' or token.dep_ == 'compound' or token.pos_ == 'ADJ' or token.pos_ == 'NOUN' or token.text in list_a:
                       #print(token.text, token.dep_,  token.pos_,  [child for child in token.children])
                       aspect_sentence.append(token.text)
                
                #print("aspect_sentence",aspect_sentence)
                if len(aspect_sentence) >= 2:
                    num_of_detection += 1
                
                    sentiments.append(aspect_sentence)
                    
                    aspect_sentence = self.remove_pronoun(aspect_sentence)
                    aspect_sentence = list(Counter(aspect_sentence).keys()) 
                    
                    if self.does_contain_adjective(aspect_sentence) == True:
                    
                        
                        score, positive,negative = self.get_polarity_score(aspect_sentence)
                        #print("-- ",i, score, positive,negative)
                        
                        total_positive_score += positive
                        total_negative_score += negative
                       
                        negation_score = self.get_negation_score(aspect_sentence)
                        if abs(negation_score) > 0 :  
                            score  +=  negation_score
                            if negation_score < 0:
                                total_positive_score -= 1
                                total_negative_score += 1
                            else:
                                total_negative_score -= 1
                                total_positive_score += 1
                                
                            #print ("Neg: ",score,  aspect_sentence, negation_score)
                        total_score += score
                        total_score += self.get_comparison_score(aspect_sentence)
                        total_negative_score -=  self.get_comparison_score(aspect_sentence)
                  
                total_aspect_term += len(aspect_sentence)

       
            predicted_label = 0
            true_label = int(Y_label[i])
        
            if total_score >= 0:
                predicted_label = 1 
                
        
            total_positive_negative = total_positive_score + total_negative_score
            
            #print(i, total_positive_score, total_negative_score, total_positive_negative, total_score)
            
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
                    
            predictions.append(predicted_label)
                
            
            i += 1
        Y_label = [y for y in Y_label]
        #print("\n!!!!!!",len(Y_label), len(predictions))
        #print("\n!!!!!!",(Y_label), (predictions))
        predictions = np.asarray(predictions,dtype=int)
        from sklearn.metrics import confusion_matrix
        conf_matrix = confusion_matrix(Y_label, predictions)
        #print("\n!!!!!!",len(Y_label), len(predictions))
        #print("\n!!!!!!",(Y_label), (predictions))
        
        from supervisedalgorithm import  Performance
        print("Prediction ------")
        performance = Performance() 
        conf_matrix, f1_score, precision,  recall,acc = performance.get_results(Y_label, predictions)
        print("----",round(f1_score,4), round(precision,4),  round(recall,4), round(acc,4) )
        
        return predictions, prediction_confidence_scores
       
        
        #write_label_prediction_to_file(X_data,  Y_label , prediction, prediction_confidence_scores, "/Users/russell/Downloads/results_process/" + current_dataset + "/" + str(process_id) + ".txt")
        
        #write_label_prediction_to_file(X_data,  Y_label , prediction, prediction_confidence_scores, "/Users/russell/Downloads/dvd/" + str(process_id) + ".txt")
           
        #write_label_prediction_to_file(X_data,  Y_label , prediction, prediction_confidence_scores, "/Users/russell/Downloads/electronics/" + str(process_id) + ".txt")
           
          

    
    