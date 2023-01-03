# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:12:54 2022

@author: Dell
"""

import pandas as pd
import numpy as np
import spacy
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
import text_hammer as th
import re
from nltk.corpus import stopwords
import contractions
from textblob import TextBlob
from sklearn.pipeline import Pipeline
import joblib
import string
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('final_deployment.pkl', 'rb'))

# creating a function for data cleaning
def text_preprocessing(sentence):
    text = []
    text = sentence 
    text = str(text).lower()
    text = th.cont_exp(text)#you're -> you are; i'm -> i am
    text = th.remove_emails(text)
    text = th.remove_html_tags(text)
    text = re.sub('\w*\d\w*','', text) # remove numbers
    text = th.remove_stopwords(text)
    #     df[column] = df[column].progress_apply(lambda x:th.spelling_correction(x))
    text = th.remove_special_chars(text)
    text = th.remove_accented_chars(text)
    text = th.make_base(text) #ran -> run,
    text = re.sub("(.)\\1{2,}", "\\1", text)
    return(text)

# creating a function for Prediction

def Sentiment_analysis(input_data):
    

    # changing the input_data to numpy array
    x = str(input_data)
    var = text_preprocessing(x)

    prediction = loaded_model.predict([var])
    print(prediction)

    if (prediction[0] == 0):
      return 'The review is Negative'
    else:
      return 'The review is Positive'
  
    
  
def main():
    
    
    # giving a title
    st.title('Sentiment Analyzer')
    
    # getting the input data from the user
    review = st.text_input('Enter the review')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Predict'):
        diagnosis = Sentiment_analysis(review)
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()

