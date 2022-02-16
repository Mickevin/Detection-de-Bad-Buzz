import joblib
import json
import numpy as np
import os
import re
import gensim
import tensorflow as tf
from gensim.corpora import Dictionary
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing import sequence
from azureml.core.model import Model


def init():
    global model
    global dct
    
    keras_path = Model.get_model_path(model_name = 'Model_deep')
    dct_path = Model.get_model_path(model_name = 'dct')

    try:
        dct = joblib.load(dct_path)
    except:
        print("dct not work")
        
    try: 
        model = load_model(keras_path)
    except:
        print("Keras model not working")

        
def run(raw_data):
    #keras_path = Model.get_model_path(model_name = 'Model_deep')
    #model = load_model(keras_path)
    text = json.loads(raw_data)["data"]
    
    X = return_index(text)
    X = sequence.pad_sequences(X,
                                value=0,
                                padding='post', # to add zeros at the end
                                truncating='post', # to cut the end of long sequences
                                maxlen=32) # the len
    return model.predict(X).tolist()



def return_index(data):
    dct_path = Model.get_model_path(model_name = 'dct')
    dct = joblib.load(dct_path)
    keys = dct.token2id.keys()
    
    tokens = []
    document = []

    for X in data:
        for x in X.lower().split():
            if x in list(keys):
                tokens.append(dct.token2id[x])
        document.append(tokens)
    return document