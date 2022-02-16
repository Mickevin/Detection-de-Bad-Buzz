import os
import json
import time
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import randrange
from datetime import datetime

import ssl
import nltk
import spacy
import fasttext
import urllib.request
from decouple import config

from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models import LsiModel
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import Word2Vec

import pyLDAvis.gensim_models

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

from keras import backend as K
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import NMF
from sklearn.manifold import TSNE 
from sklearn.preprocessing import LabelEncoder

from sklearn import cluster, metrics, manifold, decomposition
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.compat import v1

from transformers import BertTokenizer, AutoTokenizer
from transformers import TFBertModel
from transformers import pipeline


############   NLP - Traitement du language

# Transformation du label : 0 => Tweet négatif, 1 => tweet positif
def change(x):
    if x == 4: return 1
    else :return 0

    
# Fonction qui transforme le corpus du texte en tokens aves le modul spacy
def traitement_nlp(df_):
    from langdetect import detect
    
    # Import de spacy dans sa versrion "English" et de ses stop-word
    nlp = spacy.load("en_core_web_sm")
    stop_words = nltk.corpus.stopwords.words('english')

    # Yransformation de la feature "text"
    document = df_.text.apply(nlp)

    # Sélection des Tokens non-numérique, non-ponctuation, anglais et n'appartenant pas à la liste de stop word
    tokens = [[u.lemma_ for u in doc if u.is_alpha and not u.is_punct and detect(u.text)=='en' and u.text not in stop_words] for doc in document]
    with open('tokens', 'wb') as f:pickle.dump(tokens, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    # fonction qui renvois le corpus dans une seul et même string
    def corpus(document):
        sentence = ''
        for token in document:
            sentence += str(token) + ' '  
        return sentence
       
    # Création de la feature Token
    df_['tokens'] = np.array(tokens)
    df_['tokens'] = df_.tokens.apply(corpus)
    
    # Création du Bag of word
    vectorizer = CountVectorizer(stop_words='english')
    BOW = vectorizer.fit_transform(df_.tokens).toarray()
    BOW = pd.DataFrame(data=BOW, columns=vectorizer.get_feature_names())

    freq_word = BOW.sum(axis=0).sort_values()[:-50:-1]
    res = dict(zip(BOW.sum(axis=0).index,BOW.sum(axis=0).values))

    # Création du TFIDF
    vec = TfidfVectorizer(stop_words='english')
    tfidf = vec.fit_transform(df_.tokens).toarray()
    tfidf = pd.DataFrame(tfidf, columns=vec.get_feature_names())

    tfidf = tfidf[tfidf.columns]
    
    return tokens, df_, BOW, tfidf, vec

# Fonction qui affiche les topics du corpus 
def showing_topics(df_, tokens, vec, tfidf):

    from wordcloud import WordCloud, STOPWORDS
    
    nlp = spacy.load("en_core_web_sm")
    stop_words = nltk.corpus.stopwords.words('english')
    id2word = Dictionary(tokens)
    bow = [id2word.doc2bow(doc) for doc in tokens]
    model = TfidfModel(bow)
    tfidf_gensim = model[bow] 
    
    docs = []
    comment_words = ''
    for doc in df_.tokens.values:
        comment_words+=doc + " "

    coherence = [CoherenceModel(model=LdaModel(corpus=tfidf_gensim, 
                                               id2word=id2word,
                                               num_topics= n_topic, 
                                               passes=10), 
                                dictionary=id2word,
                                texts = tokens, 
                                coherence='c_v').get_coherence() for n_topic in range(2,15)]

    # Visualisation de la métric de coherence
    plt.figure(figsize=(15,7))
    plt.title('Mesure de cohérence en fonction du nombre de topic')
    plt.scatter(range(2,15),coherence)
    plt.plot(range(2,15),coherence)
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence')
    plt.show()


    stopwords = set(STOPWORDS) 

    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(comment_words) 


    # Affichage du nuage de mot
    plt.figure(figsize = (20, 8), facecolor = None) 
    plt.subplot(1,2,1)
    plt.imshow(wordcloud) 
    plt.axis("off") 

    # Appel de la fonction get_topic afin de détreminer les sujet du corpus
    n = np.array(coherence).argmax()
    lda, X_embedded, tfidf_gensim, id2word, label = get_topic(n, vec, tfidf, tokens)


    # Affichage de la répartion des mots
    plt.subplot(1,2,2)
    plt.title(f'Projection TSNE du corpus en fonction des topic déterminés')
    sns.scatterplot(X_embedded[:,0],X_embedded[:,1], hue=label)
    plt.show()
    
    pyLDAvis.enable_notebook()
    display(pyLDAvis.gensim_models.prepare(lda, corpus=tfidf_gensim, dictionary=id2word))
    
    
# Fonction qui détremine les topics du corpus via LDA et NMF
def get_topic(n, vec, tfidf, tokens):
    # Création de l'instance NMF
    # the 10 components will be the topics
    nmf = NMF(n_components=n, random_state=5)

    # Création des featurs TFIDF
    nmf_features = nmf.fit_transform(tfidf)
    components_df = pd.DataFrame(nmf.components_, columns=np.array((vec.get_feature_names())))

    label = [pd.DataFrame(nmf_features).loc[n].idxmax() for n in range(len(nmf_features))]
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(nmf_features)


    id2word = Dictionary(tokens)
    bow = [id2word.doc2bow(doc) for doc in tokens]
    model = TfidfModel(bow)
    tfidf_gensim = model[bow]

    #Création de l'instance LDA
    lda = LdaModel(corpus=tfidf_gensim, id2word=id2word,num_topics= n, passes=10)
    
    return lda, X_embedded, tfidf_gensim, id2word, label



############################################### Utilisation de Microsoft Azure service
    


def get_sentiment_from_Azure(x):

    def authenticate_client(key,endpoint):
        ta_credential = AzureKeyCredential(key)
        text_analytics_client = TextAnalyticsClient(
                endpoint=endpoint, 
                credential=ta_credential)
        return text_analytics_client
    
    key = config('Azure_nlp_key')
    endpoint = "https://sentimentanalys.cognitiveservices.azure.com/"
    client = authenticate_client(key,endpoint)
    
    return client.analyze_sentiment([x])[0].confidence_scores.values()


def model_five_layers(input_dim, dropout=0.5, regul=0.0):
    kernel_regularizer=regularizers.l2(regul)
    
    model = Sequential()
    model.add(Dense(5, input_dim=input_dim, activation='relu', kernel_regularizer= kernel_regularizer))
    model.add(Dropout(dropout))
    model.add(Dense(3, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    return model

def evaluate(y_true, y_pred,name=None):
    precision_ = precision_score(y_true, y_pred)
    acc_ = accuracy_score(y_true, y_pred)
    recall_ =recall_score(y_true, y_pred)
    f1_ = f1_score(y_true, y_pred)
    df_score = pd.read_csv('utile/df_score.csv',  index_col=0)

    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    ax = sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    
    plt.subplot(1,2,2)
    barWidth = 0.4
    y1 = df_score.loc[0].values[1:-1]
    y2 = [acc_, f1_, recall_, precision_]
    r1 = range(len(y1))
    r2 = [x + barWidth for x in r1]

    plt.bar(r1, y1, width = barWidth, color = ['yellow' for i in y1],
               edgecolor = ['blue' for i in y1])

    plt.bar(r2, y2, width = barWidth, color = ['pink' for i in y1],
               edgecolor = ['green' for i in y1])
    plt.xticks([r + barWidth / 2 for r in range(len(y1))],df_score.columns[1:-1])
    plt.legend([df_score.Model[0],name])
    plt.show()
    
    plt.show()

    return acc_, f1_, recall_, precision_
    


def not_empty(x):
    if x == ' ' or x == '': return 'neutral'
    else : return x
    
    
    
def show_tsne(X,T=False):
    pca = decomposition.PCA(n_components=0.99)
    feat_pca = pca.fit_transform(X)
    print("Dimensions dataset après réduction PCA : ", feat_pca.shape)

    tsne = manifold.TSNE(n_components=2, 
                         perplexity=30, 
                         n_iter=2000, 
                         init='random', 
                         random_state=6)

    X_tsne = tsne.fit_transform(feat_pca)
    df_tsne = pd.DataFrame(X_tsne[:,0:2], columns=['tsne1', 'tsne2'])
    
    if not T:
        
        plt.figure(figsize=(15,7))
        sns.scatterplot(x="tsne1", 
                        y="tsne2", 
                        data=df_tsne)
        sns.color_palette("Set2")

        plt.title('TSNE selon les vraies classes', fontsize = 30, pad = 35, fontweight = 'bold')
        plt.xlabel('Projection tsne1', fontsize = 26, fontweight = 'bold')
        plt.ylabel('Projection tsne2', fontsize = 26, fontweight = 'bold')
        plt.legend(prop={'size': 14}) 
    else :
        df_tsne["class"] = df_.label.values
    
        plt.figure(figsize=(15,7))
        sns.scatterplot(x="tsne1", 
                        y="tsne2", 
                        hue="class", 
                        data=df_tsne)
        sns.color_palette("Set2")

        plt.title('TSNE selon les vraies classes', fontsize = 30, pad = 35, fontweight = 'bold')
        plt.xlabel('Projection tsne1', fontsize = 26, fontweight = 'bold')
        plt.ylabel('Projection tsne2', fontsize = 26, fontweight = 'bold')
        plt.legend(prop={'size': 14})
    plt.show()
    
    
def padding(X, dim, padd_sym):
    X_padd = []
    for x in X:
        if len(x) < 32:
            n = 32 - len(x)
            padd = np.array([padd_sym for n in range(n)])
            x = np.append(x, padd).reshape(dim,32)


        elif len(x) >= 32:
            x = x[:32].reshape(dim,32)
        X_padd.append(np.array(x).reshape(dim,32))
    
    return np.array(X_padd)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def train_plot(model, X_train, y_train, X_test, y_test):
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m])
    model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=0)
    
    loss = model.history.history['loss']
    val_loss = model.history.history['val_loss']
    plt.figure(figsize=(20,9))
    plt.plot(range(len(loss)), loss)
    plt.plot(range(len(val_loss)), val_loss)
    plt.legend(['loss','val_loss'])
    plt.show()

    return evaluate(y_test, (model.predict(X_test) > 0.5).astype("int32"))

def linear_model(data, label, callback, dropout=0.5, regul=0):
    
    model = model_five_layers(data.shape[1], dropout)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.33, random_state=42, stratify=label)

    reg = SGDClassifier().fit(X_train, y_train)
    y_linear_pred = reg.predict(X_test)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'],)
    model.summary()

    # Now fit the model on 500 epoches with a batch size of 64
    # You can add the test/validation set into the fit: it will give insights on this dataset too
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, batch_size=228, callbacks=callback, verbose=0)
    plt_learningcurve(model)
    
    return evaluate(y_test, (model.predict(X_test) > 0.5).astype("int32"))


def bert_vector(x):
    x = tokenizer.encode(x, add_special_tokens=True, return_tensors='tf')
    return tf_bert_model(x)[1].numpy().reshape(-1)

def plt_learningcurve(model):
    loss = model.history.history['loss']
    val_loss = model.history.history['val_loss']
    acc = model.history.history['accuracy']
    val_acc = model.history.history['val_accuracy']

    plt.figure(figsize=(20,9))
    plt.subplot(1,2,1)
    plt.plot(model.history.epoch, acc)
    plt.plot(model.history.epoch, val_acc)
    plt.legend(['Accuracy','Validation_Accuracy'])
    
    plt.subplot(1,2,2)
    plt.plot(model.history.epoch, loss)
    plt.plot(model.history.epoch, val_loss)
    plt.legend(['Loss','Validation_Loss'])
    
    plt.show()
    


def get_model(x):

    def allowSelfSignedHttps(allowed):
        # bypass the server certificate verification on client side
        if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
            ssl._create_default_https_context = ssl._create_unverified_context

    allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

    # Request data goes here
    data = {
        'data': x
    }

    body = str.encode(json.dumps(data))

    url = 'http://0b15f0b2-7c89-4e2d-8036-33a163069b86.westeurope.azurecontainer.io/score'
    api_key = config('Azure_model_avance') # Replace this with the API key for the web service
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()
        #print(result)
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))
        
    result = np.array([float(u.split('[')[-1].split(']')[0]) for u in result.decode().split(', ')])
    return (result > 0.5).astype("int32")