#%% import libraries
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from collections import defaultdict
import random
from gensim.models import Word2Vec
from textblob import TextBlob
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

#%% load data
# load and print the data
train_data = pd.read_csv('../../Classification_ data/train.csv')
test_data = pd.read_csv('../../Classification_ data/test.csv')

#%% preprocess (not applied yet!)
def preprocess_text(text):
    
    if isinstance(text, str):  # Check if the input is a string
        # Step 1: Convert text to lowercase
        text_lower = text.lower()

        # Step 2: Remove punctuation
        text_no_punct = ''.join([char for char in text_lower if char not in string.punctuation])

        # Step 3: Tokenize the lowercased text
        tokens = word_tokenize(text_no_punct)

        # Step 4: Remove stop words
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]

        # Step 5: Perform stemming
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
        
        # Combine stemmed tokens back into a string
        preprocessed_text = ' '.join(stemmed_tokens)
        
        return preprocessed_text
    
    else: return text

# As the data is already pre-processed, I skip applying the pre-processed function
train_set_processed = train_data.copy()
test_set_processed = test_data.copy()

#%% label - data split
train_label = train_set_processed['Class']
test_label = test_set_processed['Class']

train_set_text = train_set_processed['text']
test_set_text = test_set_processed['text']
#%% feature extraction stage (n-gram + cluster with count vectorizer)
# n-gram 1-3 feature extraction
vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=1000)
train_set_vectorized = vectorizer.fit_transform(train_set_text).toarray()
test_set_vectorized = vectorizer.transform(test_set_text).toarray()

# apply word cluster technique
word_clusters = {}

def loadwordclusters():
    infile = open('./50mpaths2')
    for line in infile:
        items = str.strip(line).split()
        class_ = items[0]
        term = items[1]
        word_clusters[term] = class_
    return word_clusters

def getclusterfeatures(sent):
    terms = nltk.word_tokenize(sent)
    cluster_string = ''
    for t in terms:
        if t in word_clusters.keys():
                cluster_string += 'clust_' + word_clusters[t] + '_clust '
    return str.strip(cluster_string)

word_clusters = loadwordclusters()

test_clusters = []
training_clusters = []

for tr in train_set_text:
    training_clusters.append(getclusterfeatures(tr))
for tt in test_set_text:
    test_clusters.append(getclusterfeatures(tt))

clustervectorizer = CountVectorizer(ngram_range=(1,3), max_features=1000)

training_cluster_vectors = clustervectorizer.fit_transform(training_clusters).toarray()
test_cluster_vectors = clustervectorizer.transform(test_clusters).toarray()

#%% feature extraction Word2Vec Models
import gensim.downloader as api
print(list(api.info()['models'].keys()))

wv1 = api.load('word2vec-google-news-300')
wv2 = api.load('glove-wiki-gigaword-300')

def w2v (texts, model):
    # receives a list of sentences and embedding model: Word2Vec, gLoVe as input
    # outputs word2vec representation for each sentence in an array of size n*d 
    # where n: number of sentences, d: dimension of representation in w2v model
    
    # number of sentences
    n = len(texts)
    # dimension of representation
    d = 300
    # w2v representation
    w2v_rep = np.zeros((n, d)) 
    for i in range(n):
        # word tokenize
        words = word_tokenize(texts[i])
        # calculate mean w2v representatoin for the words of the sentence
        w2v_rep[i, :] = np.mean([model[word] for word in words if word in model], axis=0)
    return w2v_rep

# w2v 
X_train_w2v = w2v(train_set_text, wv1)
X_test_w2v = w2v(test_set_text, wv1)

# glove 
X_train_glove = w2v(train_set_text, wv2)
X_test_glove = w2v(test_set_text, wv2)

#%% other feature sets
# extract the lenght of texts as a feature set
def text_length(texts):
    return np.array([len(text) for text in texts]).reshape(-1, 1)

X_train_length = text_length(train_set_text)
X_test_length = text_length(test_set_text)

# Tf IDF as a feature set
vectorizer2 = TfidfVectorizer(ngram_range=(1, 3), max_features=1000)
train_set_vectorized2 = vectorizer2.fit_transform(train_set_text).toarray()
test_set_vectorized2 = vectorizer2.transform(test_set_text).toarray()

vectorizer3 = TfidfVectorizer(ngram_range=(1, 3), max_features=1000)
train_set_vectorized3 = vectorizer3.fit_transform(training_clusters).toarray()
test_set_vectorized3 = vectorizer3.transform(test_clusters).toarray()

# extract LDA as a feature set
n_topics_lda = 2 # number of topics
lda = LatentDirichletAllocation(n_components=n_topics_lda, random_state=42)
lda_train = lda.fit_transform(train_set_vectorized2)
lda_test = lda.transform(test_set_vectorized2)

# sentiment score as a feature set
def extract_sentiment_scores(sentences):
    sentiment_scores = []
    for sentence in sentences:
        blob = TextBlob(sentence)
        sentiment_scores.append([blob.polarity, blob.subjectivity])
    return np.array(sentiment_scores)

X_train_sentiment = extract_sentiment_scores(train_set_text)
X_test_sentiment = extract_sentiment_scores(test_set_text)

X_train_other = np.hstack((X_train_length, lda_train, X_train_sentiment))
X_test_other = np.hstack((X_test_length, lda_test, X_test_sentiment))

#%% BERT feature extractor
from transformers.modeling_bert import BertTokenizer, BertModel
import torch
# Initialize BERT model and tokenizer
bert_model = BertModel.from_pretrained('bert-base-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Function to get BERT embeddings
def get_bert_embedding(sentence):
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=100, return_tensors="pt")
    outputs = bert_model(**inputs)
    # Use the BERT representation of the `[CLS]` token as sentence embedding
    sentence_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return sentence_embedding

# extract features
bert_train = np.array([get_bert_embedding(text) for text in train_set_text]).squeeze()
bert_test = np.array([get_bert_embedding(text) for text in test_set_text]).squeeze()

# save the extracted features
np.save('bert_train.npy', bert_train)
np.save('bert_test.npy', bert_test)
#%% load the BERT extracted feature set
bert_train = np.load('bert_train.npy')
bert_test = np.load('bert_test.npy')

#%% normalization
scaler = StandardScaler()

train_set_vectorized = scaler.fit_transform(train_set_vectorized)
test_set_vectorized = scaler.transform(test_set_vectorized)
training_cluster_vectors = scaler.fit_transform(training_cluster_vectors)
test_cluster_vectors = scaler.transform(test_cluster_vectors)
X_train_w2v = scaler.fit_transform(X_train_w2v)
X_test_w2v = scaler.transform(X_test_w2v)
X_train_glove = scaler.fit_transform(X_train_glove)
X_test_glove = scaler.transform(X_test_glove)
train_set_vectorized2 = scaler.fit_transform(train_set_vectorized2)
test_set_vectorized2 = scaler.transform(test_set_vectorized2)
train_set_vectorized3 = scaler.fit_transform(train_set_vectorized3)
test_set_vectorized3 = scaler.transform(test_set_vectorized3)
X_train_other = scaler.fit_transform(X_train_other)
X_test_other = scaler.transform(X_test_other)
bert_train = scaler.fit_transform(bert_train)
bert_test = scaler.transform(bert_test)

#%% concat

X_train_all = [train_set_vectorized, training_cluster_vectors, X_train_w2v,
               X_train_glove, train_set_vectorized2, train_set_vectorized3,
               X_train_other, bert_train]

X_test_all = [test_set_vectorized, test_cluster_vectors, X_test_w2v,
              X_test_glove, test_set_vectorized2, test_set_vectorized3,
              X_test_other, bert_test]

n = 8
from itertools import combinations
# store different selction combinations
feature_index = []
for i in range(n+1):
    if i == 0: continue
    combination_length = i  
    
    # Get all combinations of the specified length from the list
    combs = list(combinations(range(n), combination_length))
    
    # Print the index of all selection combinations
    for comb in combs:
        feature_index.append(comb)
        
#%% Detect best feature set
classifiers = {
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'LightGBM': LGBMClassifier(),
    # 'CatBoost': CatBoostClassifier(verbose=0),
    # 'Neural Network': MLPClassifier()
}

# train classifiers and print evaluation metrics
performance = []
for sel in feature_index:
    print(sel)
    train_set = X_train_all[sel[0]]
    test_set = X_test_all[sel[0]]
    i = 1
    while(len(sel) > i):
        train_set = np.concatenate((train_set, X_train_all[sel[i]]), axis=1)
        test_set = np.concatenate((test_set, X_test_all[sel[i]]), axis=1)
        i += 1
    
    performance.append([])
    for name, clf in classifiers.items():
        # train classifier
        clf.fit(train_set, train_label)
        
        # make predictions
        y_pred = clf.predict(test_set)
        
        # calculate metrics
        acc = accuracy_score(test_label, y_pred)
        f1_micro = f1_score(test_label, y_pred, average='micro')
        f1_macro = f1_score(test_label, y_pred, average='macro')
        
        n = len(performance)
        performance[n-1].append((name, acc, f1_micro, f1_macro))
        
        # print metrics
        print(f"Classifier: {name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Micro-averaged F1 score: {f1_micro:.4f}")
        print(f"Macro-averaged F1 score: {f1_macro:.4f}")
        print('-' * 40)

#%% load performance array
# this array contains all the performances of each model for each of the 255 combinations of feature set
# the combinations are stored in feature_index variable
performance = np.load('performance.npy')
#%% Create best feature set
# 0: train_set_vectorized
# 1: training_cluster_vectors
# 2: X_train_w2v
# 3: X_train_glove
# 4: train_set_vectorized2
# 5: train_set_vectorized3
# 6: X_train_other
# 7: bert_train
sel = (0, 1, 5, 6)
train_set = X_train_all[sel[0]]
test_set = X_test_all[sel[0]]
i = 1
while(len(sel) > i):
    train_set = np.concatenate((train_set, X_train_all[sel[i]]), axis=1)
    test_set = np.concatenate((test_set, X_test_all[sel[i]]), axis=1)
    i += 1

#%% hyperparameter tuning
# function to find hyperparameters using grid search with 5-Fold CV
def grid_search_hyperparam_space(params, classifier, x_train, y_train):
        grid_search = GridSearchCV(estimator=classifier, param_grid=params,
                                   refit=True, cv=5, return_train_score=False,
                                   scoring='accuracy', verbose=2)
        grid_search.fit(x_train, y_train)
        return grid_search

classifiers = {
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(verbose=0),
    'Neural Network': MLPClassifier()
}

param_grids = {
    'SVM': {'C': [0.01,0.1, 1, 5, 10, 100], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'random_state': [1]},
    'Logistic Regression': {'C': [0.01,0.1, 1, 5, 10, 100], 'penalty': ['none', 'l2'], 'solver': ['liblinear', 'lbfgs'], 'random_state': [1]},
    'Random Forest': {'n_estimators': [50, 100, 200, 500], 'max_depth': [None, 5, 10, 12, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2, 4], 'random_state': [1]},
    'XGBoost': {'learning_rate': [0.01, 0.1], 'n_estimators': [50, 100, 200, 500], 'max_depth': [3, 5, 10, 12], 'random_state': [1]},
    'LightGBM': {'learning_rate': [0.01, 0.1], 'n_estimators': [50, 100, 200, 500], 'max_depth': [3, 5, 10, 12], 'random_state': [1]},
    'CatBoost': {'learning_rate': [0.01, 0.1], 'iterations': [50, 100, 200], 'depth': [3, 5, 10], 'random_state': [1]},
    'Neural Network': {'hidden_layer_sizes': [(32, 16, 8), (32, 64, 16), (64, 128, 32)], 'random_state': [1]}
}

grid_results = {}
for name, clf in classifiers.items():
    grid = grid_search_hyperparam_space(param_grids[name], clf, train_set, train_label)
    grid_results[name] = (grid, grid.best_params_, grid.best_score_)

#%% Best hyperparameters and accuracies
# SVM
 # {'C': 100, 'kernel': 'rbf', 'random_state': 1},
 # 0.8565346723241459)

# Logistic Regression
 # {'C': 0.01, 'penalty': 'l2', 'random_state': 1, 'solver': 'lbfgs'},
 # 0.8354735302103723)

# Random Forest
 # {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2,
 #  'n_estimators': 500, 'random_state': 1},
 # 0.8451471714629608)

# XGBoost
 # {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500, 'random_state': 1},
 # 0.8687764687764687)

# LightGBM
 # {'learning_rate': 0.1, 'max_depth': 12, 'n_estimators': 500, 'random_state': 1},
 # 0.8721944616681458)

# CatBoost
 # {'depth': 5, 'iterations': 200, 'learning_rate': 0.1, 'random_state': 1},
 # 0.8633678212625581)

# NN
 # {'hidden_layer_sizes': (32, 64, 16), 'random_state': 1},
 # 0.8394552447184026)












