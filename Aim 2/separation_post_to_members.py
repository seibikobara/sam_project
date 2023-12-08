'''
This script was used to split breast cancer posts to each member for manual annotation in multi-layer rule-based model development architecture
'''




import numpy as np
import pandas as pd
import Levenshtein
import nltk
import itertools
import re
import string
from cleantext import clean
import sys
import regex as re
 


# Collapse posts by unique username
FLAGS = re.MULTILINE | re.DOTALL
 
def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " {} ".format(hashtag_body.lower())
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result
 
def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"
 
 
def tokenize(text):
    text=str(text)
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"
 
    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)
 
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
 
    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)
 
    return text.lower()

def preprocessing(corpus):
    # Convert text to lower case
    corpus = corpus.lower()
    
    # Remove any date using regex
    date_pattern = r'\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b'
    cleaned_corpus = re.sub(date_pattern, '', corpus)
    
    return cleaned_corpus

# Levenshtein similarity check
SIMILARITY_THRESHOLD = 0.85

def is_similar(subsequence, target):
    return Levenshtein.ratio(subsequence, target) > SIMILARITY_THRESHOLD


# Rolling slinding window
def run_sliding_window_through_text(words, window_size):
    """
    Generate a window sliding through a sequence of words
    """
    word_iterator = iter(words) # creates an object which can be iterated one element at a time
    word_window = tuple(itertools.islice(word_iterator, window_size)) #islice() makes an iterator that returns selected elements from the the word_iterator
    yield word_window
    #now to move the window forward, one word at a time
    for w in word_iterator:
        word_window = word_window[1:] + (w,)
        yield word_window

# Nagation detection
def in_scope(neg_end, text,symptom_expression):
    '''
    Function to check if a symptom occurs within the scope of a negation based on some
    pre-defined rules.
    :param neg_end: the end index of the negation expression
    :param text:
    :param symptom_expression:
    :return:
    '''
    negated = False
    text_following_negation = text[neg_end:]
    tokenized_text_following_negation = list(nltk.word_tokenize(text_following_negation))

    three_terms_following_negation = ' '.join(tokenized_text_following_negation[:min(len(tokenized_text_following_negation),3)])
    match_object = re.search(symptom_expression,three_terms_following_negation)
    if match_object:
        period_check = re.search('\.',three_terms_following_negation)
        next_negation = 1000
        for neg in negations:
            if re.search(neg,text_following_negation):
                index = text_following_negation.find(neg)
                if index<next_negation:
                    next_negation = index
        if period_check:
            if period_check.start() > match_object.start() and next_negation > match_object.start():
                negated = True
        else:
            negated = True
    return negated

# Sypmtom identification
def detect_symptoms(post, symptoms_dict, window_size):
    detected_symptoms = set()
    negated_symptoms = set()
    symptom_match = []
    
    for symptom, CUI in symptoms_dict.items():
        if is_similar(post, symptom):
            symptom_match = symptom
            break
    return symptom_match







# load full data
df_test = pd.read_csv("./BC_full_data.csv", header=None)
df_test[7] = df_test[7].apply(lambda x: tokenize(x))
df_test = df_test.groupby(4)[7].apply(lambda x: '. '.join(x)).reset_index()

# breast cancer labels from task 1
labels = np.load("./labels_BC_full.npy")
df_test["label"] = labels

# filter only BC posts
bc_full = df_test.loc[df_test["label"]==1,:].reset_index(drop = True)

# Load the lexicon
temp = pd.read_csv("breast_cancer_drugs_cleaned.csv")
# drug name and brand name both can be used
drugs = temp["drug"].to_list() + temp["us_brand_name"].to_list() 
drugs = list(set(drugs))


# drug discovery
n, m = bc_full.shape
post_drug = {}
for i in range(n):
    percent = np.round(i/n*100, 1)
    if (percent*10) % 10 == 0:
        print("currently {:.1f} %".format(percent))
    post = bc_full.iloc[i,1]
    tokenized_post = list(nltk.word_tokenize(preprocessing(bc_full.iloc[i,1])))
    drugs_in_post = []
    exps_in_post = []
    for token in tokenized_post:
    
        for drug in drugs:
                if is_similar(token, drug.lower()):
                    drugs_in_post.append(drug) 
                    exps_in_post.append(token)
    drugs_used = ','.join(drugs_in_post)
    exps_used = ','.join(exps_in_post)

    post_drug[i] = [drugs_used, exps_used]      


res = pd.DataFrame(post_drug.values())
res.columns = ["drug_name", "expression"]


# merge to the bc_full
temp = pd.concat([bc_full, res], axis = 1)


# only post with drugs
bc_full_drug_discovered = temp.loc[~(temp["drug_name"]=="")].reset_index(drop=True)
bc_full_drug_no_discovered = temp.loc[(temp["drug_name"]=="")].reset_index(drop=True)


# split data to each member using random number generator for manual annotation

import random
n, m = bc_full_drug_discovered.shape
index = random.sample(range(n), 20)
index_manual_annotation = index[0:10]
index_gold = index[10:20]
# for 10 manual annotation
bc_manual_annotation = bc_full_drug_discovered.iloc[index_manual_annotation, :]
bc_manual_annotation.to_csv("bc_drug_discovered_manual_annotation.csv")
bc_gold = bc_full_drug_discovered.iloc[index_gold, :]
bc_gold.to_csv("bc_drug_discovered_gold_standard.csv")


# for drug no discovered post
n, m = bc_full_drug_no_discovered.shape
index = random.sample(range(n), 30)
index_1 = index[0:10]
index_2 = index[10:20]
index_3 = index[20:30]


bc_full_drug_no_discovered.iloc[index_1,:].to_csv("bc_drug_discovery_1.csv")
bc_full_drug_no_discovered.iloc[index_2,:].to_csv("bc_drug_discovery_2.csv")
bc_full_drug_no_discovered.iloc[index_3,:].to_csv("bc_drug_discovery_3.csv")

