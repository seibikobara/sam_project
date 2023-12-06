
'''
This script was used to detect side effect
'''




import numpy as np
import pandas as pd
import Levenshtein
import nltk
import itertools
import re
import string


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






# load the current lexicon of side effect
symptom_dict = {}
infile = pd.read_excel("./covid_bc_sideeffect_dictionary_ver3.xlsx")
infile = infile.astype(str)
n , m = infile.shape
for i in range(n):
    row = infile.iloc[i,:]
    symptom_dict[str.strip(row["expression"]).lower()] = row["id"]


# Load posts
data = pd.read_excel('./bc_drug_discovered_evalulation_summarised_for_test.xlsx')
data.fillna("", inplace=True)


# Load nagation signs
negations = []
infile = open('./neg_trigs.txt')
for line in infile:
    negations.append(preprocessing(str.strip(line)))

n, m = data.shape

syms = []
negs = []
for i in range(n):
    tokenized_post = list(nltk.word_tokenize(preprocessing(data[7][i])))
    all_symptoms = []
    all_nags = []
    for window_size in range(1,9):
        for window in run_sliding_window_through_text(tokenized_post, window_size):
            window_string = ' '.join(window)

            detected_symptoms = set()
            negated_symptoms = set()
            symptom_match = []
    
            for symptom, CUI in symptom_dict.items():
                if is_similar(window_string, symptom):
                    if (CUI in all_symptoms) and window_size > 1:
                        continue
                    all_symptoms.append(CUI)            

                    is_negated = False
                    for neg in negations:
                        for match in re.finditer(r'\b'+neg+r'\b', data[7][i]):
                            is_negated = in_scope(match.end(),data[7][i], symptom)
                            if is_negated:
                                all_nags.append('1')
                                break
                    if not is_negated:
                        all_nags.append('0')
                    break

        data.iloc[i,8] = '$$$'+'$$$'.join(all_symptoms)+'$$$'
        data.iloc[i,9] = '$$$'+'$$$'.join(all_nags)+'$$$'


data.to_excel("./processed/thirdround/bc_drug_discovered_gold_standard_predicted.xlsx")


