from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import pickle
import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re

import numpy as np



def tokenize_tweets(tweets, max_sequence_length):
    if not os.path.exists('data/tokenizer.pkl'):
        print('tokenizer.pkl does not exisst')
    else:
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
            print('Loaded tokenizer.pkl')

    sequences = tokenizer.texts_to_sequences(tweets)
    tweets = pad_sequences(sequences, maxlen=max_sequence_length)
    return tweets

def tokenize_tweet(string, max_sequence_length):
    if not os.path.exists('data/tokenizer.pkl'):
        print('tokenizer.pkl does not exisst')
    else:
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
            print('Loaded tokenizer.pkl')

    sequence = tokenizer.texts_to_sequences(string)
    tweet = pad_sequences(sequence, maxlen=max_sequence_length)
    return tweet
def evaluate(y_true, y_pred):
    y_true = [1 if x == '1' else x for x in y_true]
    y_true = [0 if x == '0' else x for x in y_true]
    #y_pred = [1 if x == '1' else x for x in y_pred]
    #y_pred = [0 if x == '0' else x for x in y_pred]
    print('Accuracy score : ', format(accuracy_score(y_true, y_pred), '.2f') , '\n')
    print('Precision score : ',format( precision_score(y_true, y_pred), '.2f'), '\n')
    print('Recall score: ', format(recall_score(y_true, y_pred), '.2f'), '\n')
    print('F1 score : ', format(f1_score(y_true, y_pred), '.2f'), '\n')


def preprocessing(string):
    # remove diacritics
    string = remove_diacritics(string)
    # remove non arabic sysmbols
    string = remove_non_arabic(string)
    # remove extra white spaces
    string = remove_extra_whitespace(string)
    # remove punctiations
    string = remove_punctiation(string)
    # remove dubplicated letters
    string = remove_dubplicated_letters(string)
    # remove non arabic words
    string = remove_non_arabic_words(string)
    return string


def remove_extra_whitespace(string):
    string = re.sub(r'\s+', ' ', string)
    return re.sub(r"\s{2,}", " ", string).strip()


def remove_diacritics(string):
    regex = re.compile(r'[\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652]')
    return re.sub(regex, '', string)


def remove_non_arabic(string):
    return re.sub(r'[^\u0600-\u06FF]', ' ', string)


def remove_punctiation(string):
    return re.sub(
        r'[\u060C\u061B\u061F\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669\u066A\u066B\u066C\u066D\u0640]',
        '', string)


def remove_dubplicated_letters(string):
    return re.sub(r'(.)\1{2,}', r'\1', string)


def remove_non_arabic_words(string):
    return ' '.join([word for word in string.split() if not re.findall(
        r'[^\s\u0621\u0622\u0623\u0624\u0625\u0626\u0627\u0628\u0629\u062A\u062B\u062C\u062D\u062E\u062F\u0630\u0631\u0632\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063A\u0640\u0641\u0642\u0643\u0644\u0645\u0646\u0647\u0648\u0649\u064A]',
        word)])


def remove_stop_words(string, stop_words):
    """Sanitize using standard list comprehension"""
    return ' '.join([w for w in string.split() if w not in stop_words])
