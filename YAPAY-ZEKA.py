#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 20:06:57 2020

@author: kullanici
"""
import nltk
from  nltk.stem.lancaster import  LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow
import  random

import  json

with  open("intents.json") as file:
    data = json.load(file)

words =  []
labels = []
docs = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs.append(pattern)
        
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
        