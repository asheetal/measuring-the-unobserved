#!/usr/bin/env python3
# 
# This file is part of the Era of Transformation project.
# Copyright (c) 2023 XXXX.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

#Adapted from https://www.kaggle.com/code/rtatman/fine-tuning-word2vec-2-0
#
# This file takes the baseline GLOVE and creates multiple fine-tuned models based 
# on yearly training data
#

import gensim
from gensim.models import Word2Vec 
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pandas as pd
from pandas import json_normalize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pyreadr
import openpyxl
import re
import json
import numpy

# Load pre-trained GLOVE model
glove_path = '/research/dataset/GLOVE/glove.42B.300d_gensim.txt'
glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=False)

#Puersuade 2.0
#base_dir = '/research/dataset/AP_Developmental/PERSUADE2.0/'
#wb = openpyxl.load_workbook('/research/dataset/AP_Developmental/PERSUADE2.0/for_glove_splitted.xlsx') 

#Texas
#base_dir = '/research/dataset/AP_Developmental/texas/'
#wb = openpyxl.load_workbook('/research/dataset/AP_Developmental/texas/for_glove_splitted.xlsx') 

#OKCUpid
#base_dir = '/research/dataset/AP_Developmental/OKCupid/'
#jfile = "/research/dataset/AP_Developmental/OKCupid/for_glove_splitted.json"

#RateMyProfessor
base_dir = '/research/dataset/AP_Developmental/RateMyProfessor/'
jfile = "/research/dataset/AP_Developmental/RateMyProfessor/for_glove_splitted.json"

#RateMDs
#base_dir = '/research/dataset/AP_Developmental/rateMDs/'
#jfile = "/research/dataset/AP_Developmental/rateMDs/for_glove_splitted.json"

with open(jfile) as f:
    wb_list = json.load(f)

wb_sheetnames = wb_list.keys()

res = len(wb_sheetnames)
cachedStopWords = stopwords.words("english")
pattern_order1 = r'[0-9]'
pattern_order2 = re.compile(r'\b(' + r'|'.join(cachedStopWords) + r')\b\s*')
# Load GloVe model
glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=False)  # Ensure no_header matches your file format
glove_vocab = set(glove_model.key_to_index.keys())
  
def my_year_embeddings(arrayofvalues, count, name):
  # grab forum forum posts
  sentences_1 = arrayofvalues.to_numpy().flatten()
  sentences_1 = [re.sub("[^a-zA-Z]+", ' ', i) for i in sentences_1]
  sentences_2 = list(filter(None, sentences_1))
  sentences_3 = [re.sub(pattern_order1, '', i) for i in sentences_2]
  sentences_3b = [re.sub(r'\b\d+\.\d+\b', '', i) for i in sentences_3]

  sentences_4 = list(filter(None, sentences_3b))
  sentences = [re.sub(pattern_order2, '', i) for i in sentences_4]

  # tokenize
  tokenizer = RegexpTokenizer(r'\w+')
  sentences_tokenized = [str(w).lower() for w in sentences]
  sentences_tokenized = [tokenizer.tokenize(i) for i in sentences_tokenized]
  
  sentences_filtered = [[word for word in sentence if word in glove_vocab] for sentence in sentences_tokenized]

  model_new = Word2Vec(vector_size=glove_model.vector_size, min_count=1)
  model_new.build_vocab([list(glove_model.key_to_index.keys())])

  model_new.wv.vectors_lockf = numpy.ones(len(model_new.wv), dtype=numpy.float32)  # Allow training/updating
  model_new.wv.intersect_word2vec_format(glove_path, binary=False, lockf=1.0)

  total_examples = len(sentences_filtered)
  #model_new.wv.load_word2vec_format(glove_path, binary=False)
  model_new.train(sentences_filtered, total_examples=total_examples, epochs=10)
  #model_new.wv.similar_by_word("national")
  #glove_model.similar_by_word("national")
  save_file = base_dir + "/word2vec_" + str(name).lower() + ".42B.model"
  model_new.wv.save_word2vec_format(save_file, binary = True)
  

for count, name in zip(range(0, res, 1), wb_sheetnames):
  print(name)
  print(count)
  df = json_normalize(wb_list[name])
  arrayofvalues = df['review_text']
  my_year_embeddings(arrayofvalues, count, name)
  


#model_2.wv.similar_by_word("leadership")
#model_2.wv.most_similar(positive=['leadership', 'female'], negative=['male'], topn=10)
#model_2.wv.similarity('france', 'spain')

