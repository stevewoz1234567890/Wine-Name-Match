# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 10:45:56 2021

@author: koala
"""
"""
calculate similarity
"""

import re, math
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.corpus import wordnet as wn

stop = stopwords.words('english')

WORD = re.compile(r'\w+')
stemmer = PorterStemmer()

def text_to_vector(text):
    words = WORD.findall(text)
    a = []
    for i in words:
        for ss in wn.synsets(i):
            a.extend(ss.lemma_names())
    for i in words:
        if i not in a:
            a.append(i)
    a = set(a)
    w = [stemmer.stem(i) for i in a if i not in stop]
    return Counter(w)

def get_similarity(a_set, a_vec, a_sum, b_vec, b_set):
    intersection = a_set & b_set
    numerator = sum([a_vec[x] * b_vec[x] for x in intersection])

    if not a_sum:
        return 0.0
    else:
        return float(numerator) / a_sum

"""
import and preprocessing wine table
"""

import pandas as pd

# standard table
with open('wine') as f:
  ff = f.readlines()
  ff = [fff.strip().lower().replace('"', '') for fff in ff]
  
keylist  = ff[0].split(';')

standard_wine_table = []

for i in range(len(ff)-1):
  temp = ff[i+1][1:-2].split(';')
  temp_dict = {}
  for j in range(len(keylist)):
    temp_dict[str(keylist[j])] = temp[j]
  standard_wine_table.append(temp_dict)

standard_wine_table = pd.DataFrame(standard_wine_table)
standard_wine_table.columns = ["WineID","Color","Type","Varietal","Producer","Description","Country","Region","SubRegion","subappellation","Name","created_at"]

standard_wine_name = standard_wine_table.loc[:, 'Name']
standard_wine_Type = standard_wine_table.loc[:, 'Type']
standard_wine_Varietal = standard_wine_table.loc[:, 'Varietal']
standard_wine_Producer = standard_wine_table.loc[:, 'Producer']
standard_wine_Description = standard_wine_table.loc[:, 'Description']
standard_wine_Country = standard_wine_table.loc[:, 'Country']
standard_wine_Region = standard_wine_table.loc[:, 'Region']
standard_wine_SubRegion = standard_wine_table.loc[:, 'SubRegion']
standard_wine_subappellation = standard_wine_table.loc[:, 'subappellation']

standard_wine_name_1 = [standard_wine_name[i]+' '+
                        standard_wine_Varietal[i]+' '+
                        standard_wine_Producer[i]+' '+
                        standard_wine_Description[i]+' '+
                        standard_wine_Type[i]+' '+
                        #standard_wine_Country[i]+
                        #standard_wine_Region[i]+
                        #standard_wine_SubRegion[i]+' '+
                        standard_wine_subappellation[i]+' '
                        +'wine'
                        for i in range(len(standard_wine_name))]

standard_wine_vector = [text_to_vector(a) for a in standard_wine_name_1]
standard_wine_vector_keys = [a.keys() for a in standard_wine_vector]
standard_wine_vector_sets = [set(a) for a in standard_wine_vector_keys]
standard_wine_vector_sums = [sum([a[x]**2 for x in a.keys()]) for a in standard_wine_vector]
standard_wine_vector_sums = [math.sqrt(a) for a in standard_wine_vector_sums]

# scrapping table
with open('items_1') as f:
  ff = f.readlines()
  
keylist  = ff[0][1:-2].split(',')

scrapping_wine_table = []

for i in range(len(ff)-1):
  temp = ff[i+1][1:-2].split(',')
  temp_dict = {}
  for j in range(len(keylist)):
    temp_dict[keylist[j]] = temp[j]
  scrapping_wine_table.append(temp_dict)

scrapping_wine_table = pd.DataFrame(scrapping_wine_table)

scrapping_wine_name = scrapping_wine_table.loc[:, '`Name`']


b = "Stag's Leap Wine Cellars Cask 23 Cabernet Sauvignon (1.5 Liter Magnum) 2014"
b_vec = text_to_vector(b.strip().lower())
b_set = set(b_vec.keys())

max_name_1 = 0
max_val_1 = get_similarity(standard_wine_vector_sets[0], standard_wine_vector[0], standard_wine_vector_sums[0], b_vec, b_set)
max_name_2 = 1
max_val_2 = get_similarity(standard_wine_vector_sets[1], standard_wine_vector[1], standard_wine_vector_sums[1], b_vec, b_set)
max_name_3 = 2
max_val_3 = get_similarity(standard_wine_vector_sets[2], standard_wine_vector[2], standard_wine_vector_sums[2], b_vec, b_set)

for i in range(3, len(standard_wine_name)):
  tmp_val = get_similarity(standard_wine_vector_sets[i], standard_wine_vector[i], standard_wine_vector_sums[i], b_vec, b_set)
  if tmp_val > max_val_1:
    max_val_1 = tmp_val
    max_name_1 = i
    continue
  if tmp_val > max_val_2:
    max_val_2 = tmp_val
    max_name_2 = i
    continue
  if tmp_val > max_val_3:
    max_val_3 = tmp_val
    max_name_3 = i
    continue
  
print(str(standard_wine_name[max_name_1]))
print(max_val_1)
print(str(standard_wine_name[max_name_2]))
print(max_val_2)
print(str(standard_wine_name[max_name_3]))
print(max_val_3)