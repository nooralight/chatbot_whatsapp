import pandas as pd
import random
import pickle
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import flatten
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD


df = pd.read_csv('chtbit_df.csv')


question =df.content_question.tolist()
answer = df.content_answer.tolist()
words=list(map(word_tokenize,flatten(question)))
#fetch classes i.e. tags and store in documents along with tokenized patterns

dictConversation=[]
tag = 0
for i in range(len(question)):
    dictConversation.append({"tag":str(tag),"patterns":question[i],"responses":answer[i]})
    tag+=1

pickle.dump({'conversations':dictConversation},open('conversations',"wb"))