import pandas as pd
import json
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
#print(df.keys())
print(type(df.content_question))

question = df.content_question.tolist()
answer = df.content_answer.tolist()
print(type(question))
print(type(answer))

tag = 0
dictConversation=[]
for i in range(len(question)):
    dictConversation.append({"tag":str(tag),"patterns":question[i],"responses":answer[i]})
    tag+=1
    
    
lemmatizer = WordNetLemmatizer()
words=list(map(word_tokenize,flatten(question)))

documents=[]
flag = 0
for i in range(len(words)):
    arr= [words[i],str(i)]
    documents.append(arr)

##ignore words
real_words= []
ignore_words=list("!@#$%^&*?")
for word in words:
    pick=[]
    for solo in word:
        if solo not in ignore_words:
            pick.append(solo)
    real_words.append(pick)


real_words=list(map(str.lower,flatten(real_words)))




real_words=list(map(lemmatizer.lemmatize,real_words))
real_words=sorted(list(set(real_words)))
#classes=sorted(list(set(classes)))
#training the model
cv=CountVectorizer(tokenizer=lambda txt: txt.split(),analyzer="word",stop_words=None)
training=[]

classes = []
for i in range(len(words)):
    classes.append(str(i))
    
for doc in documents:
            #lower case and lemmatize the pattern words
            pattern_words=list(map(str.lower,doc[0]))
            ##print(pattern_words)
            pattern_words=' '.join(list(map(lemmatizer.lemmatize,pattern_words)))
            ##print(pattern_words)
            #train or fit the vectorizer with all words
            #and transform into one-hot encoded vector
            vectorize=cv.fit([' '.join(real_words)])
            word_vector=vectorize.transform([pattern_words]).toarray().tolist()[0]

            #create output for the respective input
            #output size will be equal to total numbers of classes
            output_row=[0]*len(classes)

            #if the pattern is from current class put 1 in list else 0
            output_row[classes.index(doc[1])]=1
            cvop=cv.fit([' '.join(classes)])
            out_p=cvop.transform([doc[1]]).toarray().tolist()[0]

            #store vectorized word list long with its class
            training.append([word_vector,output_row])
            
random.shuffle(training)
training=np.array(training,dtype=object)
train_x=list(training[:,0])#patterns
train_y=list(training[:,1])#classes

model=Sequential()
#input layer with latent dimension of 128 neurons and ReLU activation function
model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5)) #Dropout to avoid overfitting
#second layer with the latent dimension of 64 neurons
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
#fully connected output layer with softmax activation function
model.add(Dense(len(train_y[0]),activation='softmax'))
'''Compile model with Stochastic Gradient Descent with learning rate  and
   nesterov accelerated gradient descent'''
sgd=SGD(learning_rate=1e-2,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
#fit the model with training input and output sets
hist=model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=10,verbose=1)
#save model and words,classes which can be used for prediction.
model.save('chatbot_model.h5',hist)
pickle.dump({'words':real_words,'classes':classes,'train_x':train_x,'train_y':train_y},open('training_data',"wb"))














