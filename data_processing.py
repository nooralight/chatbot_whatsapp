import pandas as pd


df = pd.read_csv('chtbit_df.csv')
#print(df.keys())
print(type(df.content_question))

question = df.content_question.tolist()
answer = df.content_answer.tolist()
print(type(question))
print(type(answer))


word2count = {}
for line in range(len(question)):
    for word in question[line].split(" "):
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
for line in range(len(answer)):
    for word in answer[line].split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
            
vocab = {}
word_num = 0
for word, count in word2count.items():
    vocab[word] = word_num
    word_num += 1
    
for i in range(len(answer)):
    answer[i] = '<SOS> ' + answer[i] + ' <EOS>'
    
    
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
x = len(vocab)
for token in range(len(tokens)):
    vocab[tokens[token]] = x
    x += 1


inv_vocab = {w:v for v, w in vocab.items()}


encoder_inp = []
for line in range(len(question)):
    lst = []
    for word in question[line].split():
        lst.append(vocab[word])       
    encoder_inp.append(lst)


decoder_inp = []
for line in range(len(answer)):
    lst = []
    for word in answer[line].split():
        lst.append(vocab[word])        
    decoder_inp.append(lst)
    
encoder_inp = pad_sequences(encoder_inp, 10, padding='post', truncating='post')
decoder_inp = pad_sequences(decoder_inp, 10, padding='post', truncating='post')


# Without <SOS> token
decoder_final_output = []
for i in range(len(decoder_inp)):
    decoder_final_output.append(decoder_inp[i][1:]) 

decoder_final_output = pad_sequences(decoder_final_output, 10, padding='post', truncating='post')

del i,word


decoder_final_output = to_categorical(decoder_final_output, len(vocab))