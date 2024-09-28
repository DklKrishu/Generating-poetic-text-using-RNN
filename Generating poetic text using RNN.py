#!/usr/bin/env python
# coding: utf-8

# In[43]:


pip install tensorflow


# In[44]:


pip install wordcloud


# In[45]:


import random # importing random which is used for shuffling or creating random index
import numpy as np # creating numpy library for mathematical calculation
import tensorflow as tf # importing tensorflow library 
from tensorflow.keras.models import Sequential # importing sequential to stack the model sequentially 
from tensorflow.keras.layers import LSTM, Dense, Activation # adding layers like LSTM,Dense,Activation
from tensorflow.keras.optimizers import RMSprop
import warnings # importing warings
# Ignore all warnings
warnings.filterwarnings('ignore')


# In[46]:


filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')


# In[47]:


text = open(filepath,'rb').read().decode(encoding='utf-8').lower()


# In[48]:


text


# In[49]:


text = text[300000:800000]


# In[50]:


text


# In[51]:


characters = sorted(set(text))


# In[52]:


char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i,c) for i, c in enumerate(characters))


# In[53]:


SEQ_LENGTH = 40 
STEP_SIZE = 3
sentences = []
next_characters =[]


# In[54]:


for i in range(0, len(text)- SEQ_LENGTH,STEP_SIZE):
    sentences.append(text[i: i+SEQ_LENGTH])
    next_characters.append(text[i+SEQ_LENGTH])


# In[55]:


X = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)


# In[56]:


y = np.zeros((len(sentences), len(characters)), dtype=bool)


# In[57]:


print(f"Length of sentences: {len(sentences)}")
print(f"Length of next_characters: {len(next_characters)}")


# In[58]:


min_length = min(len(sentences), len(next_characters))
sentences = sentences[:min_length]
next_characters = next_characters[:min_length]


# In[59]:


for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        X[i, t, char_to_index[character]] = 1
    
    if i < len(next_characters):  # Ensure i is within bounds
        y[i, char_to_index[next_characters[i]]] = 1
    else:
        print(f"Index {i} out of range for next_characters")


# In[60]:


model = Sequential()
model.add(LSTM(128, input_shape = (SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))


# In[61]:


model.compile(loss = 'categorical_crossentropy',optimizer =RMSprop(learning_rate = 0.01))


# In[62]:


model.fit(X,y, batch_size = 256, epochs = 10)


# In[65]:


model.save('textgenerator.model.keras')


# In[71]:


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')  # Fixed 'aassay' to 'asarray'
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

    


# In[72]:


def generate_text(length, temperature):
    start_index = random.randint(0, len(text)- SEQ_LENGTH-1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate (sentence):
            x [0, t, char_to_index[character]] = 1
            
        predictions = model.predict(x,verbose = 0 )[0]
        next_index = sample(predictions,temperature)
        next_character = index_to_char[next_index]
        generated += next_character
        sentence = sentence[1:] + next_character
    return generated
    
        


# In[73]:


print('--------0.2----------')
print(generate_text(300, 0.2))


# In[74]:


print('--------0.4----------')
print(generate_text(300, 0.4))


# In[75]:


print('--------0.6----------')
print(generate_text(300, 0.6
                   ))


# In[76]:


print('--------0.8----------')
print(generate_text(300, 0.8))


# In[77]:


print('--------1----------')
print(generate_text(300, 1))


# In[ ]:




