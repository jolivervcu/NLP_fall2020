
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd
import preprocessor as p
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from gensim.parsing.preprocessing import remove_stopwords




#data upload
training_data = pd.read_csv('train.tsv', sep = '\t')
#clean data
def preprocess_tweet(row):
    text = row['Text']
    text = p.clean(text)
    return text
training_data['Text'] = training_data.apply(preprocess_tweet, axis=1)
#remove stop words
def stopword_removal(row):
    text = row['Text']
    text = remove_stopwords(text)
    return text
training_data['Text'] = training_data.apply(stopword_removal, axis=1)
training_data['Text'] = training_data['Text'].str.lower().str.replace('[^\w\s]',' ').str.replace('\s\s+', ' ')
training_data = training_data.replace(to_replace=['INFORMATIVE', 'UNINFORMATIVE'], value= [0,1])
df_tweets = training_data['Text'].values
y = training_data['Label'].values
train_d, test_d, y_train, y_test = train_test_split(df_tweets, y, test_size= 0.25, random_state=1000)

#plot
def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

#Keras tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_d)

X_train = tokenizer.texts_to_sequences(train_d)
X_test = tokenizer.texts_to_sequences(test_d)

vocab_size = len(tokenizer.word_index) + 1
maxlen = 200
X_train = pad_sequences(X_train, padding='post', maxlen= maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

#create embedding layer
def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

embedding_dim = 50
embedding_matrix = create_embedding_matrix('glove.6B.50d.txt',tokenizer.word_index, embedding_dim)


#create model
model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim,
                           weights=[embedding_matrix],
                           input_length=maxlen,
                           trainable=True))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.MaxPool1D())
model.add(layers.Flatten())
model.add(layers.Dense(20, activation='relu'))
model.add(layers.Dense(1, activation='relu'))
model.compile(optimizer='rmsprop',
              loss="binary_crossentropy",
              metrics=['acc'])
model.summary()



maxlen = 200

history = model.fit(X_train, y_train,
                    epochs=20,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
#plot_history(history)




