from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import initializers
from tensorflow.keras.layers import Embedding
from numpy import zeros
from numpy import asarray
import pandas as pd
import preprocessor as p
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from gensim.parsing.preprocessing import remove_stopwords


#warnings.filterwarnings(action='ignore')


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

#Keras tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_d)

X_train = tokenizer.texts_to_sequences(train_d)
X_test = tokenizer.texts_to_sequences(test_d)

vocab_size = len(tokenizer.word_index) + 1
maxlen = 200
X_train = pad_sequences(X_train, padding='post', maxlen= maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)



#upload Word2Vec
# load embedding as a dict
def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename,'r')
	lines = file.readlines()[1:]
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = asarray(parts[1:], dtype='float32')
	return embedding

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = zeros((vocab_size, 100))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		weight_matrix[i] = embedding.get(word)
	return weight_matrix


# load embedding from file
raw_embedding = load_embedding('embedding_word2vec_all.txt')
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
# create the embedding layer
embedding_layer = Embedding(vocab_size, 100, embeddings_initializer=initializers.Constant(embedding_vectors), input_length=maxlen, trainable=False)


model = Sequential()
model.add(embedding_layer)
model.add(layers.Conv1D(128, 5, activation='sigmoid'))
model.add(layers.MaxPool1D())
model.add(layers.Flatten())
model.add(layers.Dense(20, activation='sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))
#model.add(layers.Dropout)
model.compile(optimizer= 'rmsprop',
              loss="binary_crossentropy",
              metrics=['acc'])
model.summary()



maxlen = 200

history = model.fit(X_train, y_train,
                    epochs=20,
                    verbose=True,
                    validation_data=(X_test, y_test),
                    batch_size=128)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
#plot_history(history)