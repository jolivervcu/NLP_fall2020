
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import metrics
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

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen= maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

embedding_dim = 50

#model #1 training accuracy = 0.4694, testing accuracy = 0.4792
model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(64, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc' ,metrics.Precision(), metrics.Recall()])

model.summary()
history = model.fit(X_train, y_train,
                    epochs=20,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy, precision, recall = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
print("Testing Precision:  {:.4f}".format(precision))
print("Testing Recall:  {:.4f}".format(recall))
F1 = 2 * (precision * recall) / (precision + recall)
print(F1)


