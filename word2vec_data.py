
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.parsing.preprocessing import remove_stopwords
import pandas as pd
import preprocessor as p
from sklearn.model_selection import train_test_split


training_data = pd.read_csv('train.tsv', sep = '\t')
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
training_data = training_data.replace(to_replace=['INFORMATIVE', 'UNINFORMATIVE'], value= [1,2])
df_tweets = training_data['Text'].values
y = training_data['Label'].values
train_d, test_d, y_train, y_test = train_test_split(df_tweets, y, test_size= 0.25, random_state=1000)


#tokenize words of tweet
all_data = []
for i in df_tweets:
    temp = []
    for j in word_tokenize(i):
        temp.append(j.lower())
    all_data.append(temp)

tr_data = []

for i in train_d:
    temp = []
    for j in word_tokenize(i):
        temp.append(j.lower())
    tr_data.append(temp)

tst_data = []
for i in test_d:
    temp = []
    for j in word_tokenize(i):
        temp.append(j.lower())
    tst_data.append(temp)

# #Word2Vec
word2vec_all_model = gensim.models.Word2Vec(all_data, min_count = 1, size = 100, window = 5)
word2vec_train_model = gensim.models.Word2Vec(tr_data, min_count = 1, size = 100, window = 5)
word2vec_test_model = gensim.models.Word2Vec(tst_data, min_count = 1, size = 100, window = 5)

words = list(word2vec_all_model.wv.vocab)
print('Vocabulary size: %d' % len(words))

filename = 'embedding_word2vec_all.txt'
word2vec_all_model.wv.save_word2vec_format(filename, binary=False)

filename2 = 'embedding_word2vec_train.txt'
word2vec_train_model.wv.save_word2vec_format(filename2, binary=False)

filename3 = 'embedding_word2vec_test.txt'
word2vec_test_model.wv.save_word2vec_format(filename3, binary=False)

