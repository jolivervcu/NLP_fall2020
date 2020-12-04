## Data and Code for project of Identifying Informative COVID-19 English Tweets. Project comes from WNUT-2020 Task 2: Identification of informative COVID-19 English Tweets. Data was provided as part of this project. 

Data is partitioned in three files, "Train.tsv", "unlabeled_test_with_noise.tsv", and "valid.tsv."

This program will run using Linux, and possibly Windows. 

A Convolutional Neural Network was built for this task. Three variations of the model were created with different embeddings. Two were pre-trained word embeddings and one embedding was trained with the model. The Two pre-trained word embeddings used GLOVE and Word2Vec. The GLOVE model is titled "cnn_glove.py" and the Word2Vec model is titled "word2vec_cnn.py" The final model is titled "cnnembed.py." 

The pretrained GLOVE embedding needs to be downloaded for the file "cnn_glove.py.". This can be acquired from the Stanford NLP group at "https://nlp.stanford.edu/projects/glove/" For this project, the one with embedding dimensions of 50 was used ("glove.6B.50d.txt"). 

The code is written with Python. To run the code, see "INSTALL.md" for installation requirements. After installation of proper packages, download the data and the code for the models and data. The Word2Vec embedding was created with the code titled "word2vec_data.py" This file creates the text document of the data after the Word2Vec Model. This code must be run before "word2vec_cnn.py." 



Source:
@inproceedings{covid19tweet,
title = {{WNUT-2020 Task 2: Identification of Informative COVID-19 English Tweets}},
author = {Dat Quoc Nguyen and Thanh Vu and Afshin Rahimi and Mai Hoang Dao and Linh The Nguyen and Long Doan},
booktitle = {Proceedings of the 6th Workshop on Noisy User-generated Text},
year = {2020}
}


 