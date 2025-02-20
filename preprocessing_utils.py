import csv
import numpy as np
import emoji
import random
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import random
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from tensorflow.keras.models import Model


def read_emoji_file(filename):
  phrase = []
  emoji = []

  with open (filename) as csvDataFile:
      csvReader = csv.reader(csvDataFile)
      for row in csvReader:
          row[0] = row[0].lower()
          row[0] = row[0].replace("congratulations", "congratulation")
          phrase.append(row[0])
          emoji.append(row[1])

  x = np.asarray(phrase)
  y = np.asarray(emoji, dtype=int)
  return x, y


def prepare_train_and_test_sets(train_filename, test_filename):
  X_train, Y_train = read_emoji_file(train_filename)
  X_test, Y_test = read_emoji_file(test_filename)
  len_word_to_be_added = (4*len(X_test) - len(X_train))//5
  X_train = np.append(X_train, X_test[:len_word_to_be_added])
  X_test = X_test[len_word_to_be_added:]
  Y_train = np.append(Y_train, Y_test[:len_word_to_be_added])
  Y_test = Y_test[len_word_to_be_added:]   
  X_train = np.delete(X_train, 43) 
  Y_train = np.delete(Y_train, 43)
  for count in range(len(X_train)):
    X_train[count] = X_train[count].replace("congratulation", "congrats")

  for count in range(len(X_test)):
    X_test[count] = X_test[count].replace("congratulation", "congrats")

  return X_train, Y_train, X_test, Y_test


def omit_stopwords(words):
    shallow_copy = words.copy()
    stop_words = set(stopwords.words('english'))
    for word in shallow_copy:
      for stop_word in stop_words:
        if word == stop_word:
            # print(word, end='\t')
            words.remove(word)
    
    return words


def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    lematized_nounly = [lemmatizer.lemmatize(word) for word in sentence]
    lematized_verbally = [lemmatizer.lemmatize(word, wordnet.VERB) for word in lematized_nounly]
    return lematized_verbally


def increase_len(nec_string, max_length):
  if len(nec_string) > max_length:
    res = ' '.join([s for s in nec_string[:max_length]])
 
  elif len(nec_string) < max_length:
    count1 = 0
    for _ in range(len(nec_string), max_length):
        word_to_be_add = nec_string[count1 % (len(nec_string))]
        nec_string.append(word_to_be_add)
        count1 += 1
    res = ' '.join(str(s) for s in nec_string)
  
  else:
    res = ' '.join(str(s) for s in nec_string)
  
  return res


def preprocess_corpus(corpus, max_length, return_unprocessed = False):
    corpus_copy = corpus.copy()
    for count in range(len(corpus)):
        nec_string = omit_stopwords(corpus_copy[count].split())
        lemm_sentence = lemmatize_sentence(nec_string)
        corpus_copy[count] = increase_len(lemm_sentence, max_length)    

    if return_unprocessed:        
       return corpus_copy, corpus
    else:
        return corpus_copy


def get_freq_count_len(doc, show_sentence = False):
    lengths = set()
    for sentence in doc:
      lengths.add(len(sentence.split()))
    
    length_freq = dict()
    for length in lengths:
      length_freq[length] = []

    for sentence in doc:
      length_freq[len(sentence.split())].append(sentence)

    if show_sentence:
        for i in length_freq.keys():
            print(i, " : ", length_freq[i])
    else:
       for i in length_freq.keys():
            print(i, " : ", len(length_freq[i]))

def augment_sentences(x_train, y_train, max_length):
    augmented_x = []
    augmented_y = []
    shuffled_indices = random.sample([t for t in range(0, len(x_train))], len(x_train))
    # print(shuffled_indices)

    for count in shuffled_indices:
        words_of_this_sentence = x_train[count].split()
        sentence_length = len(words_of_this_sentence)
        sh1 = random.sample([i for i in range(0, sentence_length)], max_length)
        sh2 = random.sample([i for i in range(0, sentence_length)], max_length)
        while (sh1 == sh2):
            sh2 = random.sample([i for i in range(0, sentence_length)], min(max_length, sentence_length))
        augmented_x.append(' '.join([words_of_this_sentence[j] for j in sh1]))
        augmented_x.append(' '.join([words_of_this_sentence[j] for j in sh2]))
        augmented_y.append(y_train[count])
        augmented_y.append(y_train[count])

    augmented_x = np.asarray(augmented_x)    
    augmented_y = np.asarray(augmented_y)
    permutation = np.random.permutation(len(augmented_x))

    return augmented_x[permutation], augmented_y[permutation]


def get_vocab(x_train, x_test):
    index_vocab = set()
    for i in range(len(x_train)):
        words = x_train[i].split()
        for word in words:
            index_vocab.add(word)

    for j in range(len(x_test)):
        words = x_test[j].split()
        for word in words:
            index_vocab.add(word)
    
    index_vocab = omit_stopwords(index_vocab)
    vocab_dict = dict(zip(sorted(index_vocab) + ['<unk>', '<pad>'], 
                list(range(len(index_vocab) + 2))))
    return vocab_dict


def get_embedding_of_sentences(doc, embed_model):
    # counter = 1
    doc_embedding = []
    for sentence in doc:
      words_inside_sentence = sentence.split()
      sentence_embedding = []
      for word in words_inside_sentence:
          embedding = embed_model.wv[word]
          sentence_embedding.append(embedding)
      sentence_embedding = np.asarray(sentence_embedding)
      doc_embedding.append(sentence_embedding)
    doc_embedding = np.asarray(doc_embedding)
    return doc_embedding


def calc_most_freq_words(x_doc, y_doc, frequent_words):
  for i in range(len(x_doc)):
    curr_sentence = x_doc[i].split()
    curr_label = y_doc[i]
    for word in curr_sentence:
      if word in frequent_words[curr_label].keys(): 
        frequent_words[curr_label][word] += 1 

      else:
        frequent_words[curr_label][word] = 0 
  
  return frequent_words


def count_freq_labels(doc, num_of_classes):
    y_scores = dict()

    for index in range(num_of_classes):
      y_scores[index] = 0

    for score in doc:
      y_scores[score] += 1

    for key in y_scores.keys():
      print(key, " : ", y_scores[key], " : ", round(y_scores[key] / len(doc), 3))
    
    plt.bar(y_scores.keys(), y_scores.values(), color='g')

