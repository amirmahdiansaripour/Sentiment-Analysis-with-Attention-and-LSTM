import numpy as np
import emoji
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from tensorflow.keras.models import Model
from preprocessing_utils import *

emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    return emoji.emojize(emoji_dictionary[str(label)], language='alias')
              
    
def print_predictions(X, pred):
    print()
    for i in range(X.shape[0]):
        print(X[i])
        print("Predicted emotion: ", label_to_emoji(int(pred[i])))
        print('\n')


def middle_layer_output(modelx, Ty):
    X = modelx.inputs[0] 
    s0 = modelx.inputs[1] 
    c0 = modelx.inputs[2] 
    s = s0
    c = s0    
    a = modelx.layers[2](X)  
    attentions = []

    for _ in range(Ty):
        s_prev = s
        s_prev = modelx.layers[3](s_prev)
        concat = modelx.layers[4]([a, s_prev]) 
        e = modelx.layers[5](concat) 
        energies = modelx.layers[6](e) 
        alphas = modelx.layers[7](energies) 
        context = modelx.layers[8]([alphas, a])
        s, _, c = modelx.layers[10](context, initial_state = [s, c]) 
        attentions.append(alphas)

    print("Attentions: ", attentions[-1])
    middle_layer = Model(inputs=[X, s0, c0], outputs = attentions[-1])
    
    return middle_layer


def plot_attention_map(attention_weights, prediction, Tx, Ty, important_words, input_text):
    plt.clf()
    f = plt.figure(figsize=(8, 8.5))
    ax = f.add_subplot(1, 1, 1)
    i = ax.imshow(attention_weights, interpolation='nearest', cmap='Blues')

    cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
    cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xlabel('Alpha value (Probability output of the "softmax")', labelpad=2)

    ax.tick_params(axis='y', labelsize=20)
    ax.set_yticks(range(Ty))
    ax.set_yticklabels(label_to_emoji(prediction))
    ax.set_xticks(range(Tx))
    ax.set_xticklabels(important_words.split(), rotation=45)

    print("Emoji: ", label_to_emoji(prediction))

    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')
    ax.set_title("Input sentence: " + input_text)

    ax.grid()
    f.show()


def find_attetion_between_label_and_words(modelx, text, Tx, Ty, embed_model, n_s = 64):

    middle_output_model = middle_layer_output(modelx, Ty)
    sentence_processed = preprocess_corpus(text, Tx)

    test_embedding = get_embedding_of_sentences(sentence_processed, embed_model)
    s0 = np.zeros((text.shape[0], n_s))
    c0 = np.zeros((text.shape[0], n_s))

    attention_weights = middle_output_model([test_embedding, s0, c0])
    # Normalize attention map
    row_max = attention_weights.numpy().max(axis=1)
    attention_weights = attention_weights / row_max[:, None]
    
    print("s0 and c0 shapes: ", s0.shape, c0.shape)
    print("sentence processed: ", sentence_processed)
    print("Embedding shape: ", test_embedding.shape)
    print("attention weights shape: ", attention_weights.shape)
    print("attention weights (normalized): ", attention_weights)

    prediction = modelx.predict([test_embedding, s0, c0])
    prediction = np.argmax(prediction, axis=-1)[0]
    
    plot_attention_map(attention_weights, prediction, Tx, Ty, sentence_processed[0], text[0])