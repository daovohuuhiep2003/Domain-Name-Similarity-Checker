"""Sample code for siamese neural net for detecting spoofing attacks"""
from __future__ import with_statement

import matplotlib
matplotlib.use('Agg')

import cPickle as pickle
import editdistance
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import strSimilarity

from keras.layers import Dense, Input, Lambda, Flatten, Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model, model_from_json
from keras import backend as K
from keras.optimizers import RMSprop
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Lambda, merge
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
from keras import backend as K
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, recall_score, f1_score, precision_score


# Define constants
OUTPUT_DIR = 'output'
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

isFast = True

dataset_type = 'process'  # or 'process'

if dataset_type == 'domain':
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'domain_results.pkl')
    INPUT_FILE = os.path.join('data', 'domains_spoof.pkl')
    IMAGE_FILE = os.path.join(OUTPUT_DIR, 'domains_roc_curve.png')
    OUTPUT_NAME = 'Domain Spoofing'
elif dataset_type == 'process':
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'process_results.pkl')
    INPUT_FILE = os.path.join('data', 'process_spoof.pkl')
    IMAGE_FILE = os.path.join(OUTPUT_DIR, 'process_roc_curve.png')
    OUTPUT_NAME = 'Process Spoofing'
else:
    raise Exception('Unknown dataset type: %s' % (dataset_type,))

# Utility functions
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y) + K.epsilon(), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)), axis=-1)

def build_model(vocab_size, embedding_dim, input_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dense(32))

    input_a = Input(shape=(input_length,))
    input_b = Input(shape=(input_length,))

    processed_a = model(input_a)
    processed_b = model(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model(input=[input_a, input_b], output=distance)

    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)

    return model

# Data preparation
def prepare_data(data, max_len, char_index):
    def encode_text(text, max_len, char_index):
        return [char_index.get(char, 0) for char in text[:max_len].ljust(max_len)]
    
    X1 = np.array([encode_text(x[0], max_len, char_index) for x in data])
    X2 = np.array([encode_text(x[1], max_len, char_index) for x in data])
    y = np.array([x[2] for x in data])
    
    return X1, X2, y

if not os.path.isfile(OUTPUT_FILE):
    max_epochs = 25
    max_len = 100  # Maximum length of domain/process strings
    embedding_dim = 50  # Dimension of embedding vector

    with open(INPUT_FILE, 'rb') as f:
        data = pickle.load(f)

    # Create character index
    char_set = set(''.join([x[0] + x[1] for x in data['train']]))
    char_index = {char: idx + 1 for idx, char in enumerate(char_set)}
    vocab_size = len(char_index) + 1

    if isFast:
        data['train'] = random.sample(data['train'], 4000)
        data['validate'] = random.sample(data['validate'], 100)
        data['test'] = random.sample(data['test'], 1000)
        max_epochs = 10

    # During training, save the character index
    char_index_file = os.path.join(OUTPUT_DIR, 'char_index.pkl')
    with open(char_index_file, 'wb') as f:
        pickle.dump(char_index, f)

    # Prepare data
    X1_train, X2_train, y_train = prepare_data(data['train'], max_len, char_index)
    X1_valid, X2_valid, y_valid = prepare_data(data['validate'], max_len, char_index)
    X1_test, X2_test, y_test = prepare_data(data['test'], max_len, char_index)

    model = build_model(vocab_size, embedding_dim, max_len)

    # Determine the best number of epochs based on validation AUC
    max_auc = 0
    max_idx = 0
    for i in range(max_epochs):
        model.fit([X1_train, X2_train], y_train, batch_size=8, nb_epoch=1)
        scores = model.predict([X1_valid, X2_valid]).ravel()
        t_auc = roc_auc_score(y_valid, -scores)
        if t_auc > max_auc:
            print('Updated best AUC from %f to %f' % (max_auc, t_auc))
            max_auc = t_auc
            max_idx = i + 1

    # Train on the best number of epochs
    model = build_model(vocab_size, embedding_dim, max_len)
    model.fit([X1_train, X2_train], y_train, batch_size=8, nb_epoch=max_idx)

    # Save the model
    json_string = model.to_json()
    model.save_weights(os.path.join(OUTPUT_DIR, dataset_type + '_bi_lstm.h5'), overwrite=True)
    with open(os.path.join(OUTPUT_DIR, dataset_type + '_bi_lstm.json'), 'wb') as f:
        f.write(json_string.encode('utf-8'))

    # Evaluate the model
    scores = model.predict([X1_test, X2_test]).ravel()
    fpr_siamese, tpr_siamese, _ = roc_curve(y_test, -scores)
    roc_auc_siamese = auc(fpr_siamese, tpr_siamese)

    # Compute additional metrics
    y_pred = (scores < 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy: {:.4f}".format(accuracy))
    print("Recall: {:.4f}".format(recall))
    print("Precision: {:.4f}".format(precision))
    print("F1 Score: {:.4f}".format(f1))

    # Run Edit distance
    scores = [(editdistance.eval(x[0].lower(), x[1].lower()), len(x[0]), 1.0 - x[2]) for x in data['test']]
    y_percent_score = [float(x[0]) / x[1] for x in scores]
    y_score, _, y_test = zip(*scores)
    fpr_ed, tpr_ed, _ = roc_curve(y_test, y_score)
    roc_auc_ed = auc(fpr_ed, tpr_ed)
    fpr_ps, tpr_ps, _ = roc_curve(y_test, y_percent_score)
    roc_auc_ps = auc(fpr_ps, tpr_ps)

    # Run editdistance visual similarity
    scores = [(editdistance.eval(x[0].lower(), x[1].lower()), 1.0 - x[2]) for x in data['test']]
    y_score, y_test = zip(*scores)
    fpr_edvs, tpr_edvs, _ = roc_curve(y_test, [-x for x in y_score])
    roc_auc_edvs = auc(fpr_edvs, tpr_edvs)

    # Store results
    results = {
        'editdistance_vs': {'fpr': fpr_edvs, 'tpr': tpr_edvs, 'auc': roc_auc_edvs},
        'editdistance': {'fpr': fpr_ed, 'tpr': tpr_ed, 'auc': roc_auc_ed},
        'editdistance_percent': {'fpr': fpr_ps, 'tpr': tpr_ps, 'auc': roc_auc_ps},
        'siamese': {
            'fpr': fpr_siamese,
            'tpr': tpr_siamese,
            'auc': roc_auc_siamese,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1': f1
        }
    }

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(results, f)

with open(OUTPUT_FILE, 'rb') as f:
    results = pickle.load(f)

# Make Figures
fig = plt.figure()
plt.plot(results['siamese']['fpr'], results['siamese']['tpr'], 'b', label='Siamese Bi-LSTM (AUC=%0.2f)' % results['siamese']['auc'])
plt.plot(results['editdistance_vs']['fpr'], results['editdistance_vs']['tpr'], 'g', label='Visual edit distance (AUC=%0.2f)' % results['editdistance_vs']['auc'])
plt.plot(results['editdistance']['fpr'], results['editdistance']['tpr'], 'r', label='Edit distance (AUC=%0.2f)' % results['editdistance']['auc'])
plt.plot(results['editdistance_percent']['fpr'], results['editdistance_percent']['tpr'], label='Percent edit distance (AUC=%0.2f)' % results['editdistance_percent']['auc'])
plt.plot([0, 1], [0, 1], 'k', lw=3, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('{} - Receiver Operating Characteristic'.format(OUTPUT_NAME))
plt.legend(loc="lower right")
fig.savefig(IMAGE_FILE)