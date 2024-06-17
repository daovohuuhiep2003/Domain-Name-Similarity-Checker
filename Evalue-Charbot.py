"""Sample code for siamese neural net for detecting spoofing attacks"""
from __future__ import with_statement

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pickle
import os
from keras.models import model_from_json
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
import editdistance
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from keras import backend as K
from tqdm import tqdm
import random
import pandas as pd


# Constants
OUTPUT_DIR = 'output'
dataset_type = 'domain'  # or 'process'

if dataset_type == 'domain':
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'domain_results.pkl')
    INPUT_FILE = os.path.join('data', 'malicious_domains.csv')
    IMAGE_FILE = os.path.join(OUTPUT_DIR, 'domains_roc_curve.png')
    OUTPUT_NAME = 'Domain Spoofing'
elif dataset_type == 'process':
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'process_results.pkl')
    INPUT_FILE = os.path.join('data', 'process_spoof.pkl')
    IMAGE_FILE = os.path.join(OUTPUT_DIR, 'process_roc_curve.png')
    OUTPUT_NAME = 'Process Spoofing'
else:
    raise Exception('Unknown dataset type: %s' % (dataset_type,))

# Load character index
char_index_file = os.path.join(OUTPUT_DIR, 'char_index.pkl')
with open(char_index_file, 'rb') as f:
    char_index = pickle.load(f)

# Load model architecture
with open(os.path.join(OUTPUT_DIR, dataset_type + '_bi_lstm.json'), 'r') as f:
    model_json = f.read()
model = model_from_json(model_json)

# Load model weights
model.load_weights(os.path.join(OUTPUT_DIR, dataset_type + '_bi_lstm.h5'))

# Load data
data = pd.read_csv(INPUT_FILE)
data = data.sample(n=1550)  # Randomly sample 1000 entries
print(data.shape)
print(data[:5])

# Data preparation function
def prepare_data(data, max_len, char_index):
    def encode_text(text, max_len, char_index):
        return [char_index.get(char, 0) for char in text[:max_len].ljust(max_len)]
    
    X1 = np.array([encode_text(x.iloc[0], max_len, char_index) for i, x in data.iterrows()])
    X2 = np.array([encode_text(x.iloc[1], max_len, char_index) for i, x in data.iterrows()])
    y = np.array([x.iloc[2] for i, x in data.iterrows()])
    
    return X1, X2, y

max_len = 100  # Maximum length of domain/process strings

# Prepare test data
# Split data into train and test sets
train_data = data.sample(n=1000)  # Sample 1000 entries for training
remaining_data = data.drop(train_data.index)  # Remaining entries after training
validate_data = remaining_data.sample(n=50)
test_data = remaining_data.drop(validate_data.index)  # Remaining entries for testing (500)


with open("./data/domains_spoof.pkl", 'rb') as f:
    data_add = pickle.load(f)

data_add_train = pd.DataFrame(random.sample(data_add['train'], 1000), columns=data.columns)
data_add_validate = pd.DataFrame(random.sample(data_add['validate'], 50), columns=data.columns)


# Concatenate the original train data with the additional train data
train_data = pd.concat([train_data, data_add_train], ignore_index=True)
validate_data = pd.concat([validate_data, data_add_validate], ignore_index=True)


# Prepare training data
X1_train, X2_train, y_train = prepare_data(train_data, max_len, char_index)
X1_test, X2_test, y_test = prepare_data(test_data, max_len, char_index)
X1_valid, X2_valid, y_valid = prepare_data(data_add_validate, max_len, char_index)

# Compile model
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y) + K.epsilon(), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)), axis=-1)

rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)


model.fit([X1_train, X2_train], y_train, batch_size=512, nb_epoch=1)

# Validate that all indices are within the range of the embedding layer
vocab_size = len(char_index) + 1
if np.max(X1_test) >= vocab_size or np.max(X2_test) >= vocab_size:
    raise ValueError("Found index greater than or equal to the vocabulary size.")

# Evaluate the model with tqdm progress bar
batch_size = 512  # You can adjust the batch size as needed
num_batches = int(np.ceil(X1_test.shape[0] / batch_size))

scores = model.predict([X1_test, X2_test]).ravel()

fpr_siamese, tpr_siamese, _ = roc_curve(y_test, -scores)
roc_auc_siamese = auc(fpr_siamese, tpr_siamese)

# Compute additional metrics
y_pred = (scores < 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: {:.4f}".format(accuracy))



# Define constants
OUTPUT_DIR = 'output-charbot'
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
# Save the model
json_string = model.to_json()
model.save_weights(os.path.join(OUTPUT_DIR, dataset_type + '_bi_lstm_cb.h5'), overwrite=True)
with open(os.path.join(OUTPUT_DIR, dataset_type + '_bi_lstm_cb.json'), 'wb') as f:
    f.write(json_string.encode('utf-8'))
