import preprocessing
import post_processing
import h5py
import time
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, load_model
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from keras.preprocessing.sequence import pad_sequences
import sys

print("=======loading model======")
model = keras.models.load_model('models/128BLSTM2.h5')
print(sys.argv[0])
x_train, y_train, train_ids = preprocessing.load_timit(sys.argv[1], kind='train')
x_test, y_test, test_ids = preprocessing.load_timit(sys.argv[1], kind='test')

label_encoder = LabelEncoder()
label_binarizer = LabelBinarizer()
label_fit = label_encoder.fit(np.ravel(np.vstack(y_train)))
one_hot_labels = label_binarizer.fit(np.arange(0,39))

padded_x_test = pad_sequences(x_test, dtype='float64', padding='pre', truncating='pre', maxlen=500, value=0)

print("predicting sequence...")
prediction = model.predict(padded_x_test)
output = [one_hot_labels.inverse_transform(s) for s in prediction]
output = [label_fit.inverse_transform(s) for s in output]

final_output = post_processing.seq_to_phones(sys.argv[1], sys.argv[2], output, test_ids)