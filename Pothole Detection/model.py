import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from data import get_data
import firebase_admin
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import itertools
from firebase_admin import credentials
from firebase_admin import db
include = ['Pothole', 'Bad Road', 'Speedbreaker',]
# import scienceplots

# plt.style.use(['science', 'ieee'])
# Print visible devices
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# If you want to run on CPU instead of GPU
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def train_test_split(data, longitude, latitude):

    window = []
    window_loc = []
    window_size = 60
    stride = 30

    assert len(data) > 2*window_size + 1

    for i in range(window_size, len(data)-window_size, stride):
        temp = data[i-window_size:i+window_size]
        flag = 0
        _ = []
        for j in temp:
            if j[1] in include:
                flag = 1
            _.append(j[0])
        if flag:
            window.append([_, 'Pothole'])
        else:
            window.append([_, 'Not Pothole'])
        window_loc.append([latitude[i], longitude[i]])

    data = np.array(window, dtype=object)

    train_ratio = 0.5
    sequence_len = data.shape[0]

    train_data = data[0:int(sequence_len*train_ratio)]
    test_data = data[int(sequence_len*train_ratio):]

    return train_data, test_data, window_loc


data, longitude, latitude = get_data()
train_data, test_data, location = train_test_split(data, longitude, latitude)
# Initialize sentences and labels lists
training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# Loop over all training examples
for s, l in train_data:
    # print(np.array(s).shape, end = "\n\n\n")
    training_sentences.append(np.array(s))
    if l == 'Pothole':
        training_labels.append(1)
    else:
        training_labels.append(0)

# Loop over all test examples
for s, l in test_data:
    testing_sentences.append(np.array(s))
    if l == 'Pothole':
        testing_labels.append(1)
    else:
        testing_labels.append(0)

# Convert labels lists to numpy array
training_sequences_final = np.array(training_sentences)
testing_sequences_final = np.array(testing_sentences)
training_labels_final = np.array(training_labels).reshape(-1, 1)
testing_labels_final = np.array(testing_labels).reshape(-1, 1)

# Plot Utility
# print(testing_sequences_final)

def plot_graphs(history, string):
    plt.figure(figsize=(7, 3))
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.savefig("model_"+string)
    plt.show()

# print(training_sequences_final.shape)
# print(training_labels_final.shape)
# print(testing_sequences_final.shape)
# print(testing_labels_final.shape)

# Model Definition with LSTM
# model_lstm = tf.keras.Sequential([
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, input_shape=(
#         training_sequences_final.shape[1], training_sequences_final.shape[2]), activation='tanh', return_sequences=True)),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
#         64, activation='relu', return_sequences=True)),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='relu')),
#     tf.keras.layers.Dense(10, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# # Model Definition with LSTM
model_lstm = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, input_shape=(
        training_sequences_final.shape[1], training_sequences_final.shape[2]), activation='tanh', return_sequences=True)),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
    #     128, activation='tanh', return_sequences=True)),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
    #     64, activation='tanh', return_sequences=True)),
    # tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        64, activation='relu', return_sequences=True)),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
    #     64, activation='tanh', return_sequences=True)),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
    #     64, activation='tanh', return_sequences=True)),
    # tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='relu')),
    # tf.keras.layers.Dense(40, activation='relu'),
    # tf.keras.layers.Dense(30, activation='relu'),
    # tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# model_lstm = tf.keras.Sequential([
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, input_shape=(
#         training_sequences_final.shape[1], training_sequences_final.shape[2]), activation='tanh', return_sequences=True)),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
#         128, activation='tanh', return_sequences=True)),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
#         64, activation='tanh', return_sequences=True)),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
#         64, activation='tanh', return_sequences=True)),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
#         64, activation='tanh', return_sequences=True)),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
#         64, activation='tanh', return_sequences=True)),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='tanh')),
#     tf.keras.layers.Dense(40, activation='linear'),
#     tf.keras.layers.Dense(30, activation='linear'),
#     tf.keras.layers.Dense(20, activation='linear'),
#     tf.keras.layers.Dense(10, activation='linear'),
#     tf.keras.layers.Dense(1, activation='linear')
# ])

# Set the training parameters
model_lstm.compile(loss='binary_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   metrics=['accuracy'])

NUM_EPOCHS = 10
BATCH_SIZE = 4

# Train the model
history_lstm = model_lstm.fit(x=training_sequences_final, y=training_labels_final,
                              epochs=NUM_EPOCHS,
                              validation_data=(
                                  testing_sequences_final, testing_labels_final),
                              batch_size=BATCH_SIZE)
# Print the model summary
model_lstm.summary()

# plot_graphs(history_lstm, 'accuracy')
# plot_graphs(history_lstm, 'loss')

predict = np.where(model_lstm.predict(training_sequences_final) > 0.5, 1, 0)
# plt.plot(predict, label="predict", linewidth=0.7, alpha=1.0)
# plt.plot([i for i in range(testing_labels_final.shape[0])], [1 if i == 'Pothole' else 0 for i in testing_labels_final], linewidth=0.5, label="true", alpha=0.5)
# plt.xlabel("Timestep")
# plt.ylabel("Value")
# plt.legend()
# plt.show()

pothole_locations = set()
for i in range(len(predict)):
    if predict[i] == 1:
        pothole_locations.add(tuple(location[i]))

print(pothole_locations)
# ref = db.reference("/pothole-locations/")
# session_data = list(ref.get().values())
# # print(session_data)

# old_potholes = set()
# for session in session_data:
#     for key in sorted(session):
#         try:
#             old_potholes.add((float(session[key]['latitude']), float(session[key]['longitude'])))
#         except:
#             old_potholes.add((0.0,0.0))

# new_potholes = list(pothole_locations - old_potholes)

# for i in new_potholes:
#     ref.push().set({"latitude": i[0], "longitude": i[1]})
