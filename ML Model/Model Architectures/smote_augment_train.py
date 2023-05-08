import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
from newdata import get_data
import matplotlib.pyplot as plt
import numpy as np
# cnn model
from numpy import dstack
from pandas import read_csv
from keras.layers import Dense
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import tensorflow_addons as tfa
from keras.utils import to_categorical
from scipy import signal
from scipy.interpolate import splev, splrep
import os
import seaborn as sns
sns.set()
import folium


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def resample(data, old_fs, new_fs=2):
    t = np.arange(len(data)) / old_fs
    spl = splrep(t, data)
    t1 = np.arange((len(data))*new_fs) / (old_fs*new_fs)
    return splev(t1, spl)

def train_test_split(data, longitude, latitude):

    window = []
    window_loc = []
    window_size = 60
    stride = 30
    p_count = 0
    b_count = 0
    s_count = 0
    n_count = 0

    assert len(data) > 2*window_size + 1
    count = 0
    for i in range(0, len(data)-window_size, stride):
        
        temp = data[i:i+window_size]
        without_labels = [i[0] for i in temp]
        potholes, badroads, normalroads, speedbreakers = 0, 0, 0, 0
        for j in temp:
            if j[1] == "Pothole":
                potholes += 1
            elif j[1] == "Bad Road":
                badroads += 1
            elif j[1] == "Normal Road":
                normalroads += 1
            elif j[1] == "Speedbreaker":
                speedbreakers += 1
        dic = {"potholes" : potholes, "bad roads": badroads, "normal roads": normalroads, "speedbreakers": speedbreakers}
      

        if dic["potholes"] >= 1:
            window.append([without_labels, 'Pothole'])
            p_count+=1
        elif dic['speedbreakers'] >= 1:
            window.append([without_labels, 'Speedbreakers'])
            s_count += 1
        elif dic['bad roads'] >= 1:
            window.append([without_labels, 'Bad road'])
            # print(dic)
            b_count += 1
        elif dic['normal roads'] >= 1:
            window.append([without_labels, 'Normal road'])
            n_count += 1
        else:
            continue

        window_loc.append([np.mean([latitude[j] for j in range(i, i+window_size)]), np.mean([longitude[j] for j in range(i, i+window_size)])])

    def augment(window, window_loc):
        potholes = []
        normals = []
        bads = []
        speedbreakers = []
        new_window = []
        new_window_loc = []
        n = 350


        for i in range(len(window)):
            if window[i][1] == "Pothole":
                potholes.append([window[i][0], window_loc[i]])
                new_window.append(window[i])
                new_window_loc.append(window_loc[i])

            elif window[i][1] == "Normal road":
                normals.append([window[i][0], window_loc[i]])

            elif window[i][1] == "Bad road":
                bads.append([window[i][0], window_loc[i]])

            elif window[i][1] == "Speedbreakers":
                speedbreakers.append([window[i][0], window_loc[i]])
                new_window.append(window[i])
                new_window_loc.append(window_loc[i])

        p = n-len(potholes)
        s = n-len(speedbreakers)

        for i in range(abs(p)):
            index = int(np.random.random()*len(potholes))
            temp = potholes[index][0]
            accx = [j[0] for j in temp]
            accy = [j[1] for j in temp]
            accz = [j[2] for j in temp]
            gyx = [j[3] for j in temp]
            gyy = [j[4] for j in temp]
            gyz = [j[5] for j in temp]

            if np.random.random() < 0.5:

                new_fs = int(np.random.uniform(2, 4))
                _x = signal.resample(signal.resample(accx, len(accx)//new_fs), len(accx))
                _y = signal.resample(signal.resample(accy, len(accy)//new_fs), len(accy))
                _z = signal.resample(signal.resample(accz, len(accz)//new_fs), len(accz))
                _gx = signal.resample(signal.resample(gyx, len(gyz)//new_fs), len(gyx))
                _gy = signal.resample(signal.resample(gyy, len(gyy)//new_fs), len(gyy))
                _gz = signal.resample(signal.resample(gyz, len(gyz)//new_fs), len(gyz))
                new = [[a,b,c,d,e,f] for a,b,c,d,e,f in zip(_x, _y, _z, _gx, _gy, _gz)]
                new_window.append([new, 'Pothole'])
                new_window_loc.append([potholes[index][1][0], potholes[index][1][1]])
            else:
                _x = accx+np.random.uniform(-1, 1, size=np.array(accx).shape)
                _y = accy+np.random.uniform(-1, 1, size=np.array(accx).shape)
                _z = accz+np.random.uniform(-1, 1, size=np.array(accx).shape)
                _gx = gyx+np.random.uniform(-1, 1, size=np.array(accx).shape)
                _gy = gyy+np.random.uniform(-1, 1, size=np.array(accx).shape)
                _gz = gyz+np.random.uniform(-1, 1, size=np.array(accx).shape)
                new = [[a,b,c,d,e,f] for a,b,c,d,e,f in zip(_x, _y, _z, _gx, _gy, _gz)]
                new_window.append([new, 'Pothole'])
                new_window_loc.append([potholes[index][1][0], potholes[index][1][1]])

        for i in range(abs(s)):
            index = int(np.random.random()*len(speedbreakers))
            temp = speedbreakers[index][0]
            accx = [j[0] for j in temp]
            accy = [j[1] for j in temp]
            accz = [j[2] for j in temp]
            gyx = [j[3] for j in temp]
            gyy = [j[4] for j in temp]
            gyz = [j[5] for j in temp]

            if np.random.random() < 0.5:
                new_fs = int(np.random.uniform(2, 4))
                _x = signal.resample(signal.resample(accx, len(accx)//new_fs), len(accx))
                _y = signal.resample(signal.resample(accy, len(accy)//new_fs), len(accy))
                _z = signal.resample(signal.resample(accz, len(accz)//new_fs), len(accz))
                _gx = signal.resample(signal.resample(gyx, len(gyz)//new_fs), len(gyx))
                _gy = signal.resample(signal.resample(gyy, len(gyy)//new_fs), len(gyy))
                _gz = signal.resample(signal.resample(gyz, len(gyz)//new_fs), len(gyz))
                new = [[a,b,c,d,e,f] for a,b,c,d,e,f in zip(_x, _y, _z, _gx, _gy, _gz)]
                new_window.append([new, 'Speedbreakers'])
                new_window_loc.append([speedbreakers[index][1][0], speedbreakers[index][1][1]])
            else:
                _x = accx+np.random.uniform(-1, 1, size=np.array(accx).shape)
                _y = accy+np.random.uniform(-1, 1, size=np.array(accx).shape)
                _z = accz+np.random.uniform(-1, 1, size=np.array(accx).shape)
                _gx = gyx+np.random.uniform(-1, 1, size=np.array(accx).shape)
                _gy = gyy+np.random.uniform(-1, 1, size=np.array(accx).shape)
                _gz = gyz+np.random.uniform(-1, 1, size=np.array(accx).shape)
                new = [[a,b,c,d,e,f] for a,b,c,d,e,f in zip(_x, _y, _z, _gx, _gy, _gz)]
                new_window.append([new, 'Speedbreakers'])
                new_window_loc.append([speedbreakers[index][1][0], speedbreakers[index][1][1]])

        for i in range(n):
            index = int(np.random.random()*len(normals))
            new_window.append([normals[i][0], 'Normal road'])
            new_window_loc.append(normals[i][1])
        for i in range(n):
            index = int(np.random.random()*len(bads))
            new_window.append([bads[i][0], 'Bad road'])
            new_window_loc.append(bads[i][1])

        return new_window, new_window_loc


    print((p_count, b_count, s_count, n_count))
    # max_count = max(p_count, b_count, s_count, n_count)
    # if p_count < max_count:
    #     window = augment(window)
    #window= augment(window)

    data = np.array(window, dtype=object)

    def unison_shuffled_copies(a):
        p = np.random.permutation(len(a))
        return a[p]

    data= unison_shuffled_copies(data)

    train_ratio = 0.9
    sequence_len = data.shape[0]

    train_data = data[0:int(sequence_len*train_ratio)]
    test_data = data[int(sequence_len*train_ratio):]

    return train_data, test_data


data, longitude, latitude = get_data()
train_data, test_data= train_test_split(data, longitude, latitude)

# Initialize sequences and labels lists
training_sequences = []
training_labels = []

testing_sequences = []
testing_labels = []

# Loop over all training examples
for s, l in train_data:
    # print(np.array(s).shape, end = "\n\n\n")
    training_sequences.append(np.array(s))
    if l == 'Normal road':
        training_labels.append([1,0,0,0])
    elif l == 'Pothole':
        training_labels.append([0,1,0,0])
    elif l == 'Bad road':
        training_labels.append([0,0,1,0])
    elif l == 'Speedbreakers':
        training_labels.append([0,0,0,1])

# Loop over all test examples
for s, l in test_data:
    testing_sequences.append(np.array(s))
    if l == 'Normal road':
        testing_labels.append([1,0,0,0])
    elif l == 'Pothole':
        testing_labels.append([0,1,0,0])
    elif l == 'Bad road':
        testing_labels.append([0,0,1,0])
    elif l == 'Speedbreakers':
        testing_labels.append([0,0,0,1])

# Convert labels lists to numpy array
X_train = np.array(training_sequences)
# X_train = X_train.reshape(-1, X_train.shape[1]*X_train.shape[2])
X_test = np.array(testing_sequences)
# X_test = X_test.reshape(-1, X_test.shape[1]*X_test.shape[2])
Y_train = np.array(training_labels)
Y_test = np.array(testing_labels)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

X_train = X_train.transpose(2, 0, 1)
print(X_train.shape)


from imblearn.over_sampling import SMOTE
new_X_train = []
Y_train_new = []
sm = SMOTE()
for i in range(6):
    X_train_temp, Y_train_new = sm.fit_resample(X_train[i], Y_train)
    print(np.array(X_train_temp).shape)
    new_X_train.append(X_train_temp)

# for i in Y_train_new:
#     print(i)
print(np.array(new_X_train).shape)
X_train = np.array(new_X_train).transpose(1,2,0)
print(X_train.shape)
Y_train = np.array(Y_train_new)


# Y_test = LabelEncoder().fit_transform(Y_test)
# sm = SMOTE(random_state=42, k_neighbors=5)
# X_test, Y_test = sm.fit_resample(X_test, Y_test)
# X_test = X_test.reshape(-1, 60, 6)
# Y_test = Y_test.reshape(-1, 1)

# Y_temp = []
# Y_temp1 = []

# for l in Y_train:

#     if l == 0:
#         Y_temp.append([1,0,0,0])
#     elif l == 1:
#         Y_temp.append([0,1,0,0])
#     elif l == 2:
#         Y_temp.append([0,0,1,0])
#     elif l == 3:
#         Y_temp.append([0,0,0,1])

# for l in Y_test:

#     if l == 0:
#         Y_temp1.append([1,0,0,0])
#     elif l == 1:
#         Y_temp1.append([0,1,0,0])
#     elif l == 2:
#         Y_temp1.append([0,0,1,0])
#     elif l == 3:
#         Y_temp1.append([0,0,0,1])

# Y_train = np.array(Y_temp)
# Y_test = np.array(Y_temp1)

n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], Y_train.shape[1]

def plot_graphs(history, string):
    plt.figure(figsize=(7, 3))
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.savefig("main_results/model_"+string)
    plt.show()

class grad(tf.keras.layers.Layer):
    def __init__(self):
      super(grad, self).__init__()

    def call(self, a):
        rght = tf.concat((a[..., 1:], tf.expand_dims(a[..., -1], -1)), -1)
        left = tf.concat((tf.expand_dims(a[...,0], -1), a[..., :-1]), -1)
        ones = tf.ones_like(rght[..., 2:], tf.float32)
        one = tf.expand_dims(ones[...,0], -1)
        divi = tf.concat((one, ones*2, one), -1)
        return (rght-left) / divi
    
# potential 
def gradient(x):
    d = x[1:]-x[:-1]
    fd = tf.concat([x,x[-1]], 0).expand_dims(1)
    bd = tf.concat([x[0],x], 0).expand_dims(1)
    d = tf.concat([fd,bd], 1)
    return tf.reduce_mean(d,1)

# fit and evaluate a model
def evaluate_model(X_train, Y_train, X_test, Y_test):

    verbose, epochs, batch_size = 1, 100, 128
    # n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], 4

    # Model 1 no grad 
    # il = tf.keras.Input(shape=(n_timesteps,n_features))
    # l = Conv1D(filters=16, kernel_size=3, input_shape=(n_timesteps,n_features))(il)
    # l = Conv1D(filters=32, kernel_size=3)(l)
    # l = MaxPooling1D(pool_size=2, strides=2)(l)
    # l = Conv1D(filters=32, kernel_size=5)(l)
    # l = Conv1D(filters=64, kernel_size=5)(l)
    # l = tf.keras.layers.Dropout(0.4)(l)
    # l = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(l)
    # l = tf.keras.layers.Dropout(0.2)(l)
    # l = Dense(64, activation='relu')(l)

    # gl = tf.keras.layers.Conv1DTranspose(filters=16, kernel_size=12, strides=4)(il)
    # gl = Conv1D(filters=32, kernel_size=24, strides=6)(gl)
    # gl = tf.keras.layers.Dropout(0.4)(gl)
    # gl = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(gl)
    # gl = tf.keras.layers.Dropout(0.2)(gl)
    # gl = Dense(64, activation='relu')(gl) 

    # con = tf.keras.layers.concatenate([l,gl])
    
    # final = Dense(64, activation='relu')(con)
    # final = Dense(32, activation='relu')(final)
    # final = Dense(n_outputs, activation='sigmoid')(final)

    # model = tf.keras.Model(il, final)

    # model 2 with grad
    # il = tf.keras.Input(shape=(n_timesteps,n_features))
    # il1 = Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features))(il)
    # l = Conv1D(filters=32, kernel_size=3, activation='relu')(il1)
    # l = MaxPooling1D(pool_size=2, strides=2)(l)
    # l = Conv1D(filters=32, kernel_size=5, activation='relu')(l)
    # l = Conv1D(filters=64, kernel_size=5, activation='relu')(l)
    # l = tf.keras.layers.Dropout(0.4)(l)
    # l = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='relu'))(l)
    # l = tf.keras.layers.Dropout(0.2)(l)
    # l = Dense(32, activation='relu')(l)

    # gl = grad()(il)
    # gl = Conv1D(filters=16, kernel_size=12, activation='relu')(gl)
    # gl = tf.keras.layers.Dropout(0.4)(gl)
    # gl = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='relu'))(gl)
    # gl = tf.keras.layers.Dropout(0.2)(gl)
    # gl = Dense(32, activation='relu')(gl) 

    # con = tf.keras.layers.concatenate([l,gl])
    
    # final = Dense(64, activation='relu')(con)
    # final = Dense(32, activation='relu')(final)
    # final = Dense(n_outputs, activation='sigmoid')(final)

    # model = tf.keras.Model(il, final)

    # model 3 experiment
    il = tf.keras.Input(shape=(n_timesteps,n_features))
    il1 = Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features))(il)
    l = Conv1D(filters=32, kernel_size=3, activation='relu')(il1)
    l = MaxPooling1D(pool_size=2, strides=2)(l)
    l = Conv1D(filters=32, kernel_size=5, activation='relu')(l)
    l = Conv1D(filters=64, kernel_size=5, activation='relu')(l)

    gl = Conv1D(filters=32, kernel_size=12, activation='relu')(il)
    gl = MaxPooling1D(pool_size=2, strides=2)(gl)
    gl = Conv1D(filters=64, kernel_size=5, activation='relu')(gl)

    con = tf.keras.layers.concatenate([l,gl])

    gl = tf.keras.layers.Dropout(0.4)(con)
    gl = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='relu'))(gl)
    gl = tf.keras.layers.Dropout(0.2)(gl)
    gl = Dense(64, activation='relu')(gl) 
    gl = Dense(32, activation='relu')(gl)

    final = Dense(n_outputs, activation='sigmoid')(gl)

    model = tf.keras.Model(il, final)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.experimental.AdamW(), metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # fit network
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True)

    tf.keras.models.save_model(model, "Raasta_Model")

    plot_graphs(history, 'categorical_accuracy')
    plot_graphs(history, 'loss')

    model_prediction = model.predict(X_test)

    predict = []
    onehot_predict = []
    for i in model_prediction:
        index = np.argmax(i)
        predict.append(index)
        if index == 0:
            onehot_predict.append([1, 0, 0, 0])
        elif index == 1:
            onehot_predict.append([0, 1, 0, 0])
        elif index == 2:
            onehot_predict.append([0, 0, 1, 0])
        elif index == 3:
            onehot_predict.append([0, 0, 0, 1])
    print(classification_report(onehot_predict, Y_test))


    # evaluate model
    _, accuracy, precision, recall = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)

    f1 = tfa.metrics.F1Score(num_classes=4)
    f1.update_state(Y_test, model_prediction)
    f1 = f1.result()
    
    def show_confusion_matrix(cm, labels):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, xticklabels=labels, yticklabels=labels, 
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.savefig("main_results/confusion_matrix")
        plt.show()

    test_encode = []
    for i in Y_test:
        index = np.argmax(i)
        test_encode.append(index)

    confusion_mtx = tf.math.confusion_matrix(test_encode, predict)
    show_confusion_matrix(confusion_mtx, ["Normal Road", "Pothole", "Bad Road", "Speedbreaker"])

    return accuracy, precision, recall, f1

# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# repeat experiment
scores = list()
repeats = 1
for r in range(repeats):
    score, precision, recall, f1 = evaluate_model(X_train, Y_train, X_test, Y_test)
    score = score * 100.0
    print('>#%d: Accuracy->%.3f, Precision->%.3f, Recall->%.3f, F1->%.3f' % (r, score, precision, recall, f1[0]))
    scores.append(score)
    # summarize results
    summarize_results(scores)