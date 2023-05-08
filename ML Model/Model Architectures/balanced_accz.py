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
from sklearn.model_selection import KFold
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import synthia as syn
def resample(data, old_fs, new_fs=2):
    t = np.arange(len(data)) / old_fs
    spl = splrep(t, data)
    t1 = np.arange((len(data))*new_fs) / (old_fs*new_fs)
    return splev(t1, spl)

def train_test_split(data):

    window = []
    window_loc = []
    window_size = 60
    stride = 30
    p_count = 0
    s_count = 0
    n_count = 0

    assert len(data) > 2*window_size + 1
    count = 0
    for i in range(0, len(data)-window_size, stride):
        
        temp = data[i:i+window_size]
        without_labels = [[i[0][0], i[0][1], i[0][2]] for i in temp]
        potholes, normalroads, speedbreakers = 0, 0, 0
        for j in temp:
            if j[1] == "Pothole" or j[1] == "Bad Road":
                potholes += 1
            elif j[1] == "Normal Road":
                normalroads += 1
            elif j[1] == "Speedbreaker":
                speedbreakers += 1
        dic = {"potholes" : potholes, "normal roads": normalroads, "speedbreakers": speedbreakers}
      

        if dic["potholes"] >= 1:
            window.append([without_labels, 'Pothole'])
            p_count+=1
        elif dic['speedbreakers'] >= 1:
            window.append([without_labels, 'Speedbreakers'])
            s_count += 1
        elif dic['normal roads'] >= 1:
            window.append([without_labels, 'Normal road'])
            n_count += 1
        else:
            continue

    print((p_count, s_count, n_count))

    data = np.array(window, dtype=object)

    data = data[np.random.permutation(len(data))]

    return data


data, longitude, latitude = get_data()
data = train_test_split(data)
potholes = []
speedbreakers = []
normals = []
# Loop over all training examples
for s, l in data:
    if l == 'Normal road':
        normals.append(s)
    elif l == 'Pothole' or l == 'Bad road':
        potholes.append(s)
    elif l == 'Speedbreakers':
        speedbreakers.append(s)


ind1 = np.random.randint(0, len(normals), size=10)
ind2 = np.random.randint(0, len(potholes), size=10)
ind3 = np.random.randint(0, len(speedbreakers), size=10)

tind1 = np.array(list(filter(lambda x: x not in ind1, np.arange(len(normals)))))
tind2 = np.array(list(filter(lambda x: x not in ind2, np.arange(len(potholes)))))
tind3 = np.array(list(filter(lambda x: x not in ind3, np.arange(len(speedbreakers)))))

test_data = []
test_labels = []

for i in np.array(normals)[ind1]:
    test_data.append(i)
    test_labels.append([1, 0, 0])
for i in np.array(potholes)[ind2]:
    test_data.append(i)
    test_labels.append([0, 1, 0])
for i in np.array(speedbreakers)[ind3]:
    test_data.append(i)
    test_labels.append([0, 0, 1])

train_data = []
train_labels = []

for i in np.array(normals)[tind1]:
    train_data.append(i)
    train_labels.append([1, 0, 0])
for i in np.array(potholes)[tind2]:
    train_data.append(i)
    train_labels.append([0, 1, 0])
for i in np.array(speedbreakers)[tind3]:
    train_data.append(i)
    train_labels.append([0, 0, 1])

print(np.array(test_data).shape)
print(np.array(train_data).shape)

def augment(window, label, n):
        potholes = []
        normals = []
        speedbreakers = []
        new_window = []
        new_label = []

        for j in range(len(window)):
            if label[j] == [1,0,0]:
                potholes.append(window[j])
                new_window.append(window[j])
                new_label.append([1,0,0])

            elif label[j] == [0,1,0]:
                normals.append(window[j])
                new_window.append(window[j])
                new_label.append([0,1,0])

            elif label[j] == [0,0,1]:
                speedbreakers.append(window[j])
                new_window.append(window[j])
                new_label.append([0,0,1])

        count = 0

        for data in [potholes, normals, speedbreakers]:
            data = np.array(data)
            data = np.array([np.concatenate((data[i][:,0],data[i][:,1],data[i][:,2])) for i in range(data.shape[0])])
            print(data.shape)
            generator = syn.FPCADataGenerator()        
            generator.fit(data, n_fpca_components=180)   
            samples = generator.generate(n_samples=max(n-data.shape[0],0))   
            synthetic = np.array(samples)
            if count == 0:
                for j in synthetic:
                    j = np.array((j[0:60], j[60:120], j[120:180])).T
                    new_window.append(j)
                    new_label.append([1,0,0])
            elif count == 1:
                for j in synthetic:
                    j = np.array((j[0:60], j[60:120], j[120:180])).T
                    new_window.append(j)
                    new_label.append([0,1,0])
            elif count == 2:
                for j in synthetic:
                    j = np.array((j[0:60], j[60:120], j[120:180])).T
                    new_window.append(j)
                    new_label.append([0,0,1])
            count += 1 

        return new_window, new_label

train_data, train_labels = augment(train_data, train_labels, 900)
print(np.array(train_data).shape)
print(np.array(train_labels).shape)

# # Convert labels lists to numpy array
X_train = np.array(train_data)
X_test = np.array(test_data)
Y_train = np.array(train_labels).reshape(-1, 3)
Y_test = np.array(test_labels).reshape(-1, 3)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
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
    
# # potential 
# def gradient(x):
#     d = x[1:]-x[:-1]
#     fd = tf.concat([x,x[-1]], 0).expand_dims(1)
#     bd = tf.concat([x[0],x], 0).expand_dims(1)
#     d = tf.concat([fd,bd], 1)
#     return tf.reduce_mean(d,1)

# fit and evaluate a model
def evaluate_model(X_train, Y_train, X_test, Y_test):

    verbose, epochs, batch_size = 1, 150, 32
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], Y_train.shape[1]

    tf.keras.backend.clear_session()

    # Model 1 no grad  cnn
    # il = tf.keras.Input(shape=(n_timesteps,n_features))
    # il1 = Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features))(il)
    # l = Conv1D(filters=32, kernel_size=3, activation='relu')(il1)
    # l = MaxPooling1D(pool_size=2, strides=2)(l)
    # l = Conv1D(filters=32, kernel_size=5, activation='relu')(l)
    # l = tf.keras.layers.Dropout(0.2)(l)
    # l = tf.keras.layers.Flatten()(l)
    # l = Dense(16, activation='relu')(l)
    # l = Dense(n_outputs, activation='softmax')(l)

    # gl = tfa.layers.SpectralNormalization(Conv1D(filters=16, kernel_size=12, activation='relu'), power_iterations=5)(il1)
    # gl = tf.keras.layers.Flatten()(gl)
    # gl = tf.keras.layers.Dropout(0.2)(gl)
    # gl = Dense(16, activation='relu')(gl)
    # gl = Dense(n_outputs, activation='softmax')(gl)

    # con = tf.keras.layers.concatenate([l,gl])
    # final = Dense(n_outputs, activation='softmax')(con)

    # model = tf.keras.Model(il, final)

    # Model 1.5 cnnlstm

    # il = tf.keras.Input(shape=(n_timesteps,n_features))
    # il1 = Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features))(il)
    # l = Conv1D(filters=32, kernel_size=3, activation='relu')(il1)
    # l = MaxPooling1D(pool_size=2, strides=2)(l)
    # l = Conv1D(filters=32, kernel_size=5, activation='relu')(l)
    # l = Conv1D(filters=64, kernel_size=5, activation='relu')(l)
    # l = tf.keras.layers.Dropout(0.4)(l)
    # l = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='relu'))(l)
    # l = tf.keras.layers.Dropout(0.2)(l)
    # l = Dense(64, activation='relu')(l)

    # gl = Conv1D(filters=16, kernel_size=12, activation='relu')(il1)
    # gl = tf.keras.layers.Dropout(0.4)(gl)
    # gl = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='relu'))(gl)
    # gl = tf.keras.layers.Dropout(0.2)(gl)
    # gl = Dense(64, activation='relu')(gl) 

    # con = tf.keras.layers.concatenate([l,gl])
    
    # final = Dense(64, activation='relu')(con)
    # final = Dense(32, activation='relu')(final)
    # final = Dense(n_outputs, activation='softmax')(final)

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
    # l = Dense(64, activation='relu')(l)

    # gl = grad()(il)
    # gl = Conv1D(filters=16, kernel_size=3, activation='relu')(gl)
    # gl = Conv1D(filters=32, kernel_size=3, activation='relu')(gl)
    # gl = MaxPooling1D(pool_size=2, strides=2)(gl)
    # gl = Conv1D(filters=32, kernel_size=5, activation='relu')(gl)
    # gl = Conv1D(filters=64, kernel_size=5, activation='relu')(gl)
    # gl = tf.keras.layers.Dropout(0.4)(gl)
    # gl = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='relu'))(gl)
    # gl = tf.keras.layers.Dropout(0.2)(gl)
    # gl = Dense(64, activation='relu')(gl) 

    # con = tf.keras.layers.concatenate([l,gl])
    
    # final = Dense(64, activation='relu')(con)
    # final = Dense(32, activation='relu')(final)
    # final = Dense(n_outputs, activation='sigmoid')(final)

    # model = tf.keras.Model(il, final)

    # model 3 experiment
    # il = tf.keras.Input(shape=(n_timesteps,n_features))
    # il1 = Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features))(il)
    # l = Conv1D(filters=32, kernel_size=3, activation='relu')(il1)
    # l = MaxPooling1D(pool_size=2, strides=2)(l)
    # l = Conv1D(filters=32, kernel_size=5, activation='relu')(l)
    # l = Conv1D(filters=64, kernel_size=5, activation='relu')(l)

    # gl = Conv1D(filters=32, kernel_size=12, activation='relu')(il)
    # gl = MaxPooling1D(pool_size=2, strides=2)(gl)
    # gl = Conv1D(filters=64, kernel_size=5, activation='relu')(gl)

    # con = tf.keras.layers.concatenate([l,gl])

    # gl = tf.keras.layers.Dropout(0.4)(con)
    # gl = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='relu'))(gl)
    # gl = tf.keras.layers.Dropout(0.2)(gl)
    # gl = Dense(64, activation='relu')(gl) 
    # gl = Dense(32, activation='relu')(gl)

    # final = Dense(n_outputs, activation='sigmoid')(gl)

    # model = tf.keras.Model(il, final)

    #3 streams

    il = tf.keras.Input(shape=(n_timesteps,n_features))
    l1 = Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features))(il)
    l = Conv1D(filters=32, kernel_size=3, activation='relu')(l1)
    l = MaxPooling1D(pool_size=4, strides=4)(l)
    l = Conv1D(filters=32, kernel_size=5, activation='relu')(l)
    l = Conv1D(filters=64, kernel_size=5, activation='relu')(l)
    l = tf.keras.layers.Dropout(0.4)(l)
    l = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='relu'))(l)
    l = tf.keras.layers.Dropout(0.2)(l)
    l = Dense(16, activation='relu')(l)

    gl = grad()(il)
    gl = Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features))(gl)
    gl = Conv1D(filters=32, kernel_size=3, activation='relu')(gl)
    gl = MaxPooling1D(pool_size=4, strides=4)(gl)
    gl = Conv1D(filters=32, kernel_size=5, activation='relu')(gl)
    gl = Conv1D(filters=64, kernel_size=5, activation='relu')(gl)
    gl = tf.keras.layers.Dropout(0.4)(gl)
    gl = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='relu'))(gl)
    gl = tf.keras.layers.Dropout(0.2)(gl)
    gl = Dense(16, activation='relu')(gl) 

    con = tf.keras.layers.concatenate([l,gl])
    
    final = Dense(32, activation='relu')(con)
    final = Dense(n_outputs, activation='softmax')(final)

    model = tf.keras.Model(il, final)


    # model = tf.keras.Sequential([
    # Conv1D(filters=8, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features), kernel_regularizer=tf.keras.regularizers.L1L2(0.0001)),
    # Conv1D(filters=12, kernel_size=5, activation='relu', kernel_regularizer=tf.keras.regularizers.L1L2(0.0001)),
    # MaxPooling1D(pool_size=2, strides=2),
    # MaxPooling1D(pool_size=2, strides=2),
    # tf.keras.layers.Flatten(),
    # tf.keras.layers.Dropout(0.2),
    # Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.L1L2(0.0001)),
    # Dense(n_outputs, activation='softmax')])

    # model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(lr=0.00005), metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(lr=0.0001), metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # fit network
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True)

    tf.keras.models.save_model(model, "Raasta_Model")

    plot_graphs(history, 'categorical_accuracy')
    plot_graphs(history, 'loss')
    plot_graphs(history, 'precision')
    plot_graphs(history, 'recall')

    model_prediction = model.predict(X_test)

    predict = []
    onehot_predict = []
    for i in model_prediction:
        index = np.argmax(i)
        predict.append(index)
        if index == 0:
            onehot_predict.append([1, 0, 0])
        elif index == 1:
            onehot_predict.append([0, 1, 0])
        elif index == 2:
            onehot_predict.append([0, 0, 1])

    print(classification_report(onehot_predict, Y_test))

    # evaluate model
    _, accuracy, precision, recall = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)

    f1 = tfa.metrics.F1Score(num_classes=3)
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
    show_confusion_matrix(confusion_mtx, ["Normal Road", "Pothole", "Speedbreaker"])

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