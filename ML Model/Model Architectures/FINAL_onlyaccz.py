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

    # def augment(window):
    #     potholes = []
    #     normals = []
    #     bads = []
    #     speedbreakers = []
    #     new_window = []
    #     n = 300

    #     for i in range(len(window)):
    #         if window[i][1] == "Pothole":
    #             potholes.append([window[i][0], window_loc[i]])
    #             new_window.append(window[i])

    #         elif window[i][1] == "Normal road":
    #             normals.append([window[i][0], window_loc[i]])

    #         elif window[i][1] == "Bad road":
    #             bads.append([window[i][0], window_loc[i]])

    #         elif window[i][1] == "Speedbreakers":
    #             speedbreakers.append([window[i][0], window_loc[i]])
    #             new_window.append(window[i])

    #     p = n-len(potholes)
    #     s = n-len(speedbreakers)

    #     for i in range(abs(p)):
    #         index = int(np.random.random()*len(potholes))
    #         temp = potholes[index][0]
    #         accx = [j[0] for j in temp]
    #         accy = [j[1] for j in temp]
    #         accz = [j[2] for j in temp]
    #         gyx = [j[3] for j in temp]
    #         gyy = [j[4] for j in temp]
    #         gyz = [j[5] for j in temp]
    #         randrange = 1.5
    #         if np.random.random() < 0.5:


    #             new_fs = int(np.random.uniform(2, 4))
    #             _x = signal.resample(signal.resample(accx, len(accx)//new_fs), len(accx))
    #             _y = signal.resample(signal.resample(accy, len(accy)//new_fs), len(accy))
    #             _z = signal.resample(signal.resample(accz, len(accz)//new_fs), len(accz))
    #             _gx = signal.resample(signal.resample(gyx, len(gyz)//new_fs), len(gyx))
    #             _gy = signal.resample(signal.resample(gyy, len(gyy)//new_fs), len(gyy))
    #             _gz = signal.resample(signal.resample(gyz, len(gyz)//new_fs), len(gyz))
    #             new = [[a,b,c,d,e,f] for a,b,c,d,e,f in zip(_x, _y, _z, _gx, _gy, _gz)]
    #             new_window.append([new, 'Pothole'])
    #         else:
    #             _x = accx+np.random.uniform(-randrange, randrange, size=np.array(accx).shape)
    #             _y = accy+np.random.uniform(-randrange, randrange, size=np.array(accx).shape)
    #             _z = accz+np.random.uniform(-randrange, randrange, size=np.array(accx).shape)
    #             _gx = gyx+np.random.uniform(-randrange, randrange, size=np.array(accx).shape)
    #             _gy = gyy+np.random.uniform(-randrange, randrange, size=np.array(accx).shape)
    #             _gz = gyz+np.random.uniform(-randrange, randrange, size=np.array(accx).shape)
    #             new = [[a,b,c,d,e,f] for a,b,c,d,e,f in zip(_x, _y, _z, _gx, _gy, _gz)]
    #             new_window.append([new, 'Pothole'])

    #     for i in range(abs(s)):
    #         index = int(np.random.random()*len(speedbreakers))
    #         temp = speedbreakers[index][0]
    #         accx = [j[0] for j in temp]
    #         accy = [j[1] for j in temp]
    #         accz = [j[2] for j in temp]
    #         gyx = [j[3] for j in temp]
    #         gyy = [j[4] for j in temp]
    #         gyz = [j[5] for j in temp]

    #         randrange = 1.0

    #         if np.random.random() < 0.5:
    #             new_fs = int(np.random.uniform(2, 4))
    #             _x = signal.resample(signal.resample(accx, len(accx)//new_fs), len(accx))
    #             _y = signal.resample(signal.resample(accy, len(accy)//new_fs), len(accy))
    #             _z = signal.resample(signal.resample(accz, len(accz)//new_fs), len(accz))
    #             _gx = signal.resample(signal.resample(gyx, len(gyz)//new_fs), len(gyx))
    #             _gy = signal.resample(signal.resample(gyy, len(gyy)//new_fs), len(gyy))
    #             _gz = signal.resample(signal.resample(gyz, len(gyz)//new_fs), len(gyz))
    #             new = [[a,b,c,d,e,f] for a,b,c,d,e,f in zip(_x, _y, _z, _gx, _gy, _gz)]
    #             new_window.append([new, 'Speedbreakers'])
    #         else:
    #             _x = accx+np.random.uniform(-randrange, randrange, size=np.array(accx).shape)
    #             _y = accy+np.random.uniform(-randrange, randrange, size=np.array(accx).shape)
    #             _z = accz+np.random.uniform(-randrange, randrange, size=np.array(accx).shape)
    #             _gx = gyx+np.random.uniform(-randrange, randrange, size=np.array(accx).shape)
    #             _gy = gyy+np.random.uniform(-randrange, randrange, size=np.array(accx).shape)
    #             _gz = gyz+np.random.uniform(-randrange, randrange, size=np.array(accx).shape)
    #             new = [[a,b,c,d,e,f] for a,b,c,d,e,f in zip(_x, _y, _z, _gx, _gy, _gz)]
    #             new_window.append([new, 'Speedbreakers'])

    #     for i in range(n):
    #         index = int(np.random.random()*len(normals))
    #     for i in range(n):
    #         index = int(np.random.random()*len(bads))
    #         new_window.append([bads[i][0], 'Bad road'])

    #     return new_window


    def augment(window, n):
        potholes = []
        normals = []
        speedbreakers = []
        new_window = []

        from scipy.interpolate import CubicSpline
        for j in range(len(window)):
            if window[j][1] == "Pothole" or window[j][1] == "Bad road":
                csx = CubicSpline(np.arange(60),np.array(window[j][0])[:,0])
                csy = CubicSpline(np.arange(60),np.array(window[j][0])[:,1])
                csz = CubicSpline(np.arange(60),np.array(window[j][0])[:,2])
                newx = csx(np.arange(0, 60, 0.5))
                newy = csy(np.arange(0, 60, 0.5))
                newz = csz(np.arange(0, 60, 0.5))
                new = np.vstack((newx, newy, newz)).T
                potholes.append(new)
                new_window.append([new, window[j][1]])

            elif window[j][1] == "Normal road":
                csx = CubicSpline(np.arange(60),np.array(window[j][0])[:,0])
                csy = CubicSpline(np.arange(60),np.array(window[j][0])[:,1])
                csz = CubicSpline(np.arange(60),np.array(window[j][0])[:,2])
                newx = csx(np.arange(0, 60, 0.5))
                newy = csy(np.arange(0, 60, 0.5))
                newz = csz(np.arange(0, 60, 0.5))
                new = np.vstack((newx, newy, newz)).T
                normals.append(new)
                new_window.append([new, window[j][1]])

            elif window[j][1] == "Speedbreakers":
                csx = CubicSpline(np.arange(60),np.array(window[j][0])[:,0])
                csy = CubicSpline(np.arange(60),np.array(window[j][0])[:,1])
                csz = CubicSpline(np.arange(60),np.array(window[j][0])[:,2])
                newx = csx(np.arange(0, 60, 0.5))
                newy = csy(np.arange(0, 60, 0.5))
                newz = csz(np.arange(0, 60, 0.5))
                new = np.vstack((newx, newy, newz)).T
                speedbreakers.append(new)
                new_window.append([new, window[j][1]])

        count = 0
        # for data in [potholes, normals, bads, speedbreakers]:
        #     synthetic = []
        #     data = np.array(data).transpose(2, 0, 1)
        #     for i in range(6):
        #         generator = syn.CopulaDataGenerator()     
        #         parameterizer = syn.QuantileParameterizer(n_quantiles=100)   
        #         generator.fit(data[i], copula=syn.GaussianCopula(), parameterize_by=parameterizer)  
        #         samples = generator.generate(n_samples=400)   
        #         synthetic.append(samples)
        #     synthetic = np.array(synthetic).transpose(1,2,0)
        #     if count == 0:
        #         for i in synthetic:
        #             new_window.append([i, 'Pothole'])
        #     elif count == 1:
        #         for i in synthetic:
        #             new_window.append([i, 'Normal road'])
        #     elif count == 2:
        #         for i in synthetic:
        #             new_window.append([i, 'Bad road'])
        #     elif count == 3:
        #         for i in synthetic:
        #             new_window.append([i, 'Speedbreakers'])
        #     count += 1 


        for data in [potholes, normals, speedbreakers]:
            data = np.array(data)
            data = np.array([np.concatenate((data[i][:,0],data[i][:,1],data[i][:,2])) for i in range(data.shape[0])])
            generator = syn.FPCADataGenerator()      
            generator.fit(data, n_fpca_components=300)   
            samples = generator.generate(n_samples=max(n-data.shape[0],0))   
            synthetic = np.array(samples)
            if count == 0:
                for j in synthetic:
                    j = np.array((j[0:120], j[120:240], j[240:360])).T
                    new_window.append([j, 'Pothole'])
            elif count == 1:
                for j in synthetic:
                    j = np.array((j[0:120], j[120:240], j[240:360])).T
                    new_window.append([j, 'Normal road'])
            elif count == 2:
                for j in synthetic:
                    j = np.array((j[0:120], j[120:240], j[240:360])).T
                    new_window.append([j, 'Speedbreakers'])
            count += 1 

        # for data in [potholes, normals, speedbreakers]:
        #     data = np.array(data).reshape(-1, 60*6)
        #     generator = syn.CopulaDataGenerator()       
        #     generator.fit(data, copula=syn.GaussianCopula())  
        #     samples = generator.generate(n_samples=max(500-data.shape[0],1), uniformization_ratio=0, stretch_factor=2)   
        #     synthetic = np.array(samples)
        #     if count == 0:
        #         for j in synthetic:
        #             j = np.array(j).reshape(60, 6)
        #             new_window.append([j, 'Pothole'])
        #     elif count == 1:
        #         for j in synthetic:
        #             j = np.array(j).reshape(60, 6)
        #             new_window.append([j, 'Normal road'])
        #     elif count == 2:
        #         for j in synthetic:
        #             j = np.array(j).reshape(60, 6)
        #             new_window.append([j, 'Speedbreakers'])
        #     count += 1 

        return new_window


    print((p_count, s_count, n_count))

    # window = augment(window)

    data = np.array(window, dtype=object)

    data = data[np.random.permutation(len(data))]

    train_ratio = 0.85
    sequence_len = data.shape[0]

    train_data = np.array(augment(data[0:int(sequence_len*train_ratio)], 1000), dtype=object)
    test_data = np.array(augment(data[int(sequence_len*train_ratio):], 200), dtype=object)

    return train_data, test_data


data, longitude, latitude = get_data()
train_data, test_data = train_test_split(data)
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
        training_labels.append([1, 0, 0])
    elif l == 'Pothole' or l == 'Bad road':
        training_labels.append([0, 1, 0])
    elif l == 'Speedbreakers':
        training_labels.append([0, 0, 1])

# Loop over all test examples
for s, l in test_data:
    testing_sequences.append(np.array(s))
    if l == 'Normal road':
        testing_labels.append([1, 0, 0])
    elif l == 'Pothole' or l == 'Bad road':
        testing_labels.append([0, 1, 0])
    elif l == 'Speedbreakers':
        testing_labels.append([0, 0, 1])

# Convert labels lists to numpy array
X_train = np.array(training_sequences)
X_test = np.array(testing_sequences)
Y_train = np.array(training_labels).reshape(-1, 3)
Y_test = np.array(testing_labels).reshape(-1, 3)
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

    # il = tf.keras.Input(shape=(n_timesteps,n_features))
    # l1 = Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features))(il)
    # l = Conv1D(filters=32, kernel_size=3, activation='relu')(l1)
    # l = MaxPooling1D(pool_size=4, strides=4)(l)
    # l = Conv1D(filters=32, kernel_size=5, activation='relu')(l)
    # l = Conv1D(filters=64, kernel_size=5, activation='relu')(l)
    # l = tf.keras.layers.Dropout(0.4)(l)
    # l = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='relu'))(l)
    # l = tf.keras.layers.Dropout(0.2)(l)
    # l = Dense(16, activation='relu')(l)

    # gl = grad()(il)
    # gl = Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features))(gl)
    # gl = Conv1D(filters=32, kernel_size=3, activation='relu')(gl)
    # gl = MaxPooling1D(pool_size=4, strides=4)(gl)
    # gl = Conv1D(filters=32, kernel_size=5, activation='relu')(gl)
    # gl = Conv1D(filters=64, kernel_size=5, activation='relu')(gl)
    # gl = tf.keras.layers.Dropout(0.4)(gl)
    # gl = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='relu'))(gl)
    # gl = tf.keras.layers.Dropout(0.2)(gl)
    # gl = Dense(16, activation='relu')(gl) 

    # # con = tfa.layers.StochasticDepth(survival_probability=0.85)([gl,gl1])
    # # con = tfa.layers.StochasticDepth(survival_probability=0.5)([l,con])
    # con = tf.keras.layers.concatenate([l,gl])
    
    # final = Dense(32, activation='relu')(con)
    # final = Dense(n_outputs, activation='softmax')(final)

    # model = tf.keras.Model(il, final)

    # Comparison

    il = tf.keras.Input(shape=(n_timesteps,n_features))
    l1 = Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features))(il)
    l = Conv1D(filters=32, kernel_size=3, activation='relu')(l1)
    l = MaxPooling1D(pool_size=4, strides=4)(l)
    l = Conv1D(filters=32, kernel_size=5, activation='relu')(l)
    l = Conv1D(filters=64, kernel_size=5, activation='relu')(l)
    l = tf.keras.layers.Dropout(0.4)(l)
    l = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='relu', return_sequences=True))(l)
    l = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='relu'))(l)
    l = tf.keras.layers.Dropout(0.2)(l)
    l = Dense(64, activation='relu')(l)
    final = Dense(32, activation='relu')(l)
    final = Dense(n_outputs, activation='softmax')(final)

    model = tf.keras.Model(il, final)

    # il = tf.keras.Input(shape=(n_timesteps,n_features))
    # l1 = Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features))(il)
    # l = Conv1D(filters=32, kernel_size=3, activation='relu')(l1)
    # l = MaxPooling1D(pool_size=4, strides=4)(l)
    # l = Conv1D(filters=32, kernel_size=5, activation='relu')(l)
    # l = Conv1D(filters=64, kernel_size=5, activation='relu')(l)
    # l = tf.keras.layers.Dropout(0.4)(l)
    # l = tf.keras.layers.Flatten()(l)
    # l = Dense(64, activation='relu')(l)
    # final = Dense(32, activation='relu')(l)
    # final = Dense(n_outputs, activation='softmax')(final)

    # model = tf.keras.Model(il, final)

    # il = tf.keras.Input(shape=(n_timesteps,n_features))
    
    # final = Dense(128, activation='relu')(il)
    # final = Dense(64, activation='relu')(final)
    # final = Dense(32, activation='relu')(final)
    # final = Dense(16, activation='relu')(final)
    # final = Dense(8, activation='relu')(final)
    # final = tf.keras.layers.Flatten()(final)
    # final = Dense(n_outputs, activation='softmax')(final)

    # model = tf.keras.Model(il, final)


    # model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(lr=000005), metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(lr=0.0001), metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    model.summary()
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