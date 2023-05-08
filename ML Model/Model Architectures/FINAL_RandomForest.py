import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from scipy import signal
from scipy.interpolate import splev, splrep
import random
import os
import seaborn as sns
sns.set()
import folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import tensorflow_addons as tfa
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

        for j in range(len(window)):
            if window[j][1] == "Pothole" or window[j][1] == "Bad road":
                potholes.append(window[j][0])
                new_window.append(window[j])

            elif window[j][1] == "Normal road":
                normals.append(window[j][0])
                new_window.append(window[j])

            elif window[j][1] == "Speedbreakers":
                speedbreakers.append(window[j][0])
                new_window.append(window[j])

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
            generator.fit(data, n_fpca_components=150)   
            samples = generator.generate(n_samples=max(n-data.shape[0],0))   
            synthetic = np.array(samples)
            if count == 0:
                for j in synthetic:
                    j = np.array((j[0:60], j[60:120], j[120:180])).T
                    new_window.append([j, 'Pothole'])
            elif count == 1:
                for j in synthetic:
                    j = np.array((j[0:60], j[60:120], j[120:180])).T
                    new_window.append([j, 'Normal road'])
            elif count == 2:
                for j in synthetic:
                    j = np.array((j[0:60], j[60:120], j[120:180])).T
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

    train_data = np.array(augment(data[0:int(sequence_len*train_ratio)], 900), dtype=object)
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

# fit and evaluate a model
def evaluate_model(X_train, Y_train, X_test, Y_test):


    X_train = np.array([np.concatenate((X_train[i][:,0],X_train[i][:,1],X_train[i][:,2])) for i in range(X_train.shape[0])])
    X_test = np.array([np.concatenate((X_test[i][:,0],X_test[i][:,1],X_test[i][:,2])) for i in range(X_test.shape[0])])
    Y_train = np.argmax(Y_train, axis=1)
    Y_test = np.argmax(Y_test, axis=1)  
    RF = RandomForestClassifier().fit(X_train, Y_train)
    print("training set score: %f" % RF.score(X_train, Y_train))
    print("test set score: %f" % RF.score(X_test, Y_test))

    m_predict = RF.predict(X_test)

    print("f1 "+ str(f1_score(Y_test, m_predict, zero_division=1, average='macro')))
    print("recall " + str(recall_score(Y_test, m_predict, zero_division=1, average='macro')))
    print("precision " + str(precision_score(Y_test, m_predict, zero_division=1, average='macro')))

    return 

evaluate_model(X_train, Y_train, X_test, Y_test)