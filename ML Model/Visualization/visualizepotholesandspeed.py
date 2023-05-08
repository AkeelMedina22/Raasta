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

    def augment(window, n):
        potholes = []
        normals = []
        speedbreakers = []
        new_window = []
        countt = 0
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

            elif window[j][1] == "Speedbreakers" and countt < 20:
                csx = CubicSpline(np.arange(60),np.array(window[j][0])[:,0])
                csy = CubicSpline(np.arange(60),np.array(window[j][0])[:,1])
                csz = CubicSpline(np.arange(60),np.array(window[j][0])[:,2])
                newx = csx(np.arange(0, 60, 0.5))
                newy = csy(np.arange(0, 60, 0.5))
                newz = csz(np.arange(0, 60, 0.5))
                new = np.vstack((newx, newy, newz)).T
                print(new.shape)
                speedbreakers.append(new)
                new_window.append([new, window[j][1]])
                # countt += 1

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

        fig, axs = plt.subplots(2)
        print(np.array(speedbreakers).shape)
        for i in range(1):
            axs[0].plot(range(120), np.array(speedbreakers[i])[:,2], alpha=1.0, color='tab:blue')
            axs[0].set_title("Speedbreaker")
            axs[1].plot(range(120), np.array(potholes[i])[:,2], alpha=1.0, color='tab:blue')
            axs[1].set_title("Pothole")
            
        return new_window


    print((p_count, s_count, n_count))

    # window = augment(window)

    data = np.array(window, dtype=object)

    data = data[np.random.permutation(len(data))]

    train_ratio = 0.85
    sequence_len = data.shape[0]

    train_data = np.array(augment(data[0:int(sequence_len*train_ratio)], 900), dtype=object)
    # test_data = np.array(augment(data[int(sequence_len*train_ratio):], 200), dtype=object)
    plt.tight_layout()
    plt.show()
    return train_data


data, longitude, latitude = get_data()
train_data= train_test_split(data)