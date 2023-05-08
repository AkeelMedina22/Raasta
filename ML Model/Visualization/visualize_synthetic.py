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
import synthia as syn



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
        without_labels = [[i[0][0],i[0][1],i[0][2]] for i in temp]
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


    def augment(window):
        potholes = []
        normals = []
        bads = []
        speedbreakers = []
        new_window = []

        for j in range(len(window)):
            if window[j][1] == "Pothole":
                potholes.append(window[j][0])

            elif window[j][1] == "Normal road":
                normals.append(window[j][0])

            elif window[j][1] == "Bad road":
                bads.append(window[j][0])

            elif window[j][1] == "Speedbreakers":
                speedbreakers.append(window[j][0])

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
        for data in [potholes, normals, bads, speedbreakers]:
            data = np.array(data)
            data = np.array([np.concatenate((data[i][:,0],data[i][:,1],data[i][:,2])) for i in range(data.shape[0])])
            generator = syn.FPCADataGenerator()        
            generator.fit(data, n_fpca_components=180)  
            samples = generator.generate(n_samples=100)   
            synthetic = np.array(samples)
            if count == 0:
                for j in synthetic:
                    j = np.array(j).reshape(60, 3)
                    new_window.append([j, 'Pothole'])
            elif count == 1:
                axs[0].plot(range(60*3), data[0])
                axs[1].plot(range(60*3), synthetic[0])
                for j in synthetic:
                    j = np.array(j).reshape(60, 3)
                    new_window.append([j, 'Normal road'])
            elif count == 2:
                for j in synthetic:
                    j = np.array(j).reshape(60, 3)
                    new_window.append([j, 'Bad road'])
            elif count == 3:
                for j in synthetic:
                    j = np.array(j).reshape(60, 3)
                    new_window.append([j, 'Speedbreakers'])
            count += 1 

        return new_window


    print((p_count, b_count, s_count, n_count))

    # window = augment(window)

    data = np.array(window, dtype=object)

    data = data[np.random.permutation(len(data))]

    train_ratio = 0.9
    sequence_len = data.shape[0]

    train_data = np.array(augment(data[0:int(sequence_len*train_ratio)]), dtype=object)
    test_data = np.array(data[int(sequence_len*train_ratio):], dtype=object)
    plt.show()

    return train_data, test_data

data, longitude, latitude = get_data()
train_data, test_data = train_test_split(data, longitude, latitude)