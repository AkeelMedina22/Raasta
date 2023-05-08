import firebase_admin
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import itertools
from firebase_admin import credentials
from firebase_admin import db
from scipy import signal
import folium
import seaborn as sns
sns.set()
def filter(data, fs=10, fc=2.5, order=11):
    # fc = frequency cutoff
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(order, w, 'highpass', analog=False)
    return signal.filtfilt(b, a, data)

GOOGLE_APPLICATION_CREDENTIALS="raasta-c542d-firebase-adminsdk-5v3pa-94bf94e3fb.json"

cred_obj = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
default_app = firebase_admin.initialize_app(cred_obj, {'databaseURL':"https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app/"})

ref = db.reference("/unlabelled-data/2a23cee3-ba61-4d77-bc1c-16698ce8b8ef/")
session_data = ref.get()

accelerometer_data = []
gyroscope_data = []
accelerometer_x = []
accelerometer_y = []
accelerometer_z = []
gyroscope_x = []
gyroscope_y = []
gyroscope_z = []
timestamps = []
latitude = []
longitude = []
this_map = folium.Map(prefer_canvas=True)

for key in sorted(session_data):

    try:
        try:
            timestamps.append(session_data[key]['timestamps'])
        except KeyError:
            timestamps.append(0)

        try:
            accelerometer_x.append(float(session_data[key]['accelerometer-x']))
        except KeyError:
            accelerometer_x.append(0.0)

        try:
            accelerometer_y.append(float(session_data[key]['accelerometer-y']))
        except KeyError:
            accelerometer_y.append(0.0)

        try:
            accelerometer_z.append(float(session_data[key]['accelerometer-z']))
        except KeyError:
            accelerometer_z.append(10.0)
        
        try:
            gyroscope_x.append(float(session_data[key]['gyroscope-x']))
        except KeyError:
            gyroscope_x.append(0.0)

        try:
            gyroscope_y.append(float(session_data[key]['gyroscope-y']))
        except KeyError:
            gyroscope_y.append(0.0)

        try:
            gyroscope_z.append(float(session_data[key]['gyroscope-z']))
        except KeyError:
            gyroscope_z.append(0.0)

        try:
            latitude.append(float(session_data[key]['latitude']))
        except KeyError:
            latitude.append(0)

        try:
            longitude.append(float(session_data[key]['longitude']))
        except KeyError:
            print(key)
            longitude.append(0)

        # folium.CircleMarker(location=[latitude[-1], longitude[-1]],
        #                 radius=2,
        #                 weight=2, color="black").add_to(this_map)
    except:
        pass
#Set the zoom to the maximum possible
# this_map.fit_bounds(this_map.get_bounds())

#Save the map to an HTML file
# this_map.save('folium_visualization_abeer_unlabelled.html')
# new_acc_x= []
# new_acc_y= []
# new_acc_z= []
# for index in range(len(accelerometer_x)):
#         # if accelerometer_x[index] == 0.0 or accelerometer_y[index] == 0.0 or accelerometer_z[index] == 0.0:
#         #     print(accelerometer_x[index], accelerometer_y[index], accelerometer_z[index])

#         # if accelerometer_x[index] == 0.0:
#         #     accelerometer_x[index] = 1e-6
#         # if accelerometer_y[index] == 0.0:
#         #     accelerometer_y[index] == 1e-6
#         # if accelerometer_z[index] == 0.0:
#         #     accelerometer_z[index] == 1e-6

#         alpha = np.arctan(accelerometer_y[index]/accelerometer_z[index])
#         beta = np.arctan((-accelerometer_x[index]) / np.sqrt((accelerometer_y[index]**2) + (accelerometer_z[index]**2)))

#         # OLD RE-ORIENTATION
#         # new_x = (np.cos(beta) * accelerometer_x[index]) + (np.sin(beta) * np.sin(alpha) * accelerometer_y[index]) + (np.cos(alpha) * np.sin(beta) * accelerometer_z[index])
#         # new_y = (np.cos(alpha) * accelerometer_y[index]) - (np.sin(alpha) * accelerometer_z[index])
#         # new_z = (-np.sin(beta) * accelerometer_x[index]) + (np.cos(beta) * np.sin(alpha) * accelerometer_y[index]) + (np.cos(beta) * np.cos(alpha) * accelerometer_z[index])

#         # new_acc_x.append(new_x)
#         # new_acc_y.append(new_y)
#         # new_acc_z.append(new_z)

#         # NEW ORIENTATION
#         acc = np.array([accelerometer_x[index], accelerometer_y[index], accelerometer_z[index]])

#         R_x = np.array([[1, 0, 0],
#             [0, np.cos(alpha), -np.sin(alpha)],
#             [0, np.sin(alpha), np.cos(alpha)]])
        
#         R_y = np.array([[np.cos(beta), 0, np.sin(beta)],
#             [0, 1, 0],
#             [-np.sin(beta), 0, np.cos(beta)]])
        
#         result = np.dot(acc, np.dot(R_x, R_y))
        
#         gamma = np.arctan(result[0] / result[1])
        
#         R_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
#             [np.sin(gamma), np.cos(gamma), 0],
#             [0, 0, 1]])
        
#         new_val = np.dot(result, R_z)

#         new_acc_x.append(new_val[0])
#         new_acc_y.append(new_val[1])
#         new_acc_z.append(new_val[2])

fig, axs = plt.subplots(2, 3)
axs[0,0].plot(range(len(accelerometer_x)), accelerometer_x)
axs[0,0].set_title("Raw X-Axis")
axs[0,1].plot(range(len(accelerometer_y)), accelerometer_y)
axs[0,1].set_title("Raw Y-Axis")
axs[0,2].plot(range(len(accelerometer_z)), accelerometer_z)
axs[0,2].set_title("Raw Z-Axis")
filt_acx = accelerometer_x
filt_acy = accelerometer_y
filt_acz = accelerometer_z
# filt_acx = new_acc_x
# filt_acy = new_acc_y
# filt_acz = new_acc_z
filt_gyx = gyroscope_x
filt_gyy = gyroscope_y
filt_gyz = gyroscope_z
axs[1,0].plot(range(len(filt_acx)), filt_acx)
axs[1,0].set_title("Reoriented X-Axis")
axs[1,1].plot(range(len(filt_acy)), filt_acy)
axs[1,1].set_title("Reoriented Y-Axis")
axs[1,2].plot(range(len(filt_acz)), filt_acz)
axs[1,2].set_title("Reoriented Z-Axis")

plt.tight_layout()
plt.show()
# accelerometer_data = [(i,j,k) for i,j,k in zip(filt_acx, filt_acy, filt_acz)]
# gyroscope_data = [(i,j,k) for i,j,k in zip(filt_gyx, filt_gyy, filt_gyz)]


# c = np.arange(len(latitude))

# fig = plt.figure()
# ax01 = fig.add_subplot(3,3,2)
# ax02 = fig.add_subplot(3,3,4)
# ax03 = fig.add_subplot(3,3,5)
# ax04 = fig.add_subplot(3,3,6)
# ax05 = fig.add_subplot(3,3,7)
# ax06 = fig.add_subplot(3,3,8)
# ax07 = fig.add_subplot(3,3,9)

# ax01.set_title("GPS")
# ax01.scatter(longitude, latitude, c=c, cmap='viridis', s=1)

# ax02.set_facecolor((0.0, 0.5, 1.0, 0.2))
# ax03.set_facecolor((0.0, 0.5, 1.0, 0.2))
# ax04.set_facecolor((0.0, 0.5, 1.0, 0.2))
# ax05.set_facecolor((0.0, 0.5, 1.0, 0.2))
# ax06.set_facecolor((0.0, 0.5, 1.0, 0.2))
# ax07.set_facecolor((0.0, 0.5, 1.0, 0.2))

# # for i in range(len(gyroscope_data)):
# #     if colors[i] == 0:
# #         ax02.axvspan(i-0.5, i+0.5, facecolor='salmon', alpha=1.0)
# #         ax03.axvspan(i-0.5, i+0.5, facecolor='salmon', alpha=1.0)
# #         ax04.axvspan(i-0.5, i+0.5, facecolor='salmon', alpha=1.0)

# #         ax05.axvspan(i-0.5, i+0.5, facecolor='salmon', alpha=1.0)
# #         ax06.axvspan(i-0.5, i+0.5, facecolor='salmon', alpha=1.0)
# #         ax07.axvspan(i-0.5, i+0.5, facecolor='salmon', alpha=1.0)


# ax02.set_title("Gyroscope-X")
# ax02.plot(range(len(gyroscope_data)), [i[0] for i in gyroscope_data], color='darkslategray')
# ax03.set_title("Gyroscope-Y")
# ax03.plot(range(len(gyroscope_data)), [i[1] for i in gyroscope_data], color='darkslategray')
# ax04.set_title("Gyroscope-Z")
# ax04.plot(range(len(gyroscope_data)), [i[2] for i in gyroscope_data], color='darkslategray')

# ax05.set_title("Accelerometer-X")
# ax05.plot(range(len(accelerometer_data)), [i[0] for i in accelerometer_data], color='darkslateblue')
# ax06.set_title("Accelerometer-Y")
# ax06.plot(range(len(accelerometer_data)), [i[1] for i in accelerometer_data], color='darkslateblue')
# ax07.set_title("Accelerometer-Z")
# ax07.plot(range(len(accelerometer_data)), [i[2] for i in accelerometer_data], color='darkslateblue')

# plt.tight_layout()
# plt.savefig('Visualized_datapoint_filter_2andhalf')
# plt.show()
