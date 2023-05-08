
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
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

GOOGLE_APPLICATION_CREDENTIALS="raasta-c542d-firebase-adminsdk-5v3pa-94bf94e3fb.json"

cred_obj = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
default_app = firebase_admin.initialize_app(cred_obj, {'databaseURL':"https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app/"})

# ref = db.reference("/unlabelled-data/8d8b49ed-9d55-4533-8331-246e7f7b9b20/") # 78 frames

ref = db.reference("/unlabelled-data/09790feb-e3bb-4f78-882b-53743cc20500/")

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

    except:
        pass


# fig = plt.figure()
# axis = plt.axes(xlim =(-50, 50),
#                 ylim =(-50, 50))
 
# line, = axis.plot([], [], lw = 2)
 
# def init():
#     line.set_data([], [])
#     return line,

# xdata, ydata = [], []
 
# def animate(i, axis):
#     t = 0.1 * i
     
#     # x, y values to be plotted
#     x = t * np.sin(t)
#     y = t * np.cos(t)
     
#     # appending values to the previously
#     # empty x and y data holders
#     xdata.append(x)
#     ydata.append(y)
#     line.set_data(xdata, ydata)
     
#     return line,
 
# # calling the animation function    
# anim = animation.FuncAnimation(fig, animate,
#                             init_func = init,
#                             frames = 500,
#                             interval = 20,
#                             blit = True)
 
# # saves the animation in our desktop
# anim.save('accelerometer_anim.gif', writer = 'Pillow', fps = 30)


x = accelerometer_x
y = accelerometer_y
z = accelerometer_z

fig = plt.figure(figsize=(7,4))
ax = fig.add_subplot(111)
ax.set_title("Accelerometer Data")
line, = ax.plot([],[], '-', color='red', lw=0.5)
line.set_label("X-Axis")
line2, = ax.plot([],[],'-', color='green', lw=0.5)
line2.set_label("Y-Axis")
line3, = ax.plot([],[],'-', color='blue', lw=0.5)
line3.set_label("Z-Axis")
ax.set_xlim(0, 500)
ax.set_ylim(-5, 15)

def animate(i, ax):
    i*=5
    line.set_xdata(np.arange(i, i+500))
    line.set_ydata(x[i+500:i+1000])
    line2.set_xdata(np.arange(i, i+500))
    line2.set_ydata(y[i+500:i+1000])
    line3.set_xdata(np.arange(i, i+500))
    line3.set_ydata(z[i+500:i+1000])
    ax.set_xlim(i, 500+i)
    return line,line2,line3

print(len(x))
ani = animation.FuncAnimation(fig, animate, frames=600, fargs=(ax,),
                              interval=0.1)
# plt.show()
ani.save('accelerometer_anim2.gif', writer = 'Pillow', fps = 30)