import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Fetch the service account key JSON file contents
cred = credentials.Certificate('raasta-c542d-firebase-adminsdk-5v3pa-b884299eed.json')

# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app"
})

ref = db.reference("/sensor-data")
# print all sensor data in the database
print(ref.get())