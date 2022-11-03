import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Fetch the service account key JSON file contents
# go to project settings -> service accounts -> python -> generate new private key. Download the json file and place it in the same folder as this py file. Copy the name of the file and paste below.
cred = credentials.Certificate('')

# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app"
})

ref = db.reference("/sensor-data")
# print all sensor data in the database
print(ref.get())
