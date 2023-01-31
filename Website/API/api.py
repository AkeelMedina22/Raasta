from flask import Flask, request
from flask_restful import Api, Resource
from firebase import firebase
import uuid
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, support_credentials=True)
key = ""
# connect to firebase
firebase = firebase.FirebaseApplication('https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app/', None)

@app.route('/get_key', methods=['GET'])
def api_key_request():
    # API key - make a token, send it to client
    uuid_str = str(uuid.uuid4())
    global key
    key = uuid_str
    return({"key": key}, 200, {'Access-Control-Allow-Origin': '*'})

@app.route('/get_points', methods=['GET'])
@cross_origin(origin='*')
def get_potholes():
    global key
    # get all pothole points from database and send to client
    if request.headers["Authorization"] == key:
        pothole = []
        result = firebase.get('/pothole-locations', None)
        for key, value in result.items():
            latlong = []
            for key2, value2 in value.items():
                latlong.append(value2)
            pothole.append(latlong)
        print(pothole)
        return({"Pothole" : pothole})
    else:
        print("NO")
        return({"Pothole" : "Invalid key"})

# API needs set of location points in order to return pothole/speedbreaker location points


if __name__ == "__main__":
    # never run debug in production environment, only development mode
    app.run(debug = True)

