from flask import Flask, request
from flask_restful import Api, Resource
from firebase import firebase
import uuid

app = Flask(__name__)
key = ""
# connect to firebase
firebase = firebase.FirebaseApplication('https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app/', None)

@app.route('/get_key', methods=['GET'])
def api_key_request():
    # API key - make a token, send it to client
    uuid_str = str(uuid.uuid4())
    key = uuid_str
    return({"key": uuid_str}, 200, {'Access-Control-Allow-Origin': '*'})

@app.route('/get_points', methods=['GET'])
def get_potholes():
    # get all pothole points from database and send to client
    print(request.headers)
    pothole = []
    result = firebase.get('/pothole-locations', None)
    for key, value in result.items():
        pothole.append(str(value))
    return({"Pothole" : pothole}, 200, {'Access-Control-Allow-Origin': '*'})

    # if request.headers.get["Authorization"] == key:
    #     result = firebase.get('/pothole-locations', None)
    #     for key, value in result.items():
    #         pothole.append(str(value))
    #     return({"Pothole" : pothole}, 200, {'Access-Control-Allow-Origin': '*'})
    # else:
    #     return({"Pothole" : "Incorrect API Key"}, 200, {'Access-Control-Allow-Origin': '*'})

# API needs set of location points in order to return pothole/speedbreaker location points


if __name__ == "__main__":
    # never run debug in production environment, only development mode
    app.run(debug = True)

