from flask import Flask, request
from flask_restful import Api, Resource
from firebase import firebase
import uuid
from flask_cors import CORS, cross_origin
from flasgger import Swagger
import googlemaps
import jsonify
import requests
import jsonpickle
import numpy as np

app = Flask(__name__)
CORS(app, support_credentials=True)
key = ""
# connect to firebase
firebase = firebase.FirebaseApplication('https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app/', None)
gmaps = googlemaps.Client(key = 'AIzaSyA9j3ueqN9J9KHKGJGz6iB5CJtV7x5Cuyc')

template = {
  "info": {
    "title": "Raasta API",
    "description": "API Documentation for Raasta: An Automated Road Classification System",
  },
}
Swagger(app, template=template)

@app.route('/get_key', methods=['GET'])
def api_key_request():
    """
    Generate API Key
    ---
    tags:
      - name: Authentication

    responses:
      200:
        description: API key generated. This key can be used to authenticate requests to the API and ensure that only authorized users are able to access sensitive data or perform certain actions.
        schema:
          properties:
            key:
              type: string
              description: 128-bit random UUID
      500:
        description: Error generating API key
    """
    # API key - make a token, send it to client
    uuid_str = str(uuid.uuid4())
    global key
    key = uuid_str
    return({"key": key}, 200, {'Access-Control-Allow-Origin': '*'})

@app.route('/get_points/<TypePoints>', methods=['GET'])
@cross_origin(origin='*')
def get_potholes(TypePoints):
    """
    Get specific type of points from the database.
    ---
    tags:
      - name: Points

    parameters:
        - name: TypePoints
          in: query
          type: string
          required: true
          description: A string value that specifies the type of points(Pothole, Speedbreaker).

    responses:
      200:
        description: Retrieve specific type of points in the form of latitude and longitude coordinates from a Firebase Realtime Database and display them on a map as markers.
        schema:
          properties:
            Pothole:
              type: array
              items:
                type: array
                items:
                  type: number
                  description: latitude and longitude co-ordinates
          example: [{"Pothole": [[24.9597674, 67.0627178], [24.9597674, 67.0627178]]}]
      401:
        description: Invalid API key
      500:
        description: Error retrieving points. Invalid type of points specified.
    """
    global key
    # get all pothole points from database and send to client
    print("KEY:", key)
    print("GOT KEY: ", request.headers["Authorization"])

    if request.headers["Authorization"] == key:
      pothole = []
      speedbreaker = []
      #badroad = []

      if TypePoints == "Pothole":
        result = firebase.get('/pothole-locations', None)
        for key1, value in result.items():
            latlong = []
            for key2, value2 in value.items():
                latlong.append(value2)
            pothole.append(latlong)
        return({"Points" : pothole})
      
      elif TypePoints == "Speedbreaker":
        result = firebase.get('/speedbreaker-locations', None)
        for key1, value in result.items():
            latlong = []
            for key2, value2 in value.items():
                latlong.append(value2)
            speedbreaker.append(latlong)
        return({"Points" : speedbreaker})

      else:
        return({"Points" : "Invalid type of points requested"})
          
    else:
        return({"Points" : "Invalid key"})

@app.route('/get_visited', methods=['GET'])
@cross_origin(origin='*')
def get_visited():
    """
    Get visited routes. 
    ---
    tags:
      - name: Points

    responses:
      200:
        description: Get all routes collected and stored in the database.
        schema:
          properties:
            result:
              type: array
              items:
                type: array
                items:
                  type: number
                  description: latitude and longitude co-ordinates
          example: [[[24.9597674, 67.0627178], [24.9597674, 67.0627178]]]
      401:
        description: Invalid API key
      500:
        description: Error retrieving visited routes.
    """
    global key
    print("KEY:", key)
    print("GOT KEY: ", request.headers["Authorization"])

    if request.headers["Authorization"] == key:
        result = firebase.get('/visited/-NUWWhvHtRGpdAJXQrEw', None)
        return({"result" : result})
    else:
        return({"result" : "Invalid key"})

@app.route('/directions/<origin_latitude>/<origin_longitude>/<destination_latitude>/<destination_longitude>', methods=['GET'])
def directions(origin_latitude, origin_longitude, destination_latitude, destination_longitude):
  try:
    result = gmaps.directions((origin_latitude, origin_longitude), (destination_latitude, destination_longitude), mode='driving')
    return result[0]
  except:
     return "Error"

@app.route('/autocomplete/<query>/<lat>/<long>')
def autocomplete(query, lat, long):
  try:
    location = (lat, long)
    result = gmaps.places_autocomplete(query, location = location, radius = 50000, strict_bounds= True)
    return jsonpickle.encode(result)
  except:
    return "Error"


@app.route('/get_place_coords/<place_id>')
def get_place_coords(place_id):
  try:
    place = gmaps.place(place_id, fields=['geometry'])
    return place
  except:
    return "Error"
  
if __name__ == "__main__":
    app.run(debug = True)
