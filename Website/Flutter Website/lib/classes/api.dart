import 'dart:convert';

import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:http/http.dart' as http;
String APIURL = "127.0.0.1:5000";

class API {
  static Future<String> getKey() async {
    // var headers = {'Authorization': 'b08049fe-6f37-470a-9e10-9644fb7a5cb1'};
    // var request = http.Request(
    //   'GET',
    //   Uri.parse('https://raasta.pythonanywhere.com/get_key'),
    // );
    // // request.headers.addAll(headers);

    // http.StreamedResponse response = await request.send();

    // if (response.statusCode == 200) {
    //   return jsonDecode(await response.stream.bytesToString())['key'];
    // } else {
    //   throw Exception(response.reasonPhrase);
    // }

    http.Response response = await http.get(Uri.http(APIURL, '/get_key'));
    var data = jsonDecode(response.body);
    var message = data["key"];
    final api_key = message;
    print(api_key);
    return api_key;
  }

  static Future<List<Map>> getPoints(String key, String type_point) async {
  //   var headers = {'Authorization': key};
  //   var request = http.Request(
  //     'GET',
  //     Uri.parse('https://raasta.pythonanywhere.com/get_points'),
  //   );
  //   request.headers.addAll(headers);

  //   http.StreamedResponse response = await request.send();

  //   if (response.statusCode == 200) {
  //     var x = jsonDecode(await response.stream.bytesToString());
  //     var potholes = x['potholes']
  //         .map(
  //           (e) => {
  //             'type': 'pothole',
  //             'latLng': LatLng(e[0], e[1]),
  //           },
  //         )
  //         .toList();
  //     var badroads = x['badroads']
  //         .map(
  //           (e) => {
  //             'type': 'badroad',
  //             'latLng': LatLng(e[0], e[1]),
  //           },
  //         )
  //         .toList();
  //     var speedbreakers = x['speedbreakers']
  //         .map(
  //           (e) => {
  //             'type': 'speedbreaker',
  //             'latLng': LatLng(e[0], e[1]),
  //           },
  //         )
  //         .toList();
  //     return [...potholes, ...badroads, ...speedbreakers];
  //   } else {
  //     throw Exception(response.reasonPhrase);
  //   }

    http.Response data = await http.get(Uri.http(APIURL, '/get_points/$type_point'), headers: {"Authorization": key});
    var data_points = jsonDecode(data.body);
    final points = data_points["Points"];

    var y = points.map((e) => {'type': type_point, 'latlng': LatLng(e.first, e.last),},).toList();
    return [...y,];
  }

  static Future<LatLng> getPlacePoints(place) async
  {
    http.Response response = await http.get(Uri.http(APIURL, '/get_place_coords/$place'));
    final place_info = jsonDecode(response.body);
    final place_lat = place_info['result']['geometry']['location']['lat'];
    final place_lng = place_info['result']['geometry']['location']['lng'];
    return LatLng(place_lat, place_lng);

  }

  static Future<List> getSuggestions(searchTerm, c_lat, c_lng) async
  {
    http.Response response = await http.get(Uri.http(APIURL, '/autocomplete/$searchTerm/$c_lat/$c_lng'));
    List predictions = jsonDecode(response.body);
    return predictions;
  }

  static Future<dynamic> getDirections(o_lat, o_lng, d_lat, d_lng) async 
  {
    http.Response route = await http.get(Uri.http(APIURL, '/directions/$o_lat/$o_lng/$d_lat/$d_lng'));
    var data = jsonDecode(route.body);
    return data;
  }
}
