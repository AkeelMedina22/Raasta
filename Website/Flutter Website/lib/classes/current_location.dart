import 'dart:convert';

import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:http/http.dart' as http;
import 'package:geolocator/geolocator.dart';
import 'package:geolocator_web/geolocator_web.dart';

class current_location {
  static Future<LatLng> getCurrentLocation() async {
    GeolocatorPlatform geolocator = GeolocatorPlatform.instance;

    Position position = await Geolocator.getCurrentPosition(desiredAccuracy: LocationAccuracy.high);
    final currentPosition_lat = position.latitude;
    final currentPosition_lng = position.longitude;

    return LatLng(currentPosition_lat, currentPosition_lng);
  }
}