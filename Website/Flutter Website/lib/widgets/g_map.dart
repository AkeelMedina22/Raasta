import 'dart:async';
import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';

extension Tools on Completer<GoogleMapController> {
  Future<LatLng> getLatLng() async {
    GoogleMapController gmc = await future;
    var visibleRegion = await gmc.getVisibleRegion();
    return LatLng(
      (visibleRegion.northeast.latitude + visibleRegion.southwest.latitude) / 2,
      (visibleRegion.northeast.longitude + visibleRegion.southwest.longitude) /
          2,
    );
  }

  Future<void> moveTo(LatLng latLng) async => (await future).moveCamera(
        CameraUpdate.newLatLng(latLng),
      );
}

class GMap extends StatelessWidget {
  static const String apiKey = 'AIzaSyA9j3ueqN9J9KHKGJGz6iB5CJtV7x5Cuyc';

  const GMap({
    required this.markers,
    required this.onMapCreated,
    super.key,
  });
  final List<Marker> markers;
  final void Function(GoogleMapController)? onMapCreated;
  @override
  Widget build(BuildContext context) => Stack(
        children: [
          GoogleMap(
            initialCameraPosition: const CameraPosition(
              target: LatLng(24.9059, 67.1383),
              zoom: 16,
            ),
            onMapCreated: onMapCreated,
            markers: markers.toSet(),
          ),
          Center(
            child: Padding(
              padding: EdgeInsets.only(bottom: IconTheme.of(context).size!),
              child: const Icon(Icons.location_on),
            ),
          ),
        ],
      );
}
