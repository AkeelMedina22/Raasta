import 'dart:async';
import 'dart:convert';

import 'package:custom_marker/marker_icon.dart';
import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:google_polyline_algorithm/google_polyline_algorithm.dart';
import 'package:http/http.dart' as http;
import 'package:raasta_google_maps/classes/api.dart';
import 'package:raasta_google_maps/widgets/g_map.dart';
import 'package:raasta_google_maps/widgets/indicator.dart';
import 'package:raasta_google_maps/widgets/search_address.dart';

class MapWithComponents extends StatefulWidget {
  const MapWithComponents({super.key});

  @override
  State<MapWithComponents> createState() => _MapWithComponentsState();
}

class _MapWithComponentsState extends State<MapWithComponents> {
  @override
  void initState() {
    markerFuture = updateMarkers();
    super.initState();
  }

  String APIURL = "raasta.pythonanywhere.com";

  Future<List<Marker>> updateMarkers() async {
    List<Marker> result = [];
    String key = await API.getKey();
    List<Map<dynamic, dynamic>> pothole = await API.getPoints(key, "Pothole");
    List<Map<dynamic, dynamic>> speedbreaker =
        await API.getPoints(key, "Speedbreaker");

    List<Map<dynamic, dynamic>> points = [];
    points.addAll(pothole);
    points.addAll(speedbreaker);

    for (int i = 0; i < points.length; i++) {
      points[i]['icon'] = await MarkerIcon.markerFromIcon(
        Icons.circle,
        points[i]['type'] == 'Speedbreaker'
            ? Colors.yellow
            : points[i]['type'] == "Pothole"
                ? Colors.red 
                    : Colors.blue, 
                  20,
      );
    }

    for (int i = 0; i < points.length; i++) {
      var mid = points[i]['type'];
      result.add(
        Marker(
          icon: points[i]['icon'],
          markerId: MarkerId("$mid $i"),
          position: points[i]['latlng'],
        ),
      );
    }

    return result;
  }

  Future<LatLng> getLatLng() async {
    GoogleMapController gmc = await _mapController.future;
    var visibleRegion = await gmc.getVisibleRegion();
    return LatLng(
      (visibleRegion.northeast.latitude + visibleRegion.southwest.latitude) / 2,
      (visibleRegion.northeast.longitude + visibleRegion.southwest.longitude) /
          2,
    );
  }

  Future<void> animateCamera(LatLng latLng) async {
    var x = await _mapController.future;
    var y = CameraUpdate.newLatLng(latLng);
    try {
      await x.animateCamera(y);
    } catch (e) {
      error(e);
    }
  }

  Polyline? route;
  Future<Polyline> fetchRoute(LatLng start, LatLng stop) async {
    final o_lat = start.latitude;
    final o_lng = start.longitude;

    final d_lat = stop.latitude;
    final d_lng = stop.longitude;

    var data = await API.getDirections(o_lat, o_lng, d_lat, d_lng);
    var bounds = LatLngBounds(
      southwest: LatLng(
        data['bounds']['southwest']['lat'],
        data['bounds']['southwest']['lng'],
      ),
      northeast: LatLng(
        data['bounds']['northeast']['lat'],
        data['bounds']['northeast']['lng'],
      ),
    );
    (await _mapController.future).animateCamera(
      CameraUpdate.newLatLngBounds(bounds, 100),
    );
    String polylineCode = data['overview_polyline']['points'];
    List<LatLng> polylinePoints = decodePolyline(polylineCode)
        .map<LatLng>((e) => LatLng.fromJson(e)!)
        .toList();
    return Polyline(
      polylineId: const PolylineId('route'),
      points:
          polylinePoints.map((e) => LatLng(e.latitude, e.longitude)).toList(),
    );
  }

  Future<void> error(text) => showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: Text(text.toString()),
        ),
      );

  final Completer<GoogleMapController> _mapController =
      Completer<GoogleMapController>();
  final TextEditingController searchController = TextEditingController();
  List<LatLng> potholes = [];
  List<Map<String, String>> searchResults = <Map<String, String>>[];
  LatLng? start;
  LatLng? end;
  String currentlyPicking = '';
  late Future<List<Marker>> markerFuture;
  bool selected = false;
  bool _start = false;
  bool _end = false;
  LatLng? origin;
  LatLng? destination;

  double distance(LatLng c1, LatLng c2) {
    double dy = c1.latitude - c2.latitude;
    double dx = c1.longitude - c2.longitude;
    return dx * dx + dy * dy;
  }

  /// https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
  /// credit to Joshua, https://stackoverflow.com/users/368954
  double distanceFromRoute(Polyline route, LatLng latLng) {
    List<LatLng> points = route.points;
    double minDistance = double.infinity;
    late LatLng nearest;
    for (int i = 0; i < points.length - 1; i++) {
      // s1-s2 makes line segment
      LatLng s1 = points[i];
      LatLng s2 = points[i + 1];

      double a = latLng.latitude - s1.latitude;
      double b = latLng.longitude - s1.longitude;
      double c = s2.latitude - s1.latitude;
      double d = s2.longitude - s1.longitude;

      double dot = a * c + b * d;
      double lenSq = c * c + d * d;
      double param = -1;

      //in case of 0 length line
      if (lenSq != 0) param = dot / lenSq;

      double latOnSegment, lonOnSegment;

      if (param < 0) {
        latOnSegment = s1.latitude;
        lonOnSegment = s1.longitude;
      } else if (param > 1) {
        latOnSegment = s2.latitude;
        lonOnSegment = s2.longitude;
      } else {
        latOnSegment = s1.latitude + param * c;
        lonOnSegment = s1.longitude + param * d;
      }

      double dx = latLng.latitude - latOnSegment;
      double dy = latLng.longitude - lonOnSegment;
      double distanceFromSegment = dx * dx + dy * dy;
      if (distanceFromSegment < minDistance) {
        minDistance = distanceFromSegment;
        nearest = LatLng(latOnSegment, lonOnSegment);
      }
    }
    var result = distance(nearest, latLng) * 10000;
    return result;
  }
  

//show visited functionality here
  List<Polyline>? visited = null;
  @override
  Widget build(BuildContext context) {
    final Widget visitedButton = Stack(
    children: [
      Positioned(
        bottom: 215.0,
        right: 10.0,
        
      child: ElevatedButton(
      onPressed: () async {
        if (visited != null) {
          return setState(() => visited = null);
        }
        showDialog(
          context: context,
          builder: (context) => Dialog(child: Padding(
            padding: const EdgeInsets.all(30),
            child: Text("LOADING \n Note: These are routes on which road data was collected",
            textAlign: TextAlign.center,
            ),
            
          )),
        );
        // get key
        var key = await API.getKey();
        // get visited
        visited = await API.getVisited(key);
        Navigator.of(context).pop();
        // show visited
        setState(() {});
      },
      child: Text(
        visited != null ? "Hide Visited Routes" : "Show Visited Routes",
      ),
    ),
    ),
    ],
    );

    
    final Widget searchBar = MapSearchField(
      searchController: searchController,
      onResultsGenerated: (results) {
        bool areSame = true;
        if (results[0]['name'] == "None") {
          setState(() {
            searchResults = [];
          });
        } else {
          if (results.length != searchResults.length) areSame = false;
          for (int i = 0; i < results.length && areSame; i++) {
            if (results[i].values.first != searchResults[i].values.first ||
                results[i].values.last != searchResults[i].values.last) {
              areSame = false;
            }
          }
          if (!areSame) setState(() => searchResults = results);
        }
      },
    );

    final Widget startButton = Stack(
      children: [
        if (selected && !_start && !_end)
          Positioned(
            top: 65,
            left: 600,
            right: 600,
            child: ElevatedButton(
              onPressed: () async {
                if (start != null) {
                  await animateCamera(start!);
                  setState(() {
                    _start = true;
                    selected = false;
                  });
                  searchResults = [];
                  searchController.clear();
                } else {
                  start = await getLatLng();
                }
              },
              child: const Text("Start"),
            ),
          ),
        if (selected && _start && !_end)
          Positioned(
            top: 65,
            left: 600,
            right: 600,
            child: ElevatedButton(
              onPressed: () async {
                if (end != null) {
                  if (end != start) {
                    await animateCamera(end!);
                    route = await fetchRoute(start!, end!);
                    setState(() {
                      _end = true;
                    });
                    searchResults = [];
                    searchController.clear();
                  } else {
                    // error dialog box should appear
                  }
                } else {
                  end = await getLatLng();
                }
              },
              child: const Text("End"),
            ),
          ),
        if (_start || _end)
          Positioned(
            bottom: 250.0,
            right: 50.0,
            child: ElevatedButton(
              onPressed: () async {
                await animateCamera(LatLng(24.9059, 67.1383));
                setState(() {
                  selected = false;
                  _start = false;
                  _end = false;
                  route = null;
                  start = null;
                  end = null;
                  markerFuture = updateMarkers();
                });
              },
              child: const Text("Reset"),
            ),
          ),
      ],
    );


    final Widget searchResultView = searchResults.isEmpty
        ? const SizedBox.shrink()
        : Positioned(
            top: 56,
            left: 250,
            right: 250,
            child: Card(
              elevation: 4.0,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(8.0),
              ),
              child: ListView.builder(
                shrinkWrap: true,
                itemCount: searchResults.length,
                itemBuilder: (context, index) {
                  return ListTile(
                    title: Text(searchResults[index].values.first),
                    onTap: () async {
                      final place = searchResults[index]['place_id'];
                      LatLng latLng = await API.getPlacePoints(place);
                      await animateCamera(latLng);
                      searchController.text = searchResults[index]['name']!;
                      setState(() {
                        searchResults = [];
                        selected = true;
                      });
                    },
                  );
                },
              ),
            ),
          );


    final Widget gMap = Stack(
      children: [
        FutureBuilder(
          future: markerFuture,
          initialData: [
            if (start != null)
              Marker(
                markerId: const MarkerId('START'),
                position: start!,
                infoWindow: const InfoWindow(title: 'START'),
              ),
            if (end != null)
              Marker(
                markerId: const MarkerId('END'),
                position: end!,
                infoWindow: const InfoWindow(title: 'END'),
              )
          ],
          builder: (context, snapshot) => GoogleMap(
            initialCameraPosition: const CameraPosition(
              target: LatLng(24.9059, 67.1383),
              zoom: 16,
            ),
            polylines: {
              if (route != null) route!,
              if (visited != null) ...visited!
            },
            onMapCreated: (controller) => _mapController.complete(controller),
            markers: {
              if (start != null)
                Marker(
                  markerId: const MarkerId('START'),
                  position: start!,
                  infoWindow: const InfoWindow(title: 'START'),
                ),
              if (end != null)
                Marker(
                  markerId: const MarkerId('END'),
                  position: end!,
                  infoWindow: const InfoWindow(title: 'END'),
                ),
              ...snapshot.data!.where(
                (marker) => route == null
                    ? true
                    : distanceFromRoute(route!, marker.position) < 0.0001,
              ),
            },
          ),
        ),
        // map pin
        Center(
          child: Padding(
            padding: EdgeInsets.only(bottom: IconTheme.of(context).size!),
            child: const Icon(Icons.location_on),
          ),
        ),
      ],
    );

    return Stack(
      children: [
        gMap,
        searchBar,
        visitedButton,
        searchResultView,
        myIndicator(context),
        startButton,
      ],
    );

  }
}
