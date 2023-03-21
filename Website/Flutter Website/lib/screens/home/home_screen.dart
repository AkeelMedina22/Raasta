import 'package:raasta_website/main.dart';
import '../../widgets/my_appbar.dart';
import '../../widgets/indicator.dart';
import '../../widgets/my_drawer.dart';
import '../../screens/about/about_screen.dart';
import '../../screens/docs/docs_screen.dart';
import 'dart:async';
import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:tuple/tuple.dart';
import 'package:geolocator/geolocator.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_google_places_sdk/flutter_google_places_sdk.dart' as g_places;
import 'package:flutter_map_marker_cluster/flutter_map_marker_cluster.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:geolocator_web/geolocator_web.dart';
import 'package:google_polyline_algorithm/google_polyline_algorithm.dart';

var api_key = "";

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {

  @override
  void initState() {
    super.initState();
  }

  final GlobalKey<ScaffoldState> _scaffoldKey = new GlobalKey<ScaffoldState>();
  final TextEditingController _searchController = TextEditingController();
  final FocusNode _textFieldFocusNode = FocusNode();
  static DateTime timeAPIwasLastHit = DateTime.now();
  bool _showSuggestions = false;
  late GoogleMapController mapController;
  GeolocatorPlatform geolocator = GeolocatorPlatform.instance;
  var currentPosition;
  bool locationFound = false;
  var targetMarker;
  var lat;
  var long;
  late g_places.FindAutocompletePredictionsResponse predictions;
  List<Tuple2<String, String>> fullTextList = [];
  final places = g_places.FlutterGooglePlacesSdk('AIzaSyA9j3ueqN9J9KHKGJGz6iB5CJtV7x5Cuyc');
  bool _selected = false;
  bool _start = false;
  bool _end = false;
  late LatLng origin;
  late LatLng destination;
  var newlat;
  var newlng;
  String APIURL = "127.0.0.1:5000";
  Polyline? route;
  Set<Marker> result = new Set();

  // FUNCTIONS

  // GET API KEY AND CURRENT LOCATION
  Future<void> GetKey() async {
    if (api_key == "" && !locationFound)
    {
      result = new Set();

      http.Response response = await http.get(Uri.http(APIURL, '/get_key'));
      var data = jsonDecode(response.body);
      var message = data["key"];
      api_key = message;
      print(api_key);
      
      // get current location of user
      currentPosition = await getCurrentLocation();

      // add in marker set
      result.add(
        Marker(
          markerId: MarkerId("Current Location"),flat: true, 
          alpha: 0.5,
          position: currentPosition,
        ),
      );

      // get all detected locations and add in marker set
      result = await getPoints('Pothole', result);
      result = await getPoints('Speedbreaker', result);
      result = await getPoints('BadRoad', result);

      setState(() {
        result = result;
      });

      // print(result);
    } 
  }

  // GET DETECTED LOCATIONS USING API
  Future<Set<Marker>>getPoints(type_point, result) async {
    List<LatLng> p = [];
    http.Response data = await http.get(Uri.http(APIURL, '/get_points/$type_point'), headers: {"Authorization": api_key});
    var data_points = jsonDecode(data.body);
    final points = data_points["Points"];

    for (int i = 0; i < points.length; i++) 
    {
      result.add(
        Marker(
          markerId: MarkerId("$type_point $i"),flat: true, 
          alpha: 0.5,
          position: LatLng(points[i][0], points[i][1]),
        ),
      );
    }
    // print(result);
    return result;
  }

  Future<LatLng> getCurrentLocation() async {
    if (!locationFound)
    {
      Position position = await Geolocator.getCurrentPosition(desiredAccuracy: LocationAccuracy.high);
      setState(() {
      currentPosition = LatLng(position.latitude, position.longitude);
      locationFound = true;
      lat = currentPosition.latitude;
      long = currentPosition.longitude;
      });
    
    }
    return LatLng(currentPosition.latitude, currentPosition.longitude);
  }

  // ROUTE GENERATION
  Future<Polyline> fetchRoute() async {
    final o_lat = origin.latitude;
    final o_lng = origin.longitude;

    final d_lat = destination.latitude;
    final d_lng = destination.longitude;

    // USING API 
    http.Response route = await http.get(Uri.http(APIURL, '/directions/$o_lat/$o_lng/$d_lat/$d_lng'));
    var d = jsonDecode(route.body);
    String polylineCode = d['overview_polyline']['points'];
    List<LatLng> polylinePoints = decodePolyline(polylineCode)
      .map<LatLng>((e) => LatLng.fromJson(e)!)
      .toList();

    // var bounds = LatLngBounds(
    //   southwest: LatLng(
    //     d['bounds']['southwest']['lat'],
    //     d['bounds']['southwest']['lng'],
    //   ),
    //   northeast: LatLng(
    //     d['bounds']['northeast']['lat'],
    //     d['bounds']['northeast']['lng'],
    //   ),
    // );

    // (await _mapController.future).animateCamera(
    //   CameraUpdate.newLatLngBounds(bounds, 100),
    // );
    
    return Polyline(
      polylineId: const PolylineId('route'),
      points:
          polylinePoints.map((e) => LatLng(e.latitude, e.longitude)).toList(),
    );
  }

  // WIDGET TREE BEGINS 
  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;
    return Scaffold(
        key: _scaffoldKey,
        // drawer for start and end point searching
        drawer: myDrawer(context),
        // make new appbar to include drawer just for HOME
        appBar: new AppBar(
          leading: new IconButton(
            icon: new Icon(Icons.menu),
            splashColor: Colors.transparent,
            highlightColor: Colors.transparent,
            hoverColor: Colors.transparent,
            color: Colors.black,
            onPressed: () => _scaffoldKey.currentState?.openDrawer(),
          ),
          backgroundColor: Colors.white,
          elevation: 1,
          title: const Text('RAASTA', style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold),),
          actions: [
            TextButton(onPressed: () {
              Navigator.pushReplacement(context, MaterialPageRoute(builder: (context) => const HomeScreen()));
            }, child: const Text('Home', style: TextStyle(color: Colors.black))),
            TextButton(onPressed: () {
              Navigator.push(context, MaterialPageRoute(builder: (context) => const DocsScreen()));
            }, child: const Text('Docs', style: TextStyle(color: Colors.black))),
            TextButton(onPressed: () {
              Navigator.push(context, MaterialPageRoute(builder: (context) => const AboutScreen()));
            }, child: const Text('About', style: TextStyle(color: Colors.black))),
            const SizedBox(width: 10)
          ],
        ),
        
        // MAP
        body: FutureBuilder(
          future: GetKey(),
          builder: ((context, snapshot) {
            if (locationFound) {
              return Stack(children: [

                      // GOOGLE MAP WIDGET
                      GoogleMap(
                        initialCameraPosition: CameraPosition(
                          target: LatLng(currentPosition.latitude, currentPosition.longitude),
                          zoom: 15.0,
                        ),
                        minMaxZoomPreference:  MinMaxZoomPreference(12, 20),
                        onMapCreated: (controller) {
                          mapController = controller;
                        },
                        markers: result,

                      ),

                      // FLOATING SEARCH BAR WIDGET
                      Positioned(
                        top: 8,
                        left: 250,
                        right: 250,
                        child: Container(
                                  height: 48,
                                  decoration: BoxDecoration(
                                    color: Colors.white,
                                    borderRadius: BorderRadius.circular(24),
                                    boxShadow: [
                                      BoxShadow(
                                        color: Colors.grey.withOpacity(0.5),
                                        spreadRadius: 1,
                                        blurRadius: 4,
                                        offset: Offset(0, 2),
                                      ),
                                    ],
                                  ),
                                  child: GestureDetector(
                                    behavior: HitTestBehavior.opaque,
                                    onTap: () {
                                      // Focus the text field when the user taps the search bar
                                      FocusScope.of(context).requestFocus(_textFieldFocusNode);
                                    },
                                    
                                    child: Row(
                                      children: [
                                        SizedBox(width: 16),
                                        Icon(Icons.search),
                                        SizedBox(width: 8),
                                        Expanded(
                                          child: TextField(
                                            focusNode: _textFieldFocusNode,
                                            controller: _searchController,
                                            onChanged: (searchText) async {
                                              fullTextList = [];
                                              if (_searchController.text.isNotEmpty)
                                              {
                                                final RegExp coordinateRegex1 = RegExp(r'^(-?\d+(\.\d+)?)°?\s*([N]),?\s*(-?\d+(\.\d+)?)°?\s*([E])$');
                                                final RegExp coordinateRegex2= RegExp(r'^-?([1-8]?\d(\.\d+)?|90(\.0+)?),\s*-?((1([0-7]\d)|[1-9]?\d)(\.\d+)?|180(\.0+)?)$');

                                                // USING NEW FLUTTER PLACES SDK LIBRARY
                                                // find the locationBias LatLng bounds
                                                final radius = 1000.00;
                                                //  circumference of the Earth at the equator is approximately 40,075 km
                                                final double distanceInDegrees = (radius / 1000) / (40075 * 360);
                                                final g_places.LatLng southwestBound = g_places.LatLng(lat: lat - distanceInDegrees, lng: long - distanceInDegrees);
                                                final g_places.LatLng northeastBound = g_places.LatLng(lat: lat + distanceInDegrees, lng: long + distanceInDegrees);

                                                predictions = await places.findAutocompletePredictions(_searchController.text, locationBias: g_places.LatLngBounds(southwest: southwestBound, northeast: northeastBound), origin: g_places.LatLng(lat: lat, lng: long));
                                                // print(predictions.predictions);
                                                for (var prediction in predictions.predictions) 
                                                {
                                                  String fullText = prediction.fullText;
                                                  String placeID = prediction.placeId;
                                                  if(prediction.distanceMeters! < 50000)
                                                  {
                                                    if (!fullTextList.contains(Tuple2(fullText, placeID))) 
                                                    {
                                                      fullTextList.add(Tuple2(fullText, placeID));
                                                    }
                                                  } 
                                                }
                                                setState(() {
                                                  _showSuggestions = true;
                                                });
                                                
                                              }
                                              else
                                              {
                                                setState(() {
                                                  _showSuggestions = false;
                                                  _selected = false;
                                                });
                                              }
                                            },
                                            decoration: InputDecoration(
                                            hintText: 'Search',
                                            border: InputBorder.none,
                                            ),
                                          ),
                                        ),
                                        // Search bar close icon here
                                        GestureDetector(
                                          behavior: HitTestBehavior.opaque,
                                          onTap: () {
                                            _searchController.clear();
                                            fullTextList = [];
                                            setState(() {
                                              _showSuggestions = false;
                                              _selected = false;
                                            });
                                          },
                                          child: Icon(Icons.close),
                                        ),
                                        SizedBox(width: 16),
                                      ],
                                    ),
                                  ),
                                ),
                              ),

                              // SUGGESTIONS WIDGET
                              if (_showSuggestions)
                              Positioned(
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
                                  itemCount: fullTextList.length,
                                  itemBuilder: (context, index) {
                                    return ListTile(
                                      title: Text(fullTextList[index].item1),
                                      onTap: () async {
                                        _searchController.text = fullTextList[index].item1;

                                        // USING NEW FLUTTER PLACES SDK LIBRARY
                                        final place_info = await places.fetchPlace(fullTextList[index].item2, fields: [g_places.PlaceField.Location]);
                                        final latLng = place_info.place!.latLng;
                                        newlat = latLng!.lat;
                                        newlng = latLng.lng;
                                        print(newlat);
                                        print(newlng);
                                        
                                      // move to that location

                                      fullTextList = [];
                                      setState(() {
                                          _showSuggestions = false;
                                          _selected = true;
                                        });
                                      },
                                    );
                                  },
                                ),
                                ),
                              ),

                              // START, END, RESET BUTTONS
                              if (_selected && !_start && !_end)
                              Positioned(
                                top: 65,
                                left: 600,
                                right: 600,
                                child: ElevatedButton(
                                onPressed: () async {
                                  // move to location and place marker
                                  origin = LatLng(newlat, newlng);
                                  _searchController.clear();
                                  setState(() {
                                    _start = true;
                                    _selected = false;
                                  }); 
                                },
                                child: const Text("Start"),
                              ),
                              ),

                              if (_selected && _start && !_end)
                              Positioned(
                                top: 65,
                                left: 600,
                                right: 600,
                                child: ElevatedButton(
                                onPressed: () async {
                                  destination = LatLng(newlat, newlng);
                                  // start and end location must be different
                                  if (origin.latitude != destination.latitude && origin.longitude != destination.longitude)
                                  {
                                    _searchController.clear();
                                    // generate route
                                    route = await fetchRoute();
                                    //
                                    setState(() {
                                      _end = true;
                                      _selected = false;
                                    }); 
                                  }
                                  else
                                  {
                                    // error dialog box appears
                                    // once user clicks ok, search box clears
                                    setState(() {
                                      _end = false;
                                      _selected = false;
                                    }); 
                                  }
                                },
                                child: const Text("End"),
                              ),
                              ),

                              if (_start && _end)
                              Positioned(
                                bottom: 640.0,
                                right: 35.0,
                                child: ElevatedButton(
                                onPressed: () async {
                                  // move to current location
                                  
                                  setState(() {
                                    _selected = false;
                                    _start = false;
                                    _end = false;
                                    route = null;
                                  }); 

                                },
                                child: const Text("Reset"),
                              ),
                              ),
                              
                              // add floating box for marker indicators
                              myIndicator(context),
                            ],
                  );
            } else {
              return Scaffold(body: Center(child: CircularProgressIndicator()));
            }
          }),
        ),
      );
  }
}
