import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import '../../screens/about/about_screen.dart';
import '../../screens/docs/docs_screen.dart';
import '../../widgets/my_appbar.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:raasta_website/main.dart';
import 'package:flutter_map_marker_cluster/flutter_map_marker_cluster.dart';
import '../../widgets/my_drawer.dart';

var api_key;

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  MapController _mapController = MapController();
  var marker = <Marker>[];

  @override
  void initState() {
    super.initState();
  }

  Future<void> yourAsyncFunction() async {
    Future<List> potholes = GetKey();
    List list = await potholes;
    for (int i = 0; i < list.length; i++) {
      marker.add(Marker(
          point: LatLng(list[i][0], list[i][1]),
          builder: (context) {
            return getMarker();
          }));
    }
  }

  Future<List> GetKey() async {
    String APIURL = "127.0.0.1:5000";

    http.Response response = await http.get(Uri.http(APIURL, '/get_key'));
    var data = jsonDecode(response.body);
    var message = data["key"];
    api_key = message;
    print(api_key);

    http.Response response2 = await http.get(Uri.http(APIURL, '/get_points'),
        headers: {"Authorization": api_key});
    var data2 = jsonDecode(response2.body);
    final potholes = data2["Pothole"];
    // print(potholes);
    List p = [];
    potholes.forEach((d) {
      p.add(d);
    });

    // print(p);
    return Future.value(p);
  }

  final GlobalKey<ScaffoldState> _scaffoldKey = new GlobalKey<ScaffoldState>();

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
          future: yourAsyncFunction(),
          builder: ((context, snapshot) {
            if (snapshot.connectionState == ConnectionState.done) {
              return FlutterMap(
                mapController: _mapController,
                options: MapOptions(
                  center: LatLng(24.9059, 67.1383),
                  maxZoom: 20,
                  minZoom: 12,
                ),
                children: [
                  TileLayer(
                    urlTemplate:
                        "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                    userAgentPackageName: 'dev.fleaflet.flutter_map.example',
                  ),
                  MarkerLayer(
                    markers: marker,
                  ),
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

getMarker() {
  return ClipRRect(
    borderRadius: BorderRadius.circular(100),
    child: Image.network(
      'https://cdn-icons-png.flaticon.com/512/1946/1946770.png',
      // Icons.pin_drop_outlined,
      height: 30,
      fit: BoxFit.cover, // color: Colors.red,
    ),
  );
}
