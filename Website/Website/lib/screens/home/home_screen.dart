import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import '../../screens/about/about_screen.dart';
import '../../screens/docs/docs_screen.dart';
import '../../widgets/my_appbar.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:raasta_website/main.dart';

var api_key;

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  void initState() {
    super.initState();

    GetKey().then((_) 
    {
      GetPoints();
    });
  }

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;
    return Scaffold(
      appBar: myAppBar(context),
      body: ListView(
        children: [
          Image.asset('assets/home_background.jpg',
              width: double.infinity, height: 300, fit: BoxFit.cover),
          Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              children: [
                Text('RAASTA',
                    style: Theme.of(context)
                        .textTheme
                        .headline3
                        ?.copyWith(color: Colors.brown)),
                SizedBox(
                  width: size.width * 0.5,
                  child: Text(
                    'Pothole detector using sensor data collected from smart phones to assist users in better route planning',
                    style: Theme.of(context)
                        .textTheme
                        .titleLarge
                        ?.copyWith(color: Colors.grey, height: 1.5),
                    textAlign: TextAlign.center,
                  ),
                ),
                Container(
                  margin: const EdgeInsets.symmetric(vertical: 30),
                  decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(20),
                      boxShadow: [
                        BoxShadow(
                            offset: const Offset(5, 10),
                            color:
                                Theme.of(context).primaryColor.withOpacity(0.1),
                            spreadRadius: 3,
                            blurRadius: 5)
                      ]),
                  width: size.width,
                  height: 400,
                  child: FlutterMap(
                    options: MapOptions(
                      center: LatLng(24.8607, 67.0011),
                      zoom: 9.2,
                    ),
                    children: [
                      TileLayer(
                        urlTemplate:
                            "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                        userAgentPackageName:
                            'dev.fleaflet.flutter_map.example',
                      ),
                    ],
                  ),
                ),
                Text(
                  'The goal of Raasta is to assist drivers with route planning, in ways that current navigation applications cannot, for Karachi. It is a community-driven application that collects data through built-in smartphone sensors in real-time to detect and display pothole indicators on a map and inform users about routes in terms of road safety by allowing users to ‘tag’ locations/routes. A user can specify their commute path, which will display data that the user can then analyze to check the number of potholes, pothole locations, and road safety tags to plan their travel accordingly. Through this, a commuter or driver will be able to make better-informed decisions and have a safer and more comfortable commute/driving experience. ',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      color: Colors.grey,
                      height: 1.5,
                      fontWeight: FontWeight.w300),
                )
              ],
            ),
          )
        ],
      ),
    );
  }

  Future<void> GetKey() async {
    String APIURL = "127.0.0.1:5000";

    http.Response response = await http.get(Uri.http(APIURL, '/get_key'));

    var data = jsonDecode(response.body);
    var message = data["key"];
    api_key = message;
    print(api_key);
  }

  Future GetPoints() async {
    String APIURL = "127.0.0.1:5000";

    print(api_key);
    http.Response response = await http.get(Uri.http(APIURL, '/get_points'));

    var data = jsonDecode(response.body);
    var message = data["Pothole"];
    print(message);
  }
}
