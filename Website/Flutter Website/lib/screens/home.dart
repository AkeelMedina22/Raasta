import 'package:flutter/material.dart';
import 'package:raasta_google_maps/widgets/indicator.dart';
import 'package:raasta_google_maps/widgets/map_with_components.dart';
import 'package:raasta_google_maps/classes/api.dart';
import '../../screens/about.dart';
import '../../screens/docs.dart';


class Home extends StatefulWidget {
  const Home({super.key});

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {

  @override
  Widget build(BuildContext context) {
    final GlobalKey<ScaffoldState> _scaffoldKey = new GlobalKey<ScaffoldState>();
    final size = MediaQuery.of(context).size;

    return Scaffold(
      key: _scaffoldKey,
        body: const MapWithComponents(),
    );
  }
}
