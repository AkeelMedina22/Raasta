import 'package:flutter/material.dart';
import 'package:raasta_google_maps/widgets/indicator.dart';
import 'package:raasta_google_maps/widgets/map_with_components.dart';
import 'package:raasta_google_maps/classes/api.dart';
import 'package:raasta_google_maps/widgets/my_drawer.dart';
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
              Navigator.pushReplacement(context, MaterialPageRoute(builder: (context) => const Home()));
            }, child: const Text('Home', style: TextStyle(color: Colors.black))),
            TextButton(onPressed: () {
              Navigator.push(context, MaterialPageRoute(builder: (context) => const Docs()));
            }, child: const Text('Docs', style: TextStyle(color: Colors.black))),
            TextButton(onPressed: () {
              Navigator.push(context, MaterialPageRoute(builder: (context) => const About()));
            }, child: const Text('About', style: TextStyle(color: Colors.black))),
            const SizedBox(width: 10)
          ],
        ),

        body: const MapWithComponents(),
    );



    // return Column(
    //   children: [
    //     Padding(
    //       padding: const EdgeInsets.fromLTRB(20, 0, 20, 40),
    //       child: SizedBox(
    //           height: 480,
    //           child: Stack(
    //             children: [const MapWithComponents(), myIndicator(context)],
    //           )),
    //     ),
    //   ],
    // );
  }
}
