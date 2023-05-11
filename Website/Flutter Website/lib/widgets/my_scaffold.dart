import 'dart:core';
import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import '../screens/about.dart';
import '../screens/docs.dart';
import '../screens/home.dart';

class MyScaffold extends StatefulWidget {
  const MyScaffold({Key? key}) : super(key: key);

  @override
  State<MyScaffold> createState() => _MyScaffoldState();
}

class _MyScaffoldState extends State<MyScaffold> {

  static const List<List> tabs = [    
    ['Home', Home()],
    ['Docs', Docs()],
    ['About', About()],
  ];
  int tabIndex = 0;

  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();

  @override
  Widget build(BuildContext context) {
    final List<Widget> tabButtons = List.generate(
      tabs.length,
      (index) => Padding(
        padding: const EdgeInsets.only(right: 10),
        child: TextButton(
          onPressed: () {if (tabs[index][0] == 'Docs') {
          // API DOCUMENTATION LINK
          launchUrl(Uri.parse('https://raasta.pythonanywhere.com/apidocs/'));
        } else {
          setState(() => tabIndex = index);
        }},
          child: Text(
            tabs[index][0],
            style: TextStyle(
              color: index == tabIndex ? Colors.indigo : Colors.black,
            ),
          ),
          style: ButtonStyle(
            overlayColor: MaterialStateProperty.all(Colors.transparent),
          ),
        ),
      ),
    );

    return Scaffold(
      key: _scaffoldKey,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 1,
        title: const Text(
          'RAASTA',
          style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold),
        ),
        actions: tabButtons,
      ),
      body: tabs[tabIndex][1],
    );
  }
}