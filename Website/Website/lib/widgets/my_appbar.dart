import 'package:flutter/material.dart';
import '../screens/about/about_screen.dart';
import '../screens/docs/docs_screen.dart';
import '../screens/home/home_screen.dart';

myAppBar(context) => AppBar(
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
);
