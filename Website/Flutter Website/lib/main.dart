import 'package:flutter/material.dart';
import 'package:raasta_google_maps/screens/home.dart';
import 'package:raasta_google_maps/widgets/my_scaffold.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        debugShowCheckedModeBanner: false,
        title: 'Rastaa',
        theme: ThemeData(
          primarySwatch: Colors.indigo,
        ),
        home: const MyScaffold(),
      );
}
