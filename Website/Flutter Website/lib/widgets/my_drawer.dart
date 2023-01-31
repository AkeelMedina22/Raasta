import 'package:flutter/material.dart';
import '../screens/about/about_screen.dart';
import '../screens/docs/docs_screen.dart';
import '../screens/home/home_screen.dart';

myDrawer(context) => Drawer(
        child: ListView(
            padding: const EdgeInsets.only(
              left: 15,
            ),
            children: <Widget>[
          Row(children: [
            Container(
                height: 60.0,
                child: Row(
                  children: [
                    IconButton(
                      icon: new Icon(Icons.menu),
                      splashColor: Colors.transparent,
                      highlightColor: Colors.transparent,
                      hoverColor: Colors.transparent,
                      color: Colors.black,
                      onPressed: () {
                        Navigator.pop(context);
                      },
                  
                    ),
                    Padding(
                      padding: const EdgeInsets.only(left: 15),
                      child: Text('RAASTA', style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold)),)
                            
                  ],
                ))
          ])
        ]));
