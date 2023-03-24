import 'package:flutter/material.dart';
import '../screens/about.dart';
import '../screens/docs.dart';
import '../screens/home.dart';

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
