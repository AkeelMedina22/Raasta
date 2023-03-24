import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:geolocator_web/geolocator_web.dart';
import 'package:raasta_google_maps/classes/current_location.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:raasta_google_maps/classes/api.dart';

class MapSearchField extends StatelessWidget {
  const MapSearchField({
    required this.searchController,
    required this.onResultsGenerated,
    super.key,
  });

  final TextEditingController searchController;
  final Function(List<Map<String, String>> results) onResultsGenerated;
  static DateTime timeAPIwasLastHit = DateTime.now();
  

  @override
  Widget build(BuildContext context) {

    final FocusNode _textFieldFocusNode = FocusNode();
    Future<void> error(text) => showDialog(
          context: context,
          builder: (context) => AlertDialog(
            title: Text(text.toString()),
          ),
        );
        
    Future<List<Map<String, String>>> fetchSuggestions(String searchTerm) async {
      List<Map<String, String>> results = [];

      // final currentPosition = await current_location.getCurrentLocation();
      // final c_lat = currentPosition.latitude;
      // final c_lng = currentPosition.longitude;

      List predictions = await API.getSuggestions(searchTerm, 24.9059, 67.1383);
      Map<String, String> m;
      for (var p in predictions) {
        m = {};
        m['name'] = p['description'];
        m['place_id'] = p['place_id'];
        results.add(m);
      }
      return results;
    }

    // return Container(
    //   margin: const EdgeInsets.all(20),
    //   child: TextField(
    //     controller: searchController,
    //     onChanged: (searchText) async {
    //       var x = await fetchSuggestions(searchText);
    //       onResultsGenerated(x);
    //     },
    //     decoration: InputDecoration(
    //       prefixIcon: const Icon(Icons.search),
    //       filled: true,
    //       fillColor: Colors.white,
    //       suffixIcon: IconButton(
    //         icon: const Icon(Icons.close),
    //         onPressed: () {
    //           searchController.clear();
    //           onResultsGenerated([]);
    //         },
    //       ),
    //       hintText: 'Search...',
    //       border: OutlineInputBorder(
    //         borderRadius: BorderRadius.circular(25),
    //       ),
    //     ),
    //   ),
    // );

    return Positioned(
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
                            controller: searchController,
                            onChanged: (searchText) async {
                              if (searchController.text.isNotEmpty)
                              {
                                print(searchController.text);
                                final RegExp coordinateRegex1 = RegExp(r'^(-?\d+(\.\d+)?)°?\s*([N]),?\s*(-?\d+(\.\d+)?)°?\s*([E])$');
                                final RegExp coordinateRegex2= RegExp(r'^-?([1-8]?\d(\.\d+)?|90(\.0+)?),\s*-?((1([0-7]\d)|[1-9]?\d)(\.\d+)?|180(\.0+)?)$');

                                var x = await fetchSuggestions(searchText);
                            
                                onResultsGenerated(x);
                                
                              }
                              else
                              {
                                print("Nothing");
                                onResultsGenerated([{'name' : "None"}]);
                                
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
                            searchController.clear();
                            onResultsGenerated([]);
                          },
                          child: Icon(Icons.close),
                        ),
                        SizedBox(width: 16),
                      ],
                    ),
                  ),
                ),
              );

  }
}
