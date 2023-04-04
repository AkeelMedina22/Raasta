import 'package:flutter/material.dart';

myIndicator(context) => Positioned(
              bottom: 120.0,
              right: 9.0,
              child: Container(
                padding: EdgeInsets.all(8.0),
                decoration: BoxDecoration(
                  color: Colors.white,
                  boxShadow: [
                    BoxShadow(
                      color: Colors.grey.withOpacity(0.5),
                      spreadRadius: 1,
                      blurRadius: 3,
                      offset: Offset(0, 3),
                    ),
                  ],
                  borderRadius: BorderRadius.circular(10),
                ),

                child:
                    Column(
                      mainAxisSize: MainAxisSize.max,
                      mainAxisAlignment: MainAxisAlignment.start,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisSize: MainAxisSize.max,
                          children: [Container(
                                width: 20,
                                height: 20,
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  color: Colors.red,
                                ),
                              ),
                          
                            Container(
                              width: 120,
                              height: 30,
                              child: Align(
                                alignment: AlignmentDirectional(1, 0),
                                child: Text('Start/End Pin', textAlign: TextAlign.end)
                              ),
                            ),
                          ],
                        ),

                        Row(
                          mainAxisSize: MainAxisSize.max,
                          children: [
                            Container(
                              width: 6,
                              height: 6,
                            ),
                          ],
                        ),

                        Row(
                          mainAxisSize: MainAxisSize.max,
                          children: [Container(
                                width: 20,
                                height: 20,
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  color: Colors.orange,
                                ),
                              ),
                          
                            Container(
                              width: 120,
                              height: 30,
                              child: Align(
                                alignment: AlignmentDirectional(1, 0),
                                child: Text('Bad Road', textAlign: TextAlign.end)
                              ),
                            ),
                          ],
                        ),

                        Row(
                          mainAxisSize: MainAxisSize.max,
                          children: [
                            Container(
                              width: 6,
                              height: 6,
                            ),
                          ],
                        ),

                        Row(
                          mainAxisSize: MainAxisSize.max,
                          children: [Container(
                                width: 20,
                                height: 20,
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  color: Colors.yellow,
                                ),
                              ),
                          
                            Container(
                              width: 120,
                              height: 30,
                              child: Align(
                                alignment: AlignmentDirectional(1, 0),
                                child: Text('Speedbreaker', textAlign: TextAlign.end)
                              ),
                            ),
                          ],
                        ),

                        Row(
                          mainAxisSize: MainAxisSize.max,
                          children: [
                            Container(
                              width: 6,
                              height: 6,
                            ),
                          ],
                        ),

                        Row(
                          mainAxisSize: MainAxisSize.max,
                          children: [Container(
                                width: 20,
                                height: 20,
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  color: Colors.red,
                                ),
                              ),
                          
                            Container(
                              width: 120,
                              height: 30,
                              child: Align(
                                alignment: AlignmentDirectional(1, 0),
                                child: Text('Pothole', textAlign: TextAlign.end)
                              ),
                            ),
                          ],
                        ),
                      ],
                    )
            ),
  );


