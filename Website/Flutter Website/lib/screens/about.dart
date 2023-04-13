import 'package:flutter/material.dart';
import 'package:raasta_google_maps/widgets/indicator.dart';
import 'package:raasta_google_maps/widgets/map_with_components.dart';
import 'package:raasta_google_maps/classes/api.dart';
import '../../screens/about.dart';
import '../../screens/docs.dart';
import '../../screens/home.dart';


class About extends StatefulWidget {
  const About({Key? key}) : super(key: key);

  @override
  State<About> createState() => _AboutState();
}

// inspiration from
// https://smartroadsense.it/project/about/
class _AboutState extends State<About> {
  @override
  Widget build(BuildContext context) {
    final GlobalKey<ScaffoldState> _scaffoldKey =
        new GlobalKey<ScaffoldState>();
    final size = MediaQuery.of(context).size;

    return Scaffold(
      key: _scaffoldKey,

      body: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.max,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: EdgeInsetsDirectional.fromSTEB(16, 12, 16, 0),
              child: Container(
                width: double.infinity,
                decoration: BoxDecoration(),
                child: Padding(
                  padding: EdgeInsetsDirectional.fromSTEB(12, 12, 12, 25),
                  child: Column(
                    mainAxisSize: MainAxisSize.max,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Column(
                        mainAxisSize: MainAxisSize.max,
                        children: [
                          Container(
                            width: double.infinity,
                            decoration: BoxDecoration(),
                            child: Padding(
                              padding:
                                  EdgeInsetsDirectional.fromSTEB(0, 0, 0, 10),
                              child: Row(
                                mainAxisSize: MainAxisSize.max,
                                mainAxisAlignment: MainAxisAlignment.start,
                                crossAxisAlignment: CrossAxisAlignment.center,
                                children: [
                                  Expanded(
                                    child: Column(
                                      mainAxisSize: MainAxisSize.max,
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Row(
                                          mainAxisSize: MainAxisSize.max,
                                          children: [
                                            Column(
                                              mainAxisSize: MainAxisSize.max,
                                              children: [
                                                Text('ABOUT',
                                                    textAlign: TextAlign.start,
                                                    style: TextStyle(
                                                        fontFamily: 'Outfit',
                                                        fontSize: 125,
                                                        letterSpacing: 6,
                                                        fontWeight:
                                                            FontWeight.w800)),
                                                Text('RAASTA',
                                                    textAlign: TextAlign.start,
                                                    style: TextStyle(
                                                        fontFamily: 'Outfit',
                                                        fontSize: 124,
                                                        fontWeight:
                                                            FontWeight.w800)),
                                              ],
                                            ),
                                            Expanded(
                                              child: Padding(
                                                padding: EdgeInsetsDirectional
                                                    .fromSTEB(18, 5, 10, 0),
                                                child: Text(
                                                    'Raasta is an undergraduate final-year project developed at Habib University to address the issue of rapidly deteriorating road quality in Karachi, Pakistan. We have developed a road surface classification system that uses mobile sensor data to detect potholes and other road surface anomalies, providing their location information through a geographical map.',
                                                    textAlign:
                                                        TextAlign.justify,
                                                    style: TextStyle(
                                                      fontFamily: 'Outfit',
                                                      fontSize: 34,
                                                      height: 1.3,
                                                    )),
                                              ),
                                            ),
                                          ],
                                        ),
                                      ],
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ),
                      Padding(
                        padding: EdgeInsetsDirectional.fromSTEB(0, 60, 0, 80),
                        child: Row(
                          mainAxisSize: MainAxisSize.max,
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Expanded(
                              child: Text(
                                  'The system utilizes a custom CNN-BiLSTM architecture for multi-class classification,\n leveraging a tri-axial accelerometer, gyroscope, and GPS receiver.',
                                  textAlign: TextAlign.center,
                                  style: TextStyle(
                                      fontFamily: 'Outfit',
                                      fontSize: 28,
                                      fontWeight: FontWeight.w800)),
                            ),
                          ],
                        ),
                      ),
                      Column(
                        mainAxisSize: MainAxisSize.max,
                        children: [
                          Padding(
                            padding:
                                EdgeInsetsDirectional.fromSTEB(0, 0, 0, 20),
                            child: Container(
                              width: double.infinity,
                              decoration: BoxDecoration(),
                              child: Padding(
                                padding:
                                    EdgeInsetsDirectional.fromSTEB(0, 0, 0, 10),
                                child: Row(
                                  mainAxisSize: MainAxisSize.max,
                                  mainAxisAlignment: MainAxisAlignment.start,
                                  crossAxisAlignment: CrossAxisAlignment.center,
                                  children: [
                                    Expanded(
                                      child: Column(
                                        mainAxisSize: MainAxisSize.max,
                                        crossAxisAlignment:
                                            CrossAxisAlignment.start,
                                        children: [
                                          Row(
                                            mainAxisSize: MainAxisSize.max,
                                            crossAxisAlignment:
                                                CrossAxisAlignment.start,
                                            children: [
                                              Expanded(
                                                child: Align(
                                                  alignment:
                                                      AlignmentDirectional(
                                                          -1, 0.15),
                                                  child: Padding(
                                                      padding:
                                                          EdgeInsetsDirectional
                                                              .fromSTEB(
                                                                  0, 0, 0, 10),
                                                      child: Text(
                                                        'OUR APPROACH',
                                                        textAlign:
                                                            TextAlign.start,
                                                        style: TextStyle(
                                                            fontFamily:
                                                                'Outfit',
                                                            fontSize: 55,
                                                            fontWeight:
                                                                FontWeight
                                                                    .w800),
                                                      )),
                                                ),
                                              ),
                                            ],
                                          ),
                                          Row(
                                            mainAxisSize: MainAxisSize.max,
                                            children: [
                                              Expanded(
                                                child: Padding(
                                                  padding: EdgeInsetsDirectional
                                                      .fromSTEB(0, 0, 20, 0),
                                                  child: Text(
                                                      'Project Raasta caters to the problem of the unpredictability of Karachi’s road conditions that can severely impact the commuters and drivers of the city. This mainly includes potholes since they make the commuters’ journey unpleasant and can even cause damage to vehicles’ tires, suspensions, and wheels. Whether road damage is due to rainfall or construction in the area, it can be of great inconvenience. Raasta hopes to help you in making route planning more efficient.Our approach for project Raasta is a real-time pothole identification and detection system using mobile sensing technology. Leveraging sensors like the accelerometer, gyroscope, and GPS found in smartphones, road data is collected and used to construct a pothole-detection system. The model used in this project employs a deep-learning strategy to perform multi-class classification to detect the presence of potholes, long segments of roads with bad surfaces, and speedbreakers and provide the corresponding location information of said anomalies to users displayed on the map using APIs.',
                                                      textAlign:
                                                          TextAlign.justify,
                                                      style: TextStyle(
                                                          fontFamily: 'Outfit',
                                                          fontSize: 14,
                                                          fontWeight:
                                                              FontWeight.w600)),
                                                ),
                                              ),
                                            ],
                                          ),
                                        ],
                                      ),
                                    ),
                                    Padding(
                                      padding: EdgeInsetsDirectional.fromSTEB(
                                          0, 0, 10, 0),
                                      child: Image.network(
                                        'https://picsum.photos/seed/286/600',
                                        width: 300,
                                        height: 300,
                                        fit: BoxFit.cover,
                                      ),
                                    ),
                                    Image.network(
                                      'https://picsum.photos/seed/912/600',
                                      width: 300,
                                      height: 300,
                                      fit: BoxFit.cover,
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                      Row(
                        mainAxisSize: MainAxisSize.max,
                        children: [
                          Expanded(
                            child: Padding(
                              padding:
                                  EdgeInsetsDirectional.fromSTEB(0, 0, 0, 20),
                              child: Text('DATA AVAILABILITY',
                                  style: TextStyle(
                                      fontFamily: 'Outfit',
                                      fontSize: 55,
                                      fontWeight: FontWeight.w800)),
                            ),
                          ),
                        ],
                      ),
                      Padding(
                        padding: EdgeInsetsDirectional.fromSTEB(0, 0, 0, 20),
                        child: Row(
                          mainAxisSize: MainAxisSize.max,
                          children: [
                            Container(
                              width: 520.1,
                              height: 149,
                              decoration: BoxDecoration(
                                color: Colors.grey,
                                borderRadius: BorderRadius.circular(5),
                                border: Border.all(
                                  color: Colors.black,
                                ),
                              ),
                              child: Row(
                                mainAxisSize: MainAxisSize.max,
                                children: [
                                  Expanded(
                                    child: Column(
                                      mainAxisSize: MainAxisSize.max,
                                      mainAxisAlignment:
                                          MainAxisAlignment.center,
                                      children: [
                                        Text(
                                          'Hello World',
                                        ),
                                        Text(
                                          'Hello World',
                                        ),
                                      ],
                                    ),
                                  ),
                                  Expanded(
                                    child: Column(
                                      mainAxisSize: MainAxisSize.max,
                                      mainAxisAlignment:
                                          MainAxisAlignment.center,
                                      children: [
                                        Text(
                                          'Hello World',
                                        ),
                                        Text(
                                          'Hello World',
                                        ),
                                      ],
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            Expanded(
                              child: Padding(
                                padding:
                                    EdgeInsetsDirectional.fromSTEB(20, 0, 0, 0),
                                child: Row(
                                  mainAxisSize: MainAxisSize.max,
                                  children: [
                                    Expanded(
                                      child: Text(
                                          'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc maximus, nulla ut commodo sagittis, sapien dui mattis dui, non pulvinar lorem felis nec erat Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc maximus, nulla ut commodo sagittis, sapien dui mattis dui, non pulvinar lorem felis nec erat Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc maximus, nulla ut commodo sagittis, sapien dui mattis dui, non pulvinar lorem felis nec erat Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc maximus, nulla ut commodo sagittis, sapien dui mattis dui, non pulvinar lorem felis nec erat ',
                                          textAlign: TextAlign.justify,
                                          style: TextStyle(
                                              fontFamily: 'Outfit',
                                              fontWeight: FontWeight.w600)),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                      Padding(
                        padding: EdgeInsetsDirectional.fromSTEB(0, 80, 0, 80),
                        child: Row(
                          mainAxisSize: MainAxisSize.max,
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text('COLLECT, PROCESS, AND DISPLAY.',
                                style: TextStyle(
                                    fontFamily: 'Outfit',
                                    fontSize: 28,
                                    fontWeight: FontWeight.w800)),
                          ],
                        ),
                      ),
                      Padding(
                        padding: EdgeInsetsDirectional.fromSTEB(0, 0, 0, 20),
                        child: Row(
                          mainAxisSize: MainAxisSize.max,
                          children: [
                            Expanded(
                              child: Padding(
                                padding:
                                    EdgeInsetsDirectional.fromSTEB(0, 10, 0, 0),
                                child: Text('THE TEAM',
                                    style: TextStyle(
                                        fontFamily: 'Outfit',
                                        fontSize: 55,
                                        fontWeight: FontWeight.w800)),
                              ),
                            ),
                          ],
                        ),
                      ),
                      Row(
                        mainAxisSize: MainAxisSize.max,
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Expanded(
                            child: Column(
                              mainAxisSize: MainAxisSize.max,
                              children: [
                                Padding(
                                  padding: EdgeInsetsDirectional.fromSTEB(
                                      0, 0, 10, 10),
                                  child: Container(
                                    width: 250,
                                    height: 250,
                                    clipBehavior: Clip.antiAlias,
                                    decoration: BoxDecoration(
                                      shape: BoxShape.circle,
                                    ),
                                    child: Image.asset(
                                      'assets/akeel.jpeg',
                                      fit: BoxFit.cover,
                                    ),
                                  ),
                                ),
                                Padding(
                                  padding: EdgeInsetsDirectional.fromSTEB(
                                      0, 10, 0, 0),
                                  child: Text('Akeel Ather Medina',
                                      style: TextStyle(
                                          fontFamily: 'Outfit',
                                          fontWeight: FontWeight.w600)),
                                ),
                                Text(
                                  'Bsc. in Computer Science',
                                ),
                                Text(
                                  'Habib University',
                                ),
                                Text(
                                  'am05427@st.habib.edu.pk',
                                ),
                              ],
                            ),
                          ),
                          Expanded(
                            child: Column(
                              mainAxisSize: MainAxisSize.max,
                              children: [
                                Padding(
                                  padding: EdgeInsetsDirectional.fromSTEB(
                                      0, 0, 10, 10),
                                  child: Container(
                                    width: 250,
                                    height: 250,
                                    clipBehavior: Clip.antiAlias,
                                    decoration: BoxDecoration(
                                      shape: BoxShape.circle,
                                    ),
                                    child: Image.asset(
                                      'assets/abeer.jpeg',
                                      fit: BoxFit.cover,
                                    ),
                                  ),
                                ),
                                Padding(
                                  padding: EdgeInsetsDirectional.fromSTEB(
                                      0, 10, 0, 0),
                                  child: Text('Abeer Khan',
                                      style: TextStyle(
                                          fontFamily: 'Outfit',
                                          fontWeight: FontWeight.w600)),
                                ),
                                Text(
                                  'Bsc. in Computer Science',
                                ),
                                Text(
                                  'Habib University',
                                ),
                                Text(
                                  'ak05419@st.habib.edu.pk',
                                ),
                              ],
                            ),
                          ),
                          Expanded(
                            child: Column(
                              mainAxisSize: MainAxisSize.max,
                              children: [
                                Padding(
                                  padding: EdgeInsetsDirectional.fromSTEB(
                                      0, 0, 10, 10),
                                  child: Container(
                                    width: 250,
                                    height: 250,
                                    clipBehavior: Clip.antiAlias,
                                    decoration: BoxDecoration(
                                      shape: BoxShape.circle,
                                    ),
                                    child: Image.asset(
                                      'assets/zoha.jpeg',
                                      fit: BoxFit.cover,
                                    ),
                                  ),
                                ),
                                Padding(
                                  padding: EdgeInsetsDirectional.fromSTEB(
                                      0, 10, 0, 0),
                                  child: Text('Zoha Ovais Karim',
                                      style: TextStyle(
                                          fontFamily: 'Outfit',
                                          fontWeight: FontWeight.w600)),
                                ),
                                Text(
                                  'Bsc. in Computer Science',
                                ),
                                Text(
                                  'Habib University',
                                ),
                                Text(
                                  'zk05617@st.habib.edu.pk',
                                ),
                              ],
                            ),
                          ),
                          Expanded(
                            child: Column(
                              mainAxisSize: MainAxisSize.max,
                              children: [
                                Padding(
                                  padding: EdgeInsetsDirectional.fromSTEB(
                                      0, 0, 0, 10),
                                  child: Container(
                                    width: 250,
                                    height: 250,
                                    clipBehavior: Clip.antiAlias,
                                    decoration: BoxDecoration(
                                      shape: BoxShape.circle,
                                    ),
                                    child: Image.asset(
                                      'assets/samra.jpeg',
                                      fit: BoxFit.cover,
                                    ),
                                  ),
                                ),
                                Padding(
                                  padding: EdgeInsetsDirectional.fromSTEB(
                                      0, 10, 0, 0),
                                  child: Text('Samarah Asghar Sahto',
                                      style: TextStyle(
                                          fontFamily: 'Outfit',
                                          fontWeight: FontWeight.w600)),
                                ),
                                Text(
                                  'Bsc. in Computer Science',
                                ),
                                Text(
                                  'Habib University',
                                ),
                                Text(
                                  'ss05563@st.habib.edu.pk',
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ),
            Padding(
              padding: EdgeInsetsDirectional.fromSTEB(0, 12, 0, 44),
              child: ListView(
                padding: EdgeInsets.zero,
                primary: false,
                shrinkWrap: true,
                scrollDirection: Axis.vertical,
                children: [
                  Padding(
                    padding: EdgeInsetsDirectional.fromSTEB(16, 0, 16, 0),
                    child: Container(
                      width: double.infinity,
                      decoration: BoxDecoration(),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
      //   body:    Padding(
      //   padding: const EdgeInsets.all(90),
      //   child: Column(
      //     crossAxisAlignment: CrossAxisAlignment.start,
      //     children: [
      //       // para 1 title
      //       const Padding(
      //         padding: EdgeInsets.only(bottom: 30),
      //         child: Text(
      //           'What is Raasta?',
      //           style: TextStyle(
      //             fontWeight: FontWeight.bold,
      //             fontSize: 28,
      //           ),
      //         ),
      //       ),
      //       // para 1
      //       const Padding(
      //         padding: EdgeInsets.only(bottom: 40),
      //         child: Text('Our Approach'),
      //       ),
      //       // section 2
      //       Padding(
      //         padding: const EdgeInsets.only(bottom: 40),
      //         child: Row(
      //           crossAxisAlignment: CrossAxisAlignment.start,
      //           children: [
      //             Expanded(
      //               flex: 3,
      //               child: Padding(
      //                 padding: const EdgeInsets.only(right: 40),
      //                 child: Column(
      //                   crossAxisAlignment: CrossAxisAlignment.start,
      //                   children: const [
      //                     // para 2 title
      //                     Padding(
      //                       padding: EdgeInsets.only(bottom: 30),
      //                       child: Text(
      //                         'text',
      //                         style: TextStyle(
      //                           fontWeight: FontWeight.bold,
      //                           fontSize: 28,
      //                         ),
      //                       ),
      //                     ),
      //                     // para 2
      //                     Padding(
      //                       padding: EdgeInsets.only(bottom: 40),
      //                       child: Text('text'),
      //                     ),
      //                   ],
      //                 ),
      //               ),
      //             ),
      //             const Expanded(
      //               flex: 1,
      //               child: Padding(
      //                 padding: EdgeInsets.only(right: 20),
      //                 child: Placeholder(fallbackHeight: 200),
      //               ),
      //             ),
      //             const Expanded(
      //                 flex: 1, child: Placeholder(fallbackHeight: 200)),
      //           ],
      //         ),
      //       ),
      //       // section 3
      //       const Padding(
      //         padding: EdgeInsets.only(bottom: 30),
      //         child: Text(
      //           "Data Availability",
      //           style: TextStyle(
      //             fontWeight: FontWeight.bold,
      //             fontSize: 28,
      //           ),
      //         ),
      //       ),
      //       Padding(
      //         padding: const EdgeInsets.only(bottom: 40),
      //         child: Row(
      //           children: [
      //             // grey box
      //             Container(
      //               decoration: BoxDecoration(
      //                 color: Colors.grey.shade300,
      //                 border: Border.all(width: 2),
      //               ),
      //               margin: const EdgeInsets.only(right: 30),
      //               padding: const EdgeInsets.all(60),
      //               child: Row(
      //                 children: [
      //                   Padding(
      //                     padding: const EdgeInsets.only(right: 35),
      //                     child: Column(
      //                       children: const [
      //                         Padding(
      //                           padding: EdgeInsets.only(bottom: 10),
      //                           child: Text(
      //                             '100000',
      //                             style: TextStyle(
      //                               fontWeight: FontWeight.bold,
      //                               fontSize: 28,
      //                             ),
      //                           ),
      //                         ),
      //                         // para 1
      //                         Text('Samples trained on'),
      //                       ],
      //                     ),
      //                   ),
      //                   Column(
      //                     children: const [
      //                       Padding(
      //                         padding: EdgeInsets.only(bottom: 10),
      //                         child: Text(
      //                           '100000',
      //                           style: TextStyle(
      //                             fontWeight: FontWeight.bold,
      //                             fontSize: 28,
      //                           ),
      //                         ),
      //                       ),
      //                       // para 1
      //                       Text('Accuracies acquired'),
      //                     ],
      //                   ),
      //                 ],
      //               ),
      //             ),
      //             const Text('Lorem ipsum'),
      //           ],
      //         ),
      //       ),
      //       // section 4
      //       const Padding(
      //         padding: EdgeInsets.only(bottom: 30),
      //         child: Text(
      //           "Our Team",
      //           style: TextStyle(
      //             fontWeight: FontWeight.bold,
      //             fontSize: 28,
      //           ),
      //         ),
      //       ),

      //       Row(
      //         mainAxisAlignment: MainAxisAlignment.spaceBetween,
      //         children: List.generate(
      //           1,

      //           (index) => Column(
      //             children: [
      //               Container(
      //                 decoration: BoxDecoration(
      //                   shape: BoxShape.circle,
      //                   border: Border.all(width: 3),
      //                 ),
      //                 height: 200,
      //                 width: 200,
      //                 margin: const EdgeInsets.only(bottom: 30),
      //               ),
      //               const Text('Abeer Khan'),
      //               const Text('Habib University, DSSE'),
      //               const Text('Contact: email address'),
      //               Container(
      //                 decoration: BoxDecoration(
      //                   shape: BoxShape.circle,
      //                   border: Border.all(width: 3),
      //                 ),
      //                 height: 200,
      //                 width: 200,
      //                 margin: const EdgeInsets.only(bottom: 30),
      //               ),
      //               const Text('Akeel Ather Medina'),
      //               const Text('Habib University, DSSE'),
      //               const Text('Contact: email address'),
      //               Container(
      //                 decoration: BoxDecoration(
      //                   image: DecorationImage(
      //                       image: AssetImage('samra.jpeg'), fit: BoxFit.fill),
      //                   shape: BoxShape.circle,
      //                   border: Border.all(width: 3),
      //                 ),
      //                 height: 200,
      //                 width: 200,
      //                 margin: const EdgeInsets.only(bottom: 30),
      //               ),
      //               const Text('Samarah Asghar Sahto'),
      //               const Text('Habib University, DSSE'),
      //               const Text('Contact: email address'),
      //               Container(
      //                 decoration: BoxDecoration(
      //                   image: DecorationImage(
      //                       image: AssetImage('zoha.jpeg'), fit: BoxFit.fill),
      //                   shape: BoxShape.circle,
      //                   border: Border.all(width: 3),
      //                 ),
      //                 height: 200,
      //                 width: 200,
      //                 margin: const EdgeInsets.only(bottom: 30),
      //               ),
      //               const Text('Zoha Ovais Karim'),
      //               const Text('Habib University, DSSE'),
      //               const Text('Contact: email address'),
      //             ],
      //           ),
      //           // Row(
      //           //   mainAxisAlignment: MainAxisAlignment.spaceBetween,
      //           //   children: List.generate(
      //           //     4,
      //           //     (index) => Column(
      //           //       children: [
      //           //         Container(
      //           //           decoration: BoxDecoration(
      //           //             shape: BoxShape.circle,
      //           //             border: Border.all(width: 3),
      //           //           ),
      //           //           height: 200,
      //           //           width: 200,
      //           //           margin: const EdgeInsets.only(bottom: 30),
      //           //         ),
      //           //         const Text('Abeer Khan'),
      //           //         const Text('Habib University, DSSE'),
      //           //         const Text('Contact: email address'),

      //           //       ],
      //           //     ),
      //           //   ),
      //           // ),
      //         ),
      //       ),
      //     ],
      //   ),
      // )
    );
  }
}















// import 'package:flutter/material.dart';

// class About extends StatefulWidget {
//   const About({Key? key}) : super(key: key);

//   @override
//   State<About> createState() => _AboutState();
// }

// // inspiration from
// // https://smartroadsense.it/project/about/
// class _AboutState extends State<About> {
//   @override
//   Widget build(BuildContext context) => Column(
//         children: [
//           // "about"
//           Stack(
//             children: [
//               Image.asset(
//                 'assets/about_pic.jpg',
//                 width: double.infinity,
//                 height: 250,
//                 fit: BoxFit.cover,
//               ),
//               Align(
//                 alignment: Alignment.topLeft,
//                 child: Container(
//                   width: 200,
//                   //transform: Transform.rotate(angle: 3.14 / (-20)).transform,
//                   margin: const EdgeInsets.fromLTRB(0, 0, 100, 0),
//                   padding: const EdgeInsets.all(20),
//                   decoration: BoxDecoration(
//                     color: Colors.blue,
//                     gradient: LinearGradient(
//                       colors: [
//                         Colors.blue.shade400,
//                         Colors.blue.shade100,
//                       ],
//                     ),
//                   ),
//                   child: const FittedBox(
//                     fit: BoxFit.fill,
//                     child: Text(
//                       "About",
//                       textAlign: TextAlign.start,
//                       style: TextStyle(color: Colors.white),
//                     ),
//                   ),
//                 ),
//               ),
//             ],
//           ),
//           const Padding(
//             padding: EdgeInsets.all(30),
//             child: Text(
//               'Raasta is a project built by students of Habib University in 2023. This project is to highlight the safety of the vehicles and the comfort of the drivers, and in order to do so it is important to monitor the condition of the road surfaces and any anomalies. The potholes displayed on the map are identified by a road surface classification system that uses a tri-axial accelerometer and gyroscope along with a global positioning system (GPS) receiver. A deep learning method that uses a specially designed android application was deployed that automatically recognises potholes and distinguishes them from other abnormalities of the road surface, such as speedbreakers.',
//               textAlign: TextAlign.center,
//             ),
//           ),
//         ],
//       );
// }
