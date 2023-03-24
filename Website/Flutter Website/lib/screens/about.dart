import 'package:flutter/material.dart';

class About extends StatefulWidget {
  const About({Key? key}) : super(key: key);

  @override
  State<About> createState() => _AboutState();
}

// inspiration from
// https://smartroadsense.it/project/about/
class _AboutState extends State<About> {
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.all(90),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // para 1 title
            const Padding(
              padding: EdgeInsets.only(bottom: 30),
              child: Text(
                'What is Raasta?',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 28,
                ),
              ),
            ),
            // para 1
            const Padding(
              padding: EdgeInsets.only(bottom: 40),
              child: Text('Our Approach'),
            ),
            // section 2
            Padding(
              padding: const EdgeInsets.only(bottom: 40),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    flex: 3,
                    child: Padding(
                      padding: const EdgeInsets.only(right: 40),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: const [
                          // para 2 title
                          Padding(
                            padding: EdgeInsets.only(bottom: 30),
                            child: Text(
                              'text',
                              style: TextStyle(
                                fontWeight: FontWeight.bold,
                                fontSize: 28,
                              ),
                            ),
                          ),
                          // para 2
                          Padding(
                            padding: EdgeInsets.only(bottom: 40),
                            child: Text('text'),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const Expanded(
                    flex: 1,
                    child: Padding(
                      padding: EdgeInsets.only(right: 20),
                      child: Placeholder(fallbackHeight: 200),
                    ),
                  ),
                  const Expanded(
                      flex: 1, child: Placeholder(fallbackHeight: 200)),
                ],
              ),
            ),
            // section 3
            const Padding(
              padding: EdgeInsets.only(bottom: 30),
              child: Text(
                "Data Availability",
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 28,
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(bottom: 40),
              child: Row(
                children: [
                  // grey box
                  Container(
                    decoration: BoxDecoration(
                      color: Colors.grey.shade300,
                      border: Border.all(width: 2),
                    ),
                    margin: const EdgeInsets.only(right: 30),
                    padding: const EdgeInsets.all(60),
                    child: Row(
                      children: [
                        Padding(
                          padding: const EdgeInsets.only(right: 35),
                          child: Column(
                            children: const [
                              Padding(
                                padding: EdgeInsets.only(bottom: 10),
                                child: Text(
                                  '100000',
                                  style: TextStyle(
                                    fontWeight: FontWeight.bold,
                                    fontSize: 28,
                                  ),
                                ),
                              ),
                              // para 1
                              Text('Samples trained on'),
                            ],
                          ),
                        ),
                        Column(
                          children: const [
                            Padding(
                              padding: EdgeInsets.only(bottom: 10),
                              child: Text(
                                '100000',
                                style: TextStyle(
                                  fontWeight: FontWeight.bold,
                                  fontSize: 28,
                                ),
                              ),
                            ),
                            // para 1
                            Text('Accuracies acquired'),
                          ],
                        ),
                      ],
                    ),
                  ),
                  const Text('Lorem ipsum'),
                ],
              ),
            ),
            // section 4
            const Padding(
              padding: EdgeInsets.only(bottom: 30),
              child: Text(
                "Our Team",
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 28,
                ),
              ),
            ),

            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: List.generate(
                1,

                (index) => Column(
                  children: [
                    Container(
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(width: 3),
                      ),
                      height: 200,
                      width: 200,
                      margin: const EdgeInsets.only(bottom: 30),
                    ),
                    const Text('Abeer Khan'),
                    const Text('Habib University, DSSE'),
                    const Text('Contact: email address'),
                    Container(
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(width: 3),
                      ),
                      height: 200,
                      width: 200,
                      margin: const EdgeInsets.only(bottom: 30),
                    ),
                    const Text('Akeel Ather Medina'),
                    const Text('Habib University, DSSE'),
                    const Text('Contact: email address'),
                    Container(
                      decoration: BoxDecoration(
                        image: DecorationImage(
                            image: AssetImage('samra.jpeg'), fit: BoxFit.fill),
                        shape: BoxShape.circle,
                        border: Border.all(width: 3),
                      ),
                      height: 200,
                      width: 200,
                      margin: const EdgeInsets.only(bottom: 30),
                    ),
                    const Text('Samarah Asghar Sahto'),
                    const Text('Habib University, DSSE'),
                    const Text('Contact: email address'),
                    Container(
                      decoration: BoxDecoration(
                        image: DecorationImage(
                            image: AssetImage('zoha.jpeg'), fit: BoxFit.fill),
                        shape: BoxShape.circle,
                        border: Border.all(width: 3),
                      ),
                      height: 200,
                      width: 200,
                      margin: const EdgeInsets.only(bottom: 30),
                    ),
                    const Text('Zoha Ovais Karim'),
                    const Text('Habib University, DSSE'),
                    const Text('Contact: email address'),
                  ],
                ),
                // Row(
                //   mainAxisAlignment: MainAxisAlignment.spaceBetween,
                //   children: List.generate(
                //     4,
                //     (index) => Column(
                //       children: [
                //         Container(
                //           decoration: BoxDecoration(
                //             shape: BoxShape.circle,
                //             border: Border.all(width: 3),
                //           ),
                //           height: 200,
                //           width: 200,
                //           margin: const EdgeInsets.only(bottom: 30),
                //         ),
                //         const Text('Abeer Khan'),
                //         const Text('Habib University, DSSE'),
                //         const Text('Contact: email address'),

                //       ],
                //     ),
                //   ),
                // ),
              ),
            ),
          ],
        ),
      );
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
