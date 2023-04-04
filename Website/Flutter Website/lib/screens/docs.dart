import 'package:flutter/material.dart';

class Docs extends StatefulWidget {
  const Docs({Key? key}) : super(key: key);

  @override
  State<Docs> createState() => _DocsState();
}

class _DocsState extends State<Docs> {
  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Text('Docs'),
    );
  }
}
