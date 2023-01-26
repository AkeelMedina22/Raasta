import 'package:flutter/material.dart';
import '../../widgets/my_appbar.dart';

class DocsScreen extends StatefulWidget {
  const DocsScreen({Key? key}) : super(key: key);

  @override
  State<DocsScreen> createState() => _DocsScreenState();
}

class _DocsScreenState extends State<DocsScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: myAppBar(context),
      body: const Center(
        child: Text('Docs'),
      ),
    );
  }
}
