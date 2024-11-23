import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import 'config.dart';  // Import file config.dart

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Base64 Image Viewer',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  String feedUrl = "https://io.adafruit.com/api/v2/${Config.username}/feeds?x-aio-key=${Config.feedKey}"; // Ganti dengan URL yang sesuai
  Image? image;

  @override
  void initState() {
    super.initState();
    fetchImage();
  }

  Future<void> fetchImage() async {
    final response = await http.get(Uri.parse(feedUrl));

    if (response.statusCode == 200) {
      // Parsing response JSON
      List<dynamic> feeds = json.decode(response.body);

      // Mencari feed dengan id tertentu
      var targetFeed = feeds.firstWhere(
        (feed) => feed['id'] == 2940873, // Ganti dengan ID feed yang diinginkan
        orElse: () => null,
      );

      if (targetFeed != null && targetFeed['last_value'] != null) {
        // Mendapatkan last_value yang berisi gambar dalam Base64
        String base64Image = targetFeed['last_value'];

        // Decode Base64 menjadi bytes
        Uint8List bytes = base64Decode(base64Image);

        // Update state untuk menampilkan gambar
        setState(() {
          image = Image.memory(bytes);
        });
      } else {
        print("Feed tidak ditemukan atau last_value kosong");
      }
    } else {
      print('Failed to load feed data: ${response.statusCode}');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Base64 Image Viewer'),
      ),
      body: Center(
        child: image == null
            ? CircularProgressIndicator()
            : image!,
      ),
    );
  }
}
