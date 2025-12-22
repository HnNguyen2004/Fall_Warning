import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'services/notification_service.dart';
import 'screens/events_page.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Nếu đã có firebase_options.dart sau khi chạy flutterfire configure,
  // bạn có thể thay bằng: await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);
  await Firebase.initializeApp();

  await NotificationService.init();
  NotificationService.listenForegroundMessages();
  await NotificationService.subscribeFallAlertsTopic();

  runApp(const FallWarningApp());
}

class FallWarningApp extends StatelessWidget {
  const FallWarningApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fall Warning',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.redAccent),
        useMaterial3: true,
      ),
      home: const EventsPage(),
    );
  }
}