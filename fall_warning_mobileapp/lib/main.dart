import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart';

import 'services/notification_service.dart';
import 'screens/events_page.dart';

/// Background FCM handler (bắt buộc cho Android & iOS)
@pragma('vm:entry-point')
Future<void> firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  await Firebase.initializeApp();
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Init Firebase
  await Firebase.initializeApp();

  // Register background handler
  FirebaseMessaging.onBackgroundMessage(
    firebaseMessagingBackgroundHandler,
  );

  runApp(const FallWarningApp());
}

class FallWarningApp extends StatelessWidget {
  const FallWarningApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fall Warning',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.redAccent),
        useMaterial3: true,
      ),
      home: const AppBootstrap(),
    );
  }
}

/// Widget bootstrap để setup FCM SAU KHI UI render
class AppBootstrap extends StatefulWidget {
  const AppBootstrap({super.key});

  @override
  State<AppBootstrap> createState() => _AppBootstrapState();
}

class _AppBootstrapState extends State<AppBootstrap> {
  @override
  void initState() {
    super.initState();
    _initFCM();
  }

  Future<void> _initFCM() async {
    // Init notification system
    await NotificationService.init();

    // Listen foreground messages
    NotificationService.listenForegroundMessages();

    // Chờ 1 chút cho iOS cấp APNS token
    await Future.delayed(const Duration(seconds: 2));

    // Subscribe topic (đã xử lý APNS token bên trong)
    await NotificationService.subscribeFallAlertsTopic();
  }

  @override
  Widget build(BuildContext context) {
    return const EventsPage();
  }
}
