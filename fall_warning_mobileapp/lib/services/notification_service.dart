import 'dart:io';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

final FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin =
    FlutterLocalNotificationsPlugin();

@pragma('vm:entry-point')
Future<void> firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  await Firebase.initializeApp();
}

class NotificationService {
  /// INIT – chỉ làm setup, KHÔNG subscribe topic
  static Future<void> init() async {
    const androidInit =
        AndroidInitializationSettings('@mipmap/ic_launcher');
    const iosInit = DarwinInitializationSettings();

    const initSettings = InitializationSettings(
      android: androidInit,
      iOS: iosInit,
    );

    await flutterLocalNotificationsPlugin.initialize(initSettings);

    // Xin quyền iOS
    await FirebaseMessaging.instance.requestPermission(
      alert: true,
      badge: true,
      sound: true,
    );

    // iOS foreground notification
    await FirebaseMessaging.instance
        .setForegroundNotificationPresentationOptions(
      alert: true,
      badge: true,
      sound: true,
    );
  }

  /// Foreground listener
  static void listenForegroundMessages() {
    FirebaseMessaging.onMessage.listen((RemoteMessage message) {
      final notification = message.notification;
      showLocalNotification(
        notification?.title ?? 'Fall Warning',
        notification?.body ?? 'Có cảnh báo té ngã mới',
      );
    });
  }

  /// ✅ SUBSCRIBE CHUẨN – chờ APNS token
  static Future<void> subscribeFallAlertsTopic() async {
    if (Platform.isIOS) {
      String? apnsToken;

      // Chờ tối đa ~10s cho APNS token
      for (int i = 0; i < 5; i++) {
        apnsToken = await FirebaseMessaging.instance.getAPNSToken();
        if (apnsToken != null) break;
        await Future.delayed(const Duration(seconds: 2));
      }

      if (apnsToken == null) {
        print('❌ APNS token chưa sẵn sàng – skip subscribe');
        return;
      }
    }

    await FirebaseMessaging.instance.subscribeToTopic('fall_alerts');
    print('✅ Subscribed to fall_alerts');
  }

  static Future<void> showLocalNotification(
      String title, String body) async {
    const androidDetails = AndroidNotificationDetails(
      'fall_alerts_channel',
      'Fall Alerts',
      channelDescription: 'Thông báo té ngã',
      importance: Importance.max,
      priority: Priority.high,
    );

    const iosDetails = DarwinNotificationDetails();

    const details = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    await flutterLocalNotificationsPlugin.show(
      DateTime.now().millisecondsSinceEpoch ~/ 1000,
      title,
      body,
      details,
    );
  }
}
