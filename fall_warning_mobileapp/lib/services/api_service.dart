import 'dart:convert';
import 'package:http/http.dart' as http;
import '../config/app_config.dart';

class EventItem {
  final int id;
  final DateTime? eventTime;
  final String eventType;
  final double? confidence;
  final String? imageUrl;

  EventItem({
    required this.id,
    required this.eventTime,
    required this.eventType,
    required this.confidence,
    required this.imageUrl,
  });

  factory EventItem.fromJson(Map<String, dynamic> json) {
    return EventItem(
      id: json['id'] as int,
      eventTime: json['event_time'] != null
          ? DateTime.tryParse(json['event_time'] as String)
          : null,
      eventType: json['event_type'] as String? ?? 'fall',
      confidence: json['confidence'] != null
          ? (json['confidence'] as num).toDouble()
          : null,
      imageUrl: json['image_url'] as String?,
    );
  }
}

class ApiService {
  final http.Client _client;
  ApiService({http.Client? client}) : _client = client ?? http.Client();

  Future<List<EventItem>> fetchEvents({int limit = 50}) async {
    final baseUrl = AppConfig.baseUrl;
    final uri = Uri.parse('$baseUrl/api/events?limit=$limit');
    final res = await _client.get(uri);
    if (res.statusCode != 200) {
      throw Exception('Fetch events failed: ${res.statusCode}');
    }
    final data = json.decode(res.body) as Map<String, dynamic>;
    final items = (data['items'] as List).cast<Map<String, dynamic>>();
    return items.map((e) {
      final rel = e['image_url'] as String?;
      if (rel != null && rel.startsWith('/')) {
        e = Map<String, dynamic>.from(e);
        e['image_url'] = '$baseUrl$rel';
      }
      return EventItem.fromJson(e);
    }).toList();
  }
}
