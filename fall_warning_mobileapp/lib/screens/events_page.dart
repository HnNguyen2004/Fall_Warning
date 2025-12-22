import 'package:flutter/material.dart';
import '../services/api_service.dart';

class EventsPage extends StatefulWidget {
  const EventsPage({super.key});

  @override
  State<EventsPage> createState() => _EventsPageState();
}

class _EventsPageState extends State<EventsPage> {
  final _api = ApiService();
  late Future<List<EventItem>> _future;

  @override
  void initState() {
    super.initState();
    _future = _api.fetchEvents(limit: 50);
  }

  Future<void> _refresh() async {
    final data = await _api.fetchEvents(limit: 50);
    if (mounted) {
      setState(() {
        _future = Future.value(data);
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Fall Events'),
      ),
      body: RefreshIndicator(
        onRefresh: _refresh,
        child: FutureBuilder<List<EventItem>>(
          future: _future,
          builder: (context, snapshot) {
            if (snapshot.connectionState == ConnectionState.waiting) {
              return const Center(child: CircularProgressIndicator());
            }
            if (snapshot.hasError) {
              return Center(child: Text('Lỗi tải dữ liệu: ${snapshot.error}'));
            }
            final items = snapshot.data ?? [];
            if (items.isEmpty) {
              return const Center(child: Text('Chưa có sự kiện. Kéo xuống để cập nhật.'));
            }
            return ListView.separated(
              physics: const AlwaysScrollableScrollPhysics(),
              itemCount: items.length,
              separatorBuilder: (_, __) => const Divider(height: 1),
              itemBuilder: (context, index) {
                final e = items[index];
                return ListTile(
                  leading: e.imageUrl != null
                      ? ClipRRect(
                          borderRadius: BorderRadius.circular(6),
                          child: Image.network(
                            e.imageUrl!,
                            width: 56,
                            height: 56,
                            fit: BoxFit.cover,
                            errorBuilder: (_, __, ___) => const Icon(Icons.image_not_supported),
                          ),
                        )
                      : const Icon(Icons.image_not_supported),
                  title: Text('Fall • conf: ${e.confidence?.toStringAsFixed(2) ?? '-'}'),
                  subtitle: Text(e.eventTime?.toLocal().toString() ?? ''),
                );
              },
            );
          },
        ),
      ),
    );
  }
}
