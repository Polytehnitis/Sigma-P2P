import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:app/voice/endpointer.dart';
import 'package:app/voice/speech_input.dart';
import 'package:app/voice/speech_model.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:record/record.dart';

const rate = 16000;

/// `ms` of a 220 Hz tone at `level` (0 is silence), as floats.
Float32List tone(int ms, double level) {
  final n = rate * ms ~/ 1000;
  return Float32List.fromList([
    for (var i = 0; i < n; i++) level * math.sin(2 * math.pi * 220 * i / rate),
  ]);
}

/// The same as little-endian 16-bit PCM, as the microphone sends it.
Uint8List pcm(Float32List samples) {
  final out = ByteData(samples.length * 2);
  for (var i = 0; i < samples.length; i++) {
    out.setInt16(i * 2, (samples[i] * 32767).round(), Endian.little);
  }
  return out.buffer.asUint8List();
}

/// Feeds `audio` in 100 ms chunks, as a microphone would; returns the first non-listening state.
Endpoint feed(Endpointer e, Float32List audio) {
  for (var at = 0; at < audio.length; at += rate ~/ 10) {
    final s = e.add(
      Float32List.sublistView(
        audio,
        at,
        math.min(at + rate ~/ 10, audio.length),
      ),
    );
    if (s != Endpoint.listening) return s;
  }
  return Endpoint.listening;
}

/// A microphone that plays back chunks of PCM.
class FakeRecorder implements AudioRecorder {
  FakeRecorder(this.chunks);
  final List<Uint8List> chunks;
  final _ctrl = StreamController<Uint8List>();
  bool recording = false;
  bool disposed = false;

  @override
  Future<bool> hasPermission({bool request = true}) async => true;

  @override
  Future<Stream<Uint8List>> startStream(RecordConfig config) async {
    recording = true;
    () async {
      for (final c in chunks) {
        await Future<void>.delayed(Duration.zero);
        if (!recording) return;
        _ctrl.add(c);
      }
    }();
    return _ctrl.stream;
  }

  @override
  Future<bool> isRecording() async => recording;

  @override
  Future<String?> stop() async {
    recording = false;
    return null;
  }

  @override
  Future<void> dispose() async => disposed = true;

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

Future<SpeechModelStore> installedModel() async {
  final dir = await Directory.systemTemp.createTemp('sigma-speech');
  final store = SpeechModelStore(dir, spec: SpeechModelSpec.desktop);
  store.file.writeAsStringSync('model');
  return store;
}

List<Uint8List> chunked(Float32List audio) => [
  for (var at = 0; at < audio.length; at += rate ~/ 10)
    pcm(
      Float32List.sublistView(
        audio,
        at,
        math.min(at + rate ~/ 10, audio.length),
      ),
    ),
];

Float32List concat(List<Float32List> parts) =>
    Float32List.fromList([for (final p in parts) ...p]);

void main() {
  group('Endpointer', () {
    test('speech then silence ends the utterance', () {
      final e = Endpointer();
      final audio = concat([tone(500, 0), tone(700, 0.2), tone(1000, 0)]);
      expect(feed(e, audio), Endpoint.spoken);
      expect(e.heardSpeech, isTrue);
    });

    test('a short pause inside a sentence does not end it', () {
      final e = Endpointer();
      final audio = concat([tone(600, 0.2), tone(400, 0), tone(600, 0.2)]);
      expect(feed(e, audio), Endpoint.listening);
    });

    test('silence is nothing, after the timeout', () {
      final e = Endpointer(noSpeechTimeout: const Duration(seconds: 2));
      expect(feed(e, tone(1500, 0)), Endpoint.listening);
      expect(feed(e, tone(1000, 0)), Endpoint.nothing);
      expect(e.heardSpeech, isFalse);
    });

    test('a click is not speech', () {
      final e = Endpointer(noSpeechTimeout: const Duration(seconds: 1));
      expect(feed(e, concat([tone(60, 0.5), tone(1500, 0)])), Endpoint.nothing);
    });

    test('steady room noise is learned, and speech above it still counts', () {
      final e = Endpointer();
      final hum = tone(1500, 0.02);
      expect(feed(e, hum), Endpoint.listening);
      expect(e.heardSpeech, isFalse);
      final audio = concat([tone(700, 0.3), tone(1000, 0.02)]);
      expect(feed(e, audio), Endpoint.spoken);
    });

    test('talking on is cut at the limit', () {
      final e = Endpointer(maxUtterance: const Duration(seconds: 2));
      expect(feed(e, tone(3000, 0.2)), Endpoint.tooLong);
    });
  });

  group('WhisperSpeechInput', () {
    test(
      'records until silence, then transcribes the whole utterance',
      () async {
        final audio = concat([tone(300, 0), tone(800, 0.2), tone(1000, 0)]);
        final recorder = FakeRecorder(chunked(audio));
        String? loaded;
        Float32List? sent;
        final input = WhisperSpeechInput(
          await installedModel(),
          recorder: recorder,
          load: (path) async => loaded = path,
          transcribe: (samples) async {
            sent = samples;
            return (text: ' Stop. ', confidence: 0.92);
          },
        );
        final heard = await input.listen().toList();
        expect(loaded, endsWith('ggml-small.en-q5_1.bin'));
        expect(heard.first.recognizing, isTrue);
        expect(heard.last.isFinal, isTrue);
        expect(heard.last.text, 'Stop.');
        expect(heard.last.confidence, 0.92);
        // Everything up to the end of the trailing silence was sent, nothing more.
        expect(sent!.length, greaterThan(rate * 1.8));
        expect(sent!.length, lessThan(audio.length));
        expect(recorder.recording, isFalse);
      },
    );

    test('nobody speaking never reaches whisper', () async {
      var calls = 0;
      final input = WhisperSpeechInput(
        await installedModel(),
        recorder: FakeRecorder(chunked(tone(7000, 0))),
        load: (_) async {},
        transcribe: (_) async {
          calls++;
          return (text: 'Thank you.', confidence: 0.5);
        },
      );
      final heard = await input.listen().toList();
      expect(heard.single.isFinal, isTrue);
      expect(heard.single.text, '');
      expect(calls, 0);
    });

    test('stop while speaking transcribes what was said', () async {
      final input = WhisperSpeechInput(
        await installedModel(),
        recorder: FakeRecorder(chunked(tone(3000, 0.2))),
        load: (_) async {},
        transcribe: (_) async => (text: 'undo that', confidence: 0.8),
      );
      final heard = <Hearing>[];
      final done = input.listen().forEach(heard.add);
      await Future<void>.delayed(const Duration(milliseconds: 50));
      await input.stop();
      await done;
      expect(heard.last.text, 'undo that');
    });

    test('a missing model is an error, not silence', () async {
      final dir = await Directory.systemTemp.createTemp('sigma-speech');
      final input = WhisperSpeechInput(
        SpeechModelStore(dir, spec: SpeechModelSpec.desktop),
        recorder: FakeRecorder(const []),
        load: (_) async {},
        transcribe: (_) async => (text: '', confidence: 0.0),
      );
      expect(input.listen().toList(), throwsA(isA<SpeechModelException>()));
    });
  });
}
