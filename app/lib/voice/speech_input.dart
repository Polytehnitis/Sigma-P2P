import 'dart:async';
import 'dart:typed_data';

import 'package:record/record.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa;

import 'speech_model.dart';

/// What the recognizer heard so far. `isFinal` marks the end of an utterance.
class Hearing {
  const Hearing(this.text, {this.isFinal = false});
  final String text;
  final bool isFinal;
}

/// A source of spoken utterances. One `listen` call hears one utterance.
abstract class SpeechInput {
  /// Partial hearings, then one final hearing (empty text if nothing was understood).
  Stream<Hearing> listen();

  /// Ends the current utterance early; what was heard so far becomes final.
  Future<void> stop();

  void dispose();
}

/// Streaming on-device recognition: microphone PCM (16 kHz mono) into a sherpa-onnx
/// zipformer transducer. No audio or text leaves the device.
class SherpaSpeechInput implements SpeechInput {
  SherpaSpeechInput(this.model);

  final SpeechModelStore model;
  static const sampleRate = 16000;

  /// Longest utterance before listening stops on its own.
  static const maxUtterance = Duration(seconds: 15);

  static bool _bindings = false;
  sherpa.OnlineRecognizer? _recognizer;
  final _recorder = AudioRecorder();
  StreamController<Hearing>? _out;
  StreamSubscription<Uint8List>? _mic;
  sherpa.OnlineStream? _stream;
  Timer? _timeout;
  String _heard = '';

  sherpa.OnlineRecognizer _load() {
    if (!_bindings) {
      sherpa.initBindings();
      _bindings = true;
    }
    return _recognizer ??= sherpa.OnlineRecognizer(
      sherpa.OnlineRecognizerConfig(
        model: sherpa.OnlineModelConfig(
          transducer: sherpa.OnlineTransducerModelConfig(
            encoder: model.path('encoder'),
            decoder: model.path('decoder'),
            joiner: model.path('joiner'),
          ),
          tokens: model.path('tokens'),
          numThreads: 2,
          debug: false,
        ),
        // An utterance ends after ~0.8 s of silence once something was said.
        rule2MinTrailingSilence: 0.8,
      ),
    );
  }

  @override
  Stream<Hearing> listen() {
    final out = _out = StreamController<Hearing>();
    _start(out);
    return out.stream;
  }

  Future<void> _start(StreamController<Hearing> out) async {
    try {
      if (!model.installed) {
        throw const SpeechModelException('The speech model is not installed.');
      }
      if (!await _recorder.hasPermission()) {
        throw const SpeechModelException('Microphone access was denied.');
      }
      final recognizer = _load();
      final stream = _stream = recognizer.createStream();
      _heard = '';
      final pcm = await _recorder.startStream(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: sampleRate,
          numChannels: 1,
        ),
      );
      _timeout = Timer(maxUtterance, stop);
      _mic = pcm.listen((bytes) {
        stream.acceptWaveform(
          samples: pcm16ToFloat(bytes),
          sampleRate: sampleRate,
        );
        while (recognizer.isReady(stream)) {
          recognizer.decode(stream);
        }
        final text = recognizer.getResult(stream).text.trim();
        if (text != _heard) {
          _heard = text;
          out.add(Hearing(text));
        }
        if (recognizer.isEndpoint(stream) && _heard.isNotEmpty) stop();
      }, onError: (Object e) => _finish(error: e));
    } catch (e) {
      _finish(error: e);
    }
  }

  @override
  Future<void> stop() async {
    final stream = _stream;
    final recognizer = _recognizer;
    if (stream != null && recognizer != null) {
      stream.inputFinished();
      while (recognizer.isReady(stream)) {
        recognizer.decode(stream);
      }
      _heard = recognizer.getResult(stream).text.trim();
    }
    await _finish();
  }

  Future<void> _finish({Object? error}) async {
    _timeout?.cancel();
    _timeout = null;
    await _mic?.cancel();
    _mic = null;
    if (await _recorder.isRecording()) await _recorder.stop();
    _stream?.free();
    _stream = null;
    final out = _out;
    _out = null;
    if (out == null || out.isClosed) return;
    if (error != null) {
      out.addError(error);
    } else {
      out.add(Hearing(_heard, isFinal: true));
    }
    await out.close();
  }

  @override
  void dispose() {
    _finish();
    _recognizer?.free();
    _recognizer = null;
    _recorder.dispose();
  }
}

/// Little-endian signed 16-bit PCM to floats in [-1, 1).
Float32List pcm16ToFloat(Uint8List bytes) {
  final data = ByteData.sublistView(bytes);
  final out = Float32List(bytes.length ~/ 2);
  for (var i = 0; i < out.length; i++) {
    out[i] = data.getInt16(i * 2, Endian.little) / 32768.0;
  }
  return out;
}
