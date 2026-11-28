import 'dart:async';
import 'dart:typed_data';

import 'package:record/record.dart';

import '../src/rust/api/speech.dart' as rust;
import 'endpointer.dart';
import 'speech_model.dart';

/// What the recognizer heard so far. `isFinal` marks the end of an utterance.
class Hearing {
  const Hearing(
    this.text, {
    this.isFinal = false,
    this.confidence = 0,
    this.recognizing = false,
  });
  final String text;
  final bool isFinal;

  /// The recognizer's confidence in a final `text`, in [0, 1].
  final double confidence;

  /// The user stopped speaking and the words are being worked out.
  final bool recognizing;
}

/// A source of spoken utterances. One `listen` call hears one utterance.
abstract class SpeechInput {
  /// Partial hearings, then one final hearing (empty text if nothing was understood).
  Stream<Hearing> listen();

  /// Ends the current utterance early; what was heard so far becomes final.
  Future<void> stop();

  void dispose();
}

/// Turns one utterance (16 kHz mono samples) into text with the recognizer's confidence.
typedef Transcriber = Future<({String text, double confidence})> Function(
  Float32List samples,
);

/// On-device recognition with whisper.cpp (in the Rust core): the microphone (16 kHz mono
/// PCM) is recorded until the user stops speaking, then the whole utterance is transcribed.
/// No audio or text leaves the device.
class WhisperSpeechInput implements SpeechInput {
  WhisperSpeechInput(
    this.model, {
    Future<void> Function(String modelPath)? load,
    Transcriber? transcribe,
    AudioRecorder? recorder,
  }) : _load = load ?? ((path) => rust.loadSpeechModel(modelPath: path)),
       _transcribe = transcribe ?? _rustTranscribe,
       _recorder = recorder ?? AudioRecorder();

  final SpeechModelStore model;
  final Future<void> Function(String modelPath) _load;
  final Transcriber _transcribe;
  final AudioRecorder _recorder;
  static const sampleRate = 16000;

  bool _loaded = false;
  StreamController<Hearing>? _out;
  StreamSubscription<Uint8List>? _mic;
  Endpointer? _endpointer;
  final _audio = BytesBuilder(copy: false);

  static Future<({String text, double confidence})> _rustTranscribe(
    Float32List samples,
  ) async {
    final h = await rust.transcribeSpeech(samples: samples);
    return (text: h.text, confidence: h.confidence);
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
      if (!_loaded) {
        await _load(model.file.path);
        _loaded = true;
      }
      _audio.clear();
      final endpointer = _endpointer = Endpointer(sampleRate: sampleRate);
      final pcm = await _recorder.startStream(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: sampleRate,
          numChannels: 1,
          // Whisper is trained on clean speech; the platform's own cleanup helps it.
          noiseSuppress: true,
          echoCancel: true,
        ),
      );
      _mic = pcm.listen((bytes) {
        _audio.add(bytes);
        switch (endpointer.add(pcm16ToFloat(bytes))) {
          case Endpoint.listening:
            break;
          case Endpoint.spoken || Endpoint.tooLong:
            stop();
          case Endpoint.nothing:
            _finish(transcribe: false);
        }
      }, onError: (Object e) => _finish(error: e));
    } catch (e) {
      _finish(error: e);
    }
  }

  /// Ends the utterance now. Tapping stop while speaking still transcribes what was said.
  @override
  Future<void> stop() => _finish(
    // Or a second of audio: a quiet voice may never have crossed the loudness bar.
    transcribe:
        (_endpointer?.heardSpeech ?? false) || _audio.length >= 2 * sampleRate,
  );

  Future<void> _finish({Object? error, bool transcribe = false}) async {
    final out = _out;
    _out = null;
    _endpointer = null;
    await _mic?.cancel();
    _mic = null;
    if (await _recorder.isRecording()) await _recorder.stop();
    if (out == null || out.isClosed) return;
    try {
      if (error != null) throw error;
      var heard = (text: '', confidence: 0.0);
      if (transcribe) {
        out.add(const Hearing('', recognizing: true));
        heard = await _transcribe(pcm16ToFloat(_audio.takeBytes()));
      }
      out.add(
        Hearing(heard.text.trim(), isFinal: true, confidence: heard.confidence),
      );
    } catch (e) {
      out.addError(e);
    } finally {
      _audio.clear();
      await out.close();
    }
  }

  @override
  void dispose() {
    _finish();
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
