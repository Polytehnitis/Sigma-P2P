import 'dart:math' as math;
import 'dart:typed_data';

/// Where an utterance stands, as far as the microphone can tell.
enum Endpoint {
  /// Still waiting for the user to start speaking, or still speaking.
  listening,

  /// Speech, then enough silence: the utterance is over.
  spoken,

  /// Nobody spoke before [Endpointer.noSpeechTimeout].
  nothing,

  /// Spoke for [Endpointer.maxUtterance]: cut there.
  tooLong,
}

/// Finds the end of one utterance in microphone audio, by loudness against the room's own
/// noise floor. whisper.cpp transcribes whole utterances, so this decides when to stop
/// recording, and whether anything was said at all (whisper makes words up from silence).
class Endpointer {
  Endpointer({
    this.sampleRate = 16000,
    this.minSpeech = const Duration(milliseconds: 200),
    this.trailingSilence = const Duration(milliseconds: 800),
    this.noSpeechTimeout = const Duration(seconds: 6),
    this.maxUtterance = const Duration(seconds: 15),
  }) : _frame = sampleRate ~/ 50;

  final int sampleRate;

  /// Loud frames needed in a row before it counts as speech (a click or a cough is shorter).
  final Duration minSpeech;
  final Duration trailingSilence;
  final Duration noSpeechTimeout;
  final Duration maxUtterance;

  /// 20 ms frames.
  final int _frame;
  final _pending = <double>[];
  int _frames = 0;
  int _loudRun = 0;
  int _quietRun = 0;
  bool _speech = false;

  /// RMS of the room when nobody speaks; starts low and follows quiet frames.
  double _floor = 0.003;

  /// Quietest level that counts as speech, whatever the floor.
  static const minLevel = 0.01;

  /// A loud room (or speaking right at the tap) never raises the bar above this.
  static const maxLevel = 0.06;

  /// 200 ms of the room's own noise.
  static const calibrationFrames = 10;
  double _calibration = 0;

  bool get heardSpeech => _speech;

  int _framesOf(Duration d) =>
      (d.inMicroseconds * sampleRate / 1e6 / _frame).ceil();

  Endpoint add(Float32List samples) {
    _pending.addAll(samples);
    var state = Endpoint.listening;
    var at = 0;
    while (_pending.length - at >= _frame && state == Endpoint.listening) {
      var sum = 0.0;
      for (var i = at; i < at + _frame; i++) {
        sum += _pending[i] * _pending[i];
      }
      at += _frame;
      state = _step(math.sqrt(sum / _frame));
    }
    _pending.removeRange(0, at);
    return state;
  }

  Endpoint _step(double rms) {
    _frames++;
    // The first frames, before anyone can start speaking after the tap, are the room.
    if (_frames <= calibrationFrames) {
      _calibration += rms;
      if (_frames == calibrationFrames) {
        _floor = math.max(_floor, _calibration / calibrationFrames);
      }
      return Endpoint.listening;
    }
    final loud = rms > math.max(minLevel, math.min(_floor * 3.5, maxLevel));
    if (loud) {
      _loudRun++;
      _quietRun = 0;
      if (_loudRun >= _framesOf(minSpeech)) _speech = true;
    } else {
      _loudRun = 0;
      _quietRun++;
      // The floor follows the room slowly, and only in quiet frames.
      _floor = 0.95 * _floor + 0.05 * rms;
    }
    if (_speech && _quietRun >= _framesOf(trailingSilence)) {
      return Endpoint.spoken;
    }
    if (_frames >= _framesOf(maxUtterance)) {
      return _speech ? Endpoint.tooLong : Endpoint.nothing;
    }
    if (!_speech && _loudRun == 0 && _frames >= _framesOf(noSpeechTimeout)) {
      return Endpoint.nothing;
    }
    return Endpoint.listening;
  }
}
