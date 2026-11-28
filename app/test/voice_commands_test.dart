import 'dart:io';
import 'dart:typed_data';

import 'package:app/src/rust/api/agent.dart';
import 'package:app/voice/speech_input.dart';
import 'package:app/src/rust/api/speech.dart' show SpeechEngine;
import 'package:app/voice/speech_model.dart';
import 'package:app/voice/voice_commands.dart';
import 'package:flutter_test/flutter_test.dart';

/// Records what the voice layer asks of the agent. It has no way to confirm anything,
/// just like the real [VoiceActions].
class FakeActions implements VoiceActions {
  final calls = <String>[];
  @override
  String? pendingAction;
  @override
  bool canUndo = false;

  @override
  Future<void> setGoal(String goal) async => calls.add('goal $goal');
  @override
  Future<void> observe(
    String predicate,
    int value, {
    required double confidence,
  }) async => calls.add('observe $predicate=$value @$confidence');
  @override
  Future<void> think() async => calls.add('think');
  @override
  Future<void> feedback(double reward) async => calls.add('feedback $reward');
  @override
  Future<void> undo() async => calls.add('undo');
  @override
  Future<void> cancel() async => calls.add('cancel');
  @override
  Future<void> teach() async => calls.add('teach');
  @override
  String explain() => 'explaining';
}

SpokenIntent intent(IntentKind kind, {String? predicate, int value = 0}) =>
    SpokenIntent(kind: kind, predicate: predicate, value: value);

Interpretation heard(
  SpokenIntent? i, {
  double confidence = 0.9,
  String text = 'something',
  List<ScoredIntent> alternatives = const [],
}) => Interpretation(
  normalized: text,
  intent: i,
  confidence: confidence,
  route: i == null ? InterpretationRoute.noMatch : InterpretationRoute.fastPath,
  alternatives: alternatives,
);

void main() {
  late FakeActions actions;
  late VoiceCommands voice;
  setUp(() {
    actions = FakeActions();
    voice = VoiceCommands(actions);
  });

  test('a spoken goal is pushed', () async {
    final r = await voice.handle(
      heard(intent(IntentKind.setGoal, predicate: 'GoalBackup', value: 1)),
    );
    expect(actions.calls, ['goal GoalBackup']);
    expect(r.text, 'OK, I will backup.');
  });

  test('observations carry the confidence they were heard with', () async {
    final o = intent(
      IntentKind.observe,
      predicate: 'SubgoalMountNAS',
      value: 1,
    );
    final sure = await voice.handle(heard(o, confidence: 0.9));
    final unsure = await voice.handle(heard(o, confidence: 0.4));
    expect(actions.calls, [
      'observe SubgoalMountNAS=1 @0.9',
      'observe SubgoalMountNAS=1 @0.4',
    ]);
    expect(sure.text, startsWith('Noted'));
    expect(unsure.text, startsWith('I think you said'));
  });

  test(
    '"yes, do it" never confirms; it points at the Confirm button',
    () async {
      actions.pendingAction = 'ActionDeleteBackups';
      final r = await voice.handle(heard(intent(IntentKind.confirmRequested)));
      expect(actions.calls, isEmpty);
      expect(r.text, contains('tap Confirm'));
      expect(r.text, contains('delete backups'));

      actions.pendingAction = null;
      final idle = await voice.handle(
        heard(intent(IntentKind.confirmRequested)),
      );
      expect(idle.text, 'Nothing is waiting for your approval.');
      expect(actions.calls, isEmpty);
    },
  );

  test('cancel and undo only act when there is something to act on', () async {
    await voice.handle(heard(intent(IntentKind.cancel)));
    await voice.handle(heard(intent(IntentKind.undo)));
    expect(actions.calls, isEmpty);
    actions
      ..pendingAction = 'ActionDeleteBackups'
      ..canUndo = true;
    await voice.handle(heard(intent(IntentKind.cancel)));
    await voice.handle(heard(intent(IntentKind.undo)));
    expect(actions.calls, ['cancel', 'undo']);
  });

  test('commands map to the agent', () async {
    await voice.handle(heard(intent(IntentKind.think)));
    await voice.handle(heard(intent(IntentKind.feedback, value: 1)));
    await voice.handle(heard(intent(IntentKind.feedback, value: 0)));
    await voice.handle(heard(intent(IntentKind.teach)));
    final e = await voice.handle(heard(intent(IntentKind.explain)));
    expect(actions.calls, ['think', 'feedback 1.0', 'feedback -1.0', 'teach']);
    expect(e.text, 'explaining');
  });

  test('no match offers the alternatives', () async {
    final alt = intent(IntentKind.setGoal, predicate: 'GoalBackup', value: 1);
    final r = await voice.handle(
      heard(null, alternatives: [ScoredIntent(intent: alt, score: 0.4)]),
    );
    expect(r.text, 'Did you mean…?');
    expect(r.suggestions, [alt]);
    expect(describeIntent(alt), 'backup');
    expect(actions.calls, isEmpty);
    final empty = await voice.handle(heard(null, text: ''));
    expect(empty.text, "I didn't catch that.");
  });

  test('PCM16 becomes floats', () {
    final bytes = Uint8List.fromList([0x00, 0x80, 0xff, 0x7f, 0x00, 0x00]);
    expect(pcm16ToFloat(bytes), [-1.0, 32767 / 32768, 0.0]);
  });

  test('the speech model is pinned and not installed until verified', () async {
    final dir = await Directory.systemTemp.createTemp('sigma-speech');
    final store = SpeechModelStore(dir, spec: SpeechModelSpec.desktop);
    expect(store.installed, isFalse);
    for (final spec in [SpeechModelSpec.desktop, SpeechModelSpec.mobile]) {
      expect(spec.url, startsWith('https://huggingface.co/'));
      expect(spec.url, endsWith('/${spec.file}'));
      expect(spec.sha256, matches(RegExp(r'^[0-9a-f]{64}$')));
      expect(spec.bytes, greaterThan(10 * 1000 * 1000));
    }
    expect(store.file.path, endsWith('/ggml-parakeet-tdt-0.6b-v3-q8_0.bin'));
    expect(SpeechModelSpec.desktop.engine, SpeechEngine.parakeet);
    expect(SpeechModelSpec.mobile.engine, SpeechEngine.whisper);
    // A partial download is not a model.
    File('${store.file.path}.part').writeAsStringSync('half');
    expect(store.installed, isFalse);
  });
}
