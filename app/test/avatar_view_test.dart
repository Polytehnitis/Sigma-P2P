import 'package:app/avatar/avatar_mood.dart';
import 'package:app/avatar/avatar_painter.dart';
import 'package:app/avatar/avatar_view.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';

AvatarMood mood(
  Expression e,
  Gesture g, {
  String? bubble,
  String? teach,
  bool rate = false,
}) => AvatarMood(
  expression: e,
  gesture: g,
  energy: 0.5,
  gaze: Offset.zero,
  glow: const Color(0xFFFFD45C),
  bubble: bubble,
  focus: 'GoalTidyDesk',
  teachGoal: teach,
  canRate: rate,
);

Future<void> pumpAvatar(WidgetTester tester, SigmaAvatar avatar) async {
  await tester.pumpWidget(
    MaterialApp(
      home: Scaffold(body: Center(child: avatar)),
    ),
  );
  await tester.pump(const Duration(milliseconds: 300));
}

void main() {
  testWidgets('every expression and gesture paints', (tester) async {
    for (final e in Expression.values) {
      for (final g in Gesture.values) {
        await pumpAvatar(tester, SigmaAvatar(mood: mood(e, g, bubble: 'hi')));
        await tester.pump(const Duration(milliseconds: 500));
        expect(tester.takeException(), isNull, reason: '$e $g');
      }
    }
  });

  testWidgets('click thinks, double-click opens Sigma', (tester) async {
    var thinks = 0, opens = 0;
    await pumpAvatar(
      tester,
      SigmaAvatar(
        mood: mood(Expression.idle, Gesture.rest),
        onThink: () => thinks++,
        onOpen: () => opens++,
      ),
    );
    await tester.tap(find.byKey(const Key('avatar-body')));
    await tester.pump(const Duration(milliseconds: 400));
    expect(thinks, 1);
    await tester.tap(find.byKey(const Key('avatar-body')));
    await tester.pump(const Duration(milliseconds: 50));
    await tester.tap(find.byKey(const Key('avatar-body')));
    await tester.pump(const Duration(milliseconds: 400));
    expect(opens, 1);
  });

  testWidgets('no thinking while busy', (tester) async {
    var thinks = 0;
    await pumpAvatar(
      tester,
      SigmaAvatar(
        mood: mood(Expression.idle, Gesture.rest),
        busy: true,
        onThink: () => thinks++,
      ),
    );
    await tester.tap(find.byKey(const Key('avatar-body')));
    await tester.pump(const Duration(milliseconds: 400));
    expect(thinks, 0);
  });

  testWidgets('thumbs give feedback on a selected action', (tester) async {
    final rewards = <double>[];
    await pumpAvatar(
      tester,
      SigmaAvatar(
        mood: mood(Expression.hopeful, Gesture.point, rate: true),
        onFeedback: rewards.add,
        alwaysShowControls: true,
      ),
    );
    await tester.tap(find.byKey(const Key('avatar-thumb-up')));
    await tester.tap(find.byKey(const Key('avatar-thumb-down')));
    expect(rewards, [1, -1]);
  });

  testWidgets('thumbs only appear on hover on desktop', (tester) async {
    await pumpAvatar(
      tester,
      SigmaAvatar(
        mood: mood(Expression.hopeful, Gesture.point, rate: true),
        onFeedback: (_) {},
      ),
    );
    expect(find.byKey(const Key('avatar-thumb-up')), findsNothing);
    final mouse = await tester.createGesture(kind: PointerDeviceKind.mouse);
    await mouse.addPointer(location: Offset.zero);
    await mouse.moveTo(tester.getCenter(find.byKey(const Key('avatar-body'))));
    await tester.pump(const Duration(milliseconds: 50));
    expect(find.byKey(const Key('avatar-thumb-up')), findsOneWidget);
    await mouse.removePointer();
  });

  testWidgets('an impasse offers teaching', (tester) async {
    var teach = 0;
    await pumpAvatar(
      tester,
      SigmaAvatar(
        mood: mood(
          Expression.worried,
          Gesture.shrug,
          bubble: 'Teach me?',
          teach: 'GoalTidyDesk',
        ),
        onTeach: () => teach++,
      ),
    );
    expect(find.text('Teach me?'), findsOneWidget);
    await tester.tap(find.byKey(const Key('avatar-teach')));
    expect(teach, 1);
  });

  testWidgets('a note replaces the bubble', (tester) async {
    await pumpAvatar(
      tester,
      SigmaAvatar(
        mood: mood(Expression.idle, Gesture.rest, bubble: 'mood'),
        note: 'Thanks!',
      ),
    );
    expect(find.text('Thanks!'), findsOneWidget);
    expect(find.text('mood'), findsNothing);
  });

  testWidgets('the right-click menu rates, teaches and opens', (tester) async {
    final calls = <String>[];
    await pumpAvatar(
      tester,
      SigmaAvatar(
        mood: mood(Expression.idle, Gesture.point, rate: true),
        onThink: () => calls.add('think'),
        onOpen: () => calls.add('open'),
        onFeedback: (r) => calls.add('rate $r'),
        onTeach: () => calls.add('teach'),
        autoThink: true,
        onAutoThinkChanged: (v) => calls.add('auto $v'),
      ),
    );
    for (final item in [
      'Good choice',
      'Teach a skill…',
      'Think on my own',
      'Open Sigma',
    ]) {
      await tester.tapAt(
        tester.getTopLeft(find.byKey(const Key('avatar-body'))) +
            const Offset(20, 20),
        buttons: kSecondaryButton,
      );
      for (var i = 0; i < 10; i++) {
        await tester.pump(const Duration(milliseconds: 100));
      }
      await tester.tap(
        find.ancestor(
          of: find.text(item),
          matching: find.byWidgetPredicate((w) => w is PopupMenuEntry),
        ),
      );
      await tester.pump(const Duration(milliseconds: 400));
    }
    expect(calls, ['rate 1.0', 'teach', 'auto false', 'open']);
  });

  test('poses ease between moods', () {
    final a = AvatarPose.of(Expression.sad, Gesture.droop);
    final b = AvatarPose.of(Expression.happy, Gesture.cheer);
    final mid = AvatarPose.lerp(a, b, 0.5);
    expect(mid.mouthCurve, closeTo((a.mouthCurve + b.mouthCurve) / 2, 1e-9));
    expect(AvatarPose.lerp(a, b, 1).leftArm, b.leftArm);
    expect(b.leftArm, greaterThan(a.leftArm), reason: 'cheering raises arms');
    expect(b.mouthCurve, greaterThan(0));
    expect(a.mouthCurve, lessThan(0));
  });
}
