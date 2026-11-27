import 'dart:io';

import 'package:app/src/rust/api/agent.dart';
import 'package:app/src/rust/frb_generated.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

/// The real keyboard, mouse and screen of the (virtual) display, through the FFI.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(() async => await RustLib.init());

  test(
    'desktop control is opt-in, confirmed, undoable and stoppable',
    () async {
      final dir = await Directory.systemTemp.createTemp('sigma-desktop');
      await initAgent(
        storageDir: dir.path,
        swarmPassphrase:
            'desktop-test-${DateTime.now().microsecondsSinceEpoch}',
        bootstrapPeers: const [],
      );
      expect(desktopControlSupported(), isTrue);
      var status = await desktopStatus();
      expect(status.control, isFalse, reason: 'off at every start');
      expect(status.perception, isFalse);
      expect(describeAction(action: 'desktop.key(ctrl+s)'), 'press ctrl+s');

      // Sigma knows to move the pointer; control is still off, so nothing happens.
      await teachAndShareSkill(
        skillName: 'point',
        predicates: [
          PredicateDemo(name: 'GoalPoint', role: Role.goal, value: 1),
          PredicateDemo(
            name: 'desktop.move(40,50)',
            role: Role.action,
            value: 1,
          ),
        ],
      );
      var step = await runCognitiveStep(
        evidence: const [],
        activeGoals: ['GoalPoint'],
      );
      expect(step.motorCommands, ['desktop.move(40,50)']);
      expect(step.desktopEvents.single.ok, isFalse);
      expect(step.desktopEvents.single.message, 'desktop control is off');

      // On: a pointer move runs by itself and can be undone.
      status = await setDesktopControl(enabled: true);
      expect(status.control, isTrue);
      step = await runCognitiveStep(evidence: const [], activeGoals: const []);
      expect(
        step.desktopEvents.single.ok,
        isTrue,
        reason: step.desktopEvents.single.message,
      );
      expect(step.desktopEvents.single.message, 'move the mouse to (40, 50)');
      expect(step.undoStack, contains('desktop.move(40,50)'));
      final undone = await undoLastAction();
      expect(undone.reverseAction, 'desktop.undo_move');

      // A click waits for the user's confirmation, then reaches the mouse.
      await teachAndShareSkill(
        skillName: 'point',
        predicates: [
          PredicateDemo(name: 'GoalPoint', role: Role.goal, value: 1),
          PredicateDemo(
            name: 'desktop.click(60,70)',
            role: Role.action,
            value: 1,
          ),
        ],
      );
      step = await runCognitiveStep(evidence: const [], activeGoals: const []);
      expect(step.motorCommands, isNot(contains('desktop.click(60,70)')));
      final req = step.awaitingConfirmation!;
      expect(req.action, 'desktop.click(60,70)');
      expect(await confirmAction(token: req.token), 'desktop.click(60,70)');

      // Perception: Sigma looks at the screen before each step.
      status = await setScreenPerception(enabled: true);
      expect(status.perception, isTrue);
      step = await runCognitiveStep(evidence: const [], activeGoals: const []);
      expect(
        step.desktopEvents.where(
          (e) => !e.ok && e.action == 'look at the screen',
        ),
        isEmpty,
      );

      // Stop: everything off, and a click waiting for confirmation is dropped.
      expect(step.awaitingConfirmation?.action, 'desktop.click(60,70)');
      status = await stopDesktopControl();
      expect(status.control, isFalse);
      expect(status.perception, isFalse);
      await expectLater(
        confirmAction(token: step.awaitingConfirmation!.token),
        throwsA(anything),
      );
    },
  );
}
