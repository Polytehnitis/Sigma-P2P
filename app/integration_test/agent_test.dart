import 'dart:io';

import 'package:app/src/rust/api/agent.dart';
import 'package:app/src/rust/frb_generated.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(() async => await RustLib.init());

  test('impasse -> teach -> resolve through the FFI', () async {
    final dir = await Directory.systemTemp.createTemp('sigma-app');
    final info = await initAgent(
      storageDir: dir.path,
      swarmPassphrase: 'ffi-test-${DateTime.now().microsecondsSinceEpoch}',
      bootstrapPeers: const [],
    );
    expect(info.authorPubkey, hasLength(64));

    var step = await runCognitiveStep(
      evidence: const [],
      activeGoals: ['GoalTidyDesk'],
    );
    expect(step.impasse, 'GoalTidyDesk');
    expect(step.selectedAction, isNull);
    // The avatar's inputs: a lost agent is afraid and has no candidate action.
    expect(step.appraisal.emotion, Emotion.fear);
    expect(step.appraisal.control, 0);
    expect(step.decisionImpasse?.kind, DecisionImpasseKind.noCandidates);
    expect(step.decisionImpasse?.goal, 'GoalTidyDesk');

    await teachAndShareSkill(
      skillName: 'tidy',
      predicates: [
        PredicateDemo(name: 'GoalTidyDesk', role: Role.goal, value: 1),
        PredicateDemo(name: 'ActionSortPapers', role: Role.action, value: 1),
      ],
    );
    step = await runCognitiveStep(evidence: const [], activeGoals: const []);
    expect(step.goalStack, ['GoalTidyDesk']);
    expect(step.selectedAction, 'ActionSortPapers');
    expect(step.impasse, isNull);
    expect(step.provenance.single.authorPubkey, info.authorPubkey);
    expect(step.appraisal.control, inInclusiveRange(0, 1));
    expect(step.appraisal.desirability, inInclusiveRange(-1, 1));

    final fb = await provideFeedback(reward: 1);
    expect(fb.evictedSkills, isEmpty);
    expect(fb.reputations.single.reputation, 1.0);

    // Multi-device sync surface: no paired devices yet, nothing to reach.
    expect(await pairedDevices(), isEmpty);
    expect(await syncNow(), 0);
    final code = await createPairingCode();
    expect(code, startsWith('sigma-pair:${info.endpointId}:'));
    await expectLater(pairDevice(code: code), throwsA(anything));
    await expectLater(pairDevice(code: 'not-a-code'), throwsA(anything));
    // Errors come back as exceptions and the app keeps running.
    await expectLater(trustPeerAuthor(hexPubkey: 'zz'), throwsA(anything));
    expect(await pairedDevices(), isEmpty);
  });
}
