import 'package:app/avatar/avatar_mood.dart';
import 'package:app/src/rust/api/agent.dart';
import 'package:flutter_test/flutter_test.dart';

CognitiveStep step({
  List<String> goals = const ['GoalTidyDesk'],
  String? action,
  String? impasse,
  DecisionImpasse? decision,
  Emotion emotion = Emotion.neutral,
  double desirability = 0,
  double surprise = 0,
  double control = 0,
  List<String> attended = const [],
}) => CognitiveStep(
  goalStack: goals,
  selectedAction: action,
  beliefs: const [],
  provenance: const [],
  impasse: impasse,
  decisionImpasse: decision,
  appraisal: Appraisal(
    desirability: desirability,
    surprise: surprise,
    control: control,
    familiarity: 0,
    emotion: emotion,
  ),
  attendedSkills: attended,
);

void main() {
  test('asleep before the agent starts, awake before the first step', () {
    expect(AvatarMood.of(null, online: false).expression, Expression.asleep);
    final awake = AvatarMood.of(null, online: true);
    expect(awake.expression, Expression.idle);
    expect(awake.gesture, Gesture.wave);
    expect(awake.canRate, isFalse);
  });

  test('a missing skill makes the avatar ask to be taught', () {
    final m = AvatarMood.of(
      step(impasse: 'GoalTidyDesk', emotion: Emotion.fear),
      online: true,
    );
    expect(m.expression, Expression.worried);
    expect(m.gesture, Gesture.shrug);
    expect(m.needsTeaching, isTrue);
    expect(m.teachGoal, 'GoalTidyDesk');
    expect(m.bubble, contains('tidy desk'));
    expect(m.canRate, isFalse);
  });

  test('a tie between actions shows pondering over both candidates', () {
    final m = AvatarMood.of(
      step(
        decision: const DecisionImpasse(
          kind: DecisionImpasseKind.tie,
          goal: 'GoalTidyDesk',
          candidates: ['ActionSortPapers', 'ActionDust'],
          resolved: false,
          cycles: 1,
        ),
      ),
      online: true,
    );
    expect(m.expression, Expression.thinking);
    expect(m.gesture, Gesture.ponder);
    expect(m.bubble, 'Hmm, sort papers or dust?');
  });

  test('a resolved decision impasse is shown as the chosen action', () {
    final m = AvatarMood.of(
      step(
        action: 'ActionSortPapers',
        emotion: Emotion.hope,
        control: 0.9,
        decision: const DecisionImpasse(
          kind: DecisionImpasseKind.tie,
          goal: 'GoalTidyDesk',
          candidates: ['ActionSortPapers', 'ActionDust'],
          resolved: true,
          cycles: 1,
        ),
      ),
      online: true,
    );
    expect(m.expression, Expression.hopeful);
    expect(m.gesture, Gesture.point);
    expect(m.bubble, 'I know this one: sort papers.');
    expect(m.canRate, isTrue);
  });

  test('a no-change impasse still lets the user rate the repeated action', () {
    final m = AvatarMood.of(
      step(
        action: 'ActionSortPapers',
        decision: const DecisionImpasse(
          kind: DecisionImpasseKind.noChange,
          goal: 'GoalTidyDesk',
          candidates: ['ActionSortPapers'],
          resolved: false,
          cycles: 3,
        ),
      ),
      online: true,
    );
    expect(m.expression, Expression.thinking);
    expect(m.canRate, isTrue);
    expect(m.bubble, contains('sort papers'));
  });

  test('no candidate actions asks to be taught the goal', () {
    final m = AvatarMood.of(
      step(
        decision: const DecisionImpasse(
          kind: DecisionImpasseKind.noCandidates,
          goal: 'GoalWaterPlants',
          candidates: [],
          resolved: false,
          cycles: 1,
        ),
      ),
      online: true,
    );
    expect(m.teachGoal, 'GoalWaterPlants');
  });

  test('each emotion has its own face and gesture', () {
    final faces = {
      for (final e in Emotion.values)
        e: AvatarMood.of(step(emotion: e, action: 'ActionX'), online: true),
    };
    expect(faces[Emotion.joy]!.expression, Expression.happy);
    expect(faces[Emotion.joy]!.gesture, Gesture.cheer);
    expect(faces[Emotion.distress]!.expression, Expression.sad);
    expect(faces[Emotion.distress]!.gesture, Gesture.droop);
    expect(faces[Emotion.surprise]!.expression, Expression.surprised);
    expect(faces[Emotion.hope]!.expression, Expression.hopeful);
    expect(faces[Emotion.fear]!.expression, Expression.worried);
    expect(faces[Emotion.neutral]!.gesture, Gesture.point);
    expect(faces.values.map((m) => m.glow).toSet(), hasLength(6));
  });

  test('energy follows surprise and desirability', () {
    final calm = AvatarMood.of(step(), online: true).energy;
    final excited = AvatarMood.of(
      step(surprise: 1, desirability: 1, emotion: Emotion.joy),
      online: true,
    ).energy;
    final down = AvatarMood.of(
      step(desirability: -1, emotion: Emotion.distress),
      online: true,
    ).energy;
    expect(excited, greaterThan(calm));
    expect(down, lessThan(calm));
    for (final e in [calm, excited, down]) {
      expect(e, inInclusiveRange(0, 1));
    }
  });

  test('attention focus prefers attended skills, then the deepest goal', () {
    expect(AvatarMood.of(step(attended: ['tidy']), online: true).focus, 'tidy');
    expect(
      AvatarMood.of(
        step(goals: ['GoalBackup', 'SubgoalMountNAS']),
        online: true,
      ).focus,
      'SubgoalMountNAS',
    );
    expect(
      AvatarMood.of(step(goals: [], action: 'ActionPing'), online: true).focus,
      'ActionPing',
    );
  });

  test('all goals done is celebrated', () {
    final m = AvatarMood.of(step(goals: []), online: true);
    expect(m.expression, Expression.happy);
    expect(m.bubble, 'All goals done!');
  });

  test('humanize turns predicate names into words', () {
    expect(humanize('ActionSortPapers'), 'sort papers');
    expect(humanize('GoalBackup'), 'backup');
    expect(humanize('SubgoalMountNAS'), 'mount nas');
    expect(humanize('Does(bob,ActionWave)'), 'wave');
    expect(humanize('water_plants'), 'water plants');
    expect(humanize('Goal'), 'Goal');
  });
}
