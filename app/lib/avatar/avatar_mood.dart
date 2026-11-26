import 'dart:math' as math;

import 'package:flutter/painting.dart';

import '../src/rust/api/agent.dart';

/// The avatar's face.
enum Expression {
  asleep,
  idle,
  happy,
  sad,
  surprised,
  hopeful,
  worried,
  thinking,
}

/// What the avatar does with its arms.
enum Gesture { rest, wave, cheer, droop, startle, point, ponder, shrug }

/// How the avatar shows one moment of Sigma's mind. Pure data, derived from the
/// last cognitive step, so the mapping can be tested without the Rust agent.
class AvatarMood {
  const AvatarMood({
    required this.expression,
    required this.gesture,
    required this.energy,
    required this.gaze,
    required this.glow,
    this.bubble,
    this.focus,
    this.teachGoal,
    this.canRate = false,
  });

  final Expression expression;
  final Gesture gesture;

  /// 0 (sluggish) to 1 (bouncy): how fast and how high the avatar bobs.
  final double energy;

  /// Where the eyes look, each axis in [-1, 1] (x: right, y: down).
  final Offset gaze;

  /// Colour of the antenna light.
  final Color glow;

  /// What the avatar says, if anything.
  final String? bubble;

  /// What Sigma is attending to, shown under the avatar.
  final String? focus;

  /// Set when Sigma is stuck on a missing skill that the user can demonstrate.
  final String? teachGoal;

  /// True when there is a selected action whose outcome can be rated.
  final bool canRate;

  bool get needsTeaching => teachGoal != null;

  static const asleep = AvatarMood(
    expression: Expression.asleep,
    gesture: Gesture.rest,
    energy: 0.1,
    gaze: Offset(0, 0.4),
    glow: Color(0xFF8A8FA3),
    bubble: 'Zzz… double-click to open Sigma and start me.',
  );

  static const awake = AvatarMood(
    expression: Expression.idle,
    gesture: Gesture.wave,
    energy: 0.5,
    gaze: Offset.zero,
    glow: Color(0xFF7FB8FF),
    bubble: 'Hi! Click me and I will think.',
  );

  /// Maps a cognitive step to the avatar. `online` is false before the agent starts.
  factory AvatarMood.of(CognitiveStep? step, {required bool online}) {
    if (!online) return asleep;
    if (step == null) return awake;
    final a = step.appraisal;
    final focus = _focus(step);
    final action = step.selectedAction;
    final energy =
        (0.35 +
                0.35 * a.surprise +
                0.25 * math.max(a.desirability, 0) +
                0.15 * a.control -
                0.25 * math.max(-a.desirability, 0))
            .clamp(0.05, 1.0)
            .toDouble();

    final missing = step.impasse;
    if (missing != null) {
      return AvatarMood(
        expression: Expression.worried,
        gesture: Gesture.shrug,
        energy: math.min(energy, 0.4),
        gaze: const Offset(0, -0.5),
        glow: _glow(Emotion.fear),
        bubble: "I don't know how to ${humanize(missing)} yet. Teach me?",
        focus: focus,
        teachGoal: missing,
      );
    }

    final d = step.decisionImpasse;
    if (d != null && !d.resolved) {
      return switch (d.kind) {
        DecisionImpasseKind.tie => AvatarMood(
          expression: Expression.thinking,
          gesture: Gesture.ponder,
          energy: 0.3,
          gaze: const Offset(0.6, -0.6),
          glow: const Color(0xFFB59CFF),
          bubble: d.candidates.length >= 2
              ? 'Hmm, ${humanize(d.candidates[0])} or ${humanize(d.candidates[1])}?'
              : 'Hmm, I cannot decide.',
          focus: focus,
        ),
        DecisionImpasseKind.noChange => AvatarMood(
          expression: Expression.thinking,
          gesture: Gesture.ponder,
          energy: 0.25,
          gaze: const Offset(-0.6, -0.5),
          glow: const Color(0xFFB59CFF),
          bubble: action == null
              ? 'Nothing changed. What should I try?'
              : 'I keep wanting to ${humanize(action)}, but nothing changes.',
          focus: focus,
          canRate: action != null,
        ),
        DecisionImpasseKind.noCandidates => AvatarMood(
          expression: Expression.worried,
          gesture: Gesture.shrug,
          energy: 0.3,
          gaze: const Offset(0, -0.5),
          glow: _glow(Emotion.fear),
          bubble: "I have no idea how to ${humanize(d.goal)}. Teach me?",
          focus: focus,
          teachGoal: d.goal,
        ),
      };
    }

    if (step.goalStack.isEmpty && action == null) {
      return AvatarMood(
        expression: Expression.happy,
        gesture: Gesture.cheer,
        energy: math.max(energy, 0.6),
        gaze: const Offset(0, -0.3),
        glow: _glow(Emotion.joy),
        bubble: 'All goals done!',
        focus: focus,
      );
    }

    final (expression, gesture, gaze) = switch (a.emotion) {
      Emotion.joy => (Expression.happy, Gesture.cheer, const Offset(0, -0.3)),
      Emotion.distress => (Expression.sad, Gesture.droop, const Offset(0, 0.7)),
      Emotion.surprise => (Expression.surprised, Gesture.startle, Offset.zero),
      Emotion.hope => (
        Expression.hopeful,
        Gesture.point,
        const Offset(0.7, -0.2),
      ),
      Emotion.fear => (
        Expression.worried,
        Gesture.shrug,
        const Offset(0, -0.5),
      ),
      Emotion.neutral => (
        Expression.idle,
        action == null ? Gesture.rest : Gesture.point,
        action == null ? Offset.zero : const Offset(0.7, 0),
      ),
    };
    return AvatarMood(
      expression: expression,
      gesture: gesture,
      energy: energy,
      gaze: gaze,
      glow: _glow(a.emotion),
      bubble: action == null ? null : _say(a.emotion, action),
      focus: focus,
      canRate: action != null,
    );
  }

  /// Attention focus: the attended skill, else the deepest open goal, else the action.
  static String? _focus(CognitiveStep s) {
    if (s.attendedSkills.isNotEmpty) return s.attendedSkills.first;
    if (s.goalStack.isNotEmpty) return s.goalStack.last;
    return s.selectedAction;
  }

  static String _say(Emotion e, String action) {
    final a = humanize(action);
    return switch (e) {
      Emotion.joy => "Yes! I'll $a.",
      Emotion.distress => "Oh no… I'll try to $a.",
      Emotion.surprise => "Whoa! Let me $a.",
      Emotion.hope => "I know this one: $a.",
      Emotion.fear => "Not sure, but I'll $a.",
      Emotion.neutral => "I'll $a.",
    };
  }

  static Color _glow(Emotion e) => switch (e) {
    Emotion.joy => const Color(0xFFFFD45C),
    Emotion.distress => const Color(0xFF6C8CD5),
    Emotion.surprise => const Color(0xFFFF9F5C),
    Emotion.hope => const Color(0xFF6EE7A8),
    Emotion.fear => const Color(0xFFFF6B7A),
    Emotion.neutral => const Color(0xFF7FB8FF),
  };
}

/// Turns a predicate name such as `ActionSortPapers` or `Does(bob,ActionX)` into
/// words a person would say: `sort papers`.
String humanize(String predicate) {
  var p = predicate.trim();
  final call = RegExp(r'^\w+\((.*)\)$').firstMatch(p);
  if (call != null) p = call.group(1)!.split(',').last.trim();
  p = p.replaceFirst(RegExp(r'^(Action|Subgoal|Goal)(?=[A-Z0-9_]|$)'), '');
  final words = p
      .replaceAll(RegExp(r'[_\-]+'), ' ')
      .replaceAllMapped(RegExp(r'(?<=[a-z0-9])(?=[A-Z])'), (_) => ' ')
      .trim()
      .toLowerCase()
      .split(RegExp(r'\s+'))
      .where((w) => w.isNotEmpty);
  final out = words.join(' ');
  return out.isEmpty ? predicate : out;
}
