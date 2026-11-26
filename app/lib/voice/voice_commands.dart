import '../avatar/avatar_mood.dart' show humanize;
import '../src/rust/api/agent.dart';

/// What spoken (or typed) language may do to the agent. There is deliberately no way to
/// confirm an action here: only the Confirm button holds the confirmation token.
abstract class VoiceActions {
  /// Pushes a goal and runs a cycle.
  Future<void> setGoal(String goal);

  /// Hard evidence (`certain`) or one cycle of soft evidence, then a cycle.
  Future<void> observe(
    String predicate,
    int value, {
    required double confidence,
  });
  Future<void> think();
  Future<void> feedback(double reward);
  Future<void> undo();
  Future<void> cancel();
  Future<void> teach();

  /// A sentence on what Sigma is doing and why.
  String explain();

  /// The action waiting for the user's confirmation, if any.
  String? get pendingAction;

  /// Whether there is an action to undo.
  bool get canUndo;
}

/// The answer to one utterance: what Sigma says, and readings to offer when it is unsure.
class VoiceReply {
  const VoiceReply(this.text, {this.suggestions = const []});
  final String text;

  /// "Did you mean…?" choices. Picking one should teach the phrase (see [teachPhrase]).
  final List<SpokenIntent> suggestions;
}

/// Spoken observations at or above this confidence become hard evidence; below it they are
/// perceived as uncertain evidence for one cycle.
const hardEvidenceConfidence = 0.7;

/// Plain-words description of an intent, for replies and suggestion chips.
String describeIntent(SpokenIntent i) {
  final p = humanize(i.predicate ?? '');
  return switch (i.kind) {
    IntentKind.setGoal => p,
    IntentKind.observe => i.value == 1 ? '$p is true' : '$p is not true',
    IntentKind.think => 'think about what to do next',
    IntentKind.feedback => i.value == 1 ? 'that was good' : 'that was bad',
    IntentKind.undo => 'undo the last action',
    IntentKind.cancel => 'cancel the waiting action',
    IntentKind.confirmRequested => 'approve the waiting action',
    IntentKind.teach => 'teach a skill',
    IntentKind.explain => 'say what I am doing',
  };
}

/// Turns interpretations into agent calls and a reply.
class VoiceCommands {
  VoiceCommands(this.actions);
  final VoiceActions actions;

  Future<VoiceReply> handle(Interpretation i) async {
    final intent = i.intent;
    if (intent == null) {
      final options = [for (final a in i.alternatives) a.intent];
      return VoiceReply(
        i.normalized.isEmpty
            ? "I didn't catch that."
            : options.isEmpty
            ? "Sorry, I don't understand \"${i.normalized}\" yet."
            : 'Did you mean…?',
        suggestions: options,
      );
    }
    return run(intent, confidence: i.confidence);
  }

  /// Carries out one intent (also used when the user picks a suggestion).
  Future<VoiceReply> run(SpokenIntent intent, {double confidence = 1}) async {
    final p = intent.predicate ?? '';
    switch (intent.kind) {
      case IntentKind.setGoal:
        await actions.setGoal(p);
        return VoiceReply('OK, I will ${humanize(p)}.');
      case IntentKind.observe:
        await actions.observe(p, intent.value, confidence: confidence);
        final sure = confidence >= hardEvidenceConfidence;
        return VoiceReply(
          sure
              ? 'Noted: ${describeIntent(intent)}.'
              : 'I think you said ${describeIntent(intent)}. I will keep that in mind.',
        );
      case IntentKind.think:
        await actions.think();
        return const VoiceReply('Let me think.');
      case IntentKind.feedback:
        final good = intent.value == 1;
        await actions.feedback(good ? 1 : -1);
        return VoiceReply(
          good ? "Thanks! I'll do that again." : "Sorry. I'll do that less.",
        );
      case IntentKind.undo:
        if (!actions.canUndo) {
          return const VoiceReply('There is nothing to undo.');
        }
        await actions.undo();
        return const VoiceReply('Undoing that.');
      case IntentKind.cancel:
        if (actions.pendingAction == null) {
          return const VoiceReply('There is nothing to cancel.');
        }
        await actions.cancel();
        return const VoiceReply('Cancelled. I will not do it.');
      case IntentKind.confirmRequested:
        // Never confirmed by voice: the token lives only behind the Confirm button.
        final pending = actions.pendingAction;
        return VoiceReply(
          pending == null
              ? 'Nothing is waiting for your approval.'
              : 'To let me ${humanize(pending)}, tap Confirm. '
                    'Actions that cannot be undone need your tap, not your voice.',
        );
      case IntentKind.teach:
        await actions.teach();
        return const VoiceReply('Show me how.');
      case IntentKind.explain:
        return VoiceReply(actions.explain());
    }
  }
}
