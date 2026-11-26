import 'dart:async';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:material_ui/material_ui.dart';
import 'package:path_provider/path_provider.dart';

import 'avatar/avatar_mood.dart';
import 'avatar/avatar_view.dart';
import 'avatar/avatar_window.dart';
import 'src/rust/api/agent.dart';
import 'src/rust/frb_generated.dart';
import 'voice/speech_input.dart';
import 'voice/speech_model.dart';
import 'voice/voice_commands.dart';

Future<void> main() async {
  await RustLib.init();
  runApp(const SigmaApp());
}

class SigmaApp extends StatelessWidget {
  const SigmaApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'Sigma',
    theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
    darkTheme: ThemeData(
      colorSchemeSeed: Colors.indigo,
      brightness: Brightness.dark,
      useMaterial3: true,
    ),
    home: const AgentPage(),
  );
}

class AgentPage extends StatefulWidget {
  const AgentPage({super.key});

  @override
  State<AgentPage> createState() => _AgentPageState();
}

class _AgentPageState extends State<AgentPage> implements VoiceActions {
  final _swarm = TextEditingController(text: 'sigma-public');
  final _peer = TextEditingController();
  final _trust = TextEditingController();
  final _goal = TextEditingController(text: 'GoalBackup');
  final _evidence = TextEditingController();
  final _pairCode = TextEditingController();
  AgentInfo? _info;
  List<String> _devices = const [];
  CognitiveStep? _step;
  ConfirmationRequest? _pending;
  List<String> _undoStack = const [];
  bool _busy = false;

  /// Desktop avatar mode: the window becomes a small always-on-top companion.
  bool _avatar = false;
  final _avatarWindow = AvatarWindow();
  bool _autoThink = true;
  Timer? _heartbeat;
  String? _note;
  Timer? _noteTimer;

  /// Voice: on-device recognition, then the language layer in Rust.
  final _utterance = TextEditingController();
  late final _voice = VoiceCommands(this);
  SpeechModelStore? _speechModel;
  SpeechInput? _speech;
  bool _listening = false;
  String _heard = '';
  double? _modelProgress;
  String? _lastUtterance;
  List<SpokenIntent> _suggestions = const [];

  @override
  void dispose() {
    _heartbeat?.cancel();
    _noteTimer?.cancel();
    _speech?.dispose();
    super.dispose();
  }

  Future<void> _guard(Future<void> Function() body) async {
    setState(() => _busy = true);
    try {
      await body();
    } catch (e) {
      _toast('$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  void _toast(String msg) {
    if (_avatar) return _say(msg);
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
    }
  }

  /// Shows `msg` in the avatar's speech bubble for a few seconds.
  void _say(String msg) {
    _noteTimer?.cancel();
    setState(() => _note = msg);
    _noteTimer = Timer(const Duration(seconds: 5), () {
      if (mounted) setState(() => _note = null);
    });
  }

  void _enterAvatar() {
    final error = _avatarWindow.enter();
    setState(() => _avatar = true);
    if (error != null) _say(error);
    _scheduleHeartbeat();
  }

  void _exitAvatar() {
    _heartbeat?.cancel();
    final error = _avatarWindow.exit();
    setState(() {
      _avatar = false;
      _note = null;
    });
    if (error != null) _toast(error);
  }

  /// While the avatar is out, Sigma keeps thinking on its own every few seconds.
  void _scheduleHeartbeat() {
    _heartbeat?.cancel();
    if (!_avatar || !_autoThink) return;
    _heartbeat = Timer.periodic(const Duration(seconds: 10), (_) {
      if (_info != null && !_busy) _think();
    });
  }

  Future<void> _teachFromAvatar() async {
    _exitAvatar();
    await _teach();
    if (mounted) _enterAvatar();
  }

  Future<void> _start() => _guard(() async {
    final dir = await getApplicationDocumentsDirectory();
    final info = await initAgent(
      storageDir: '${dir.path}/sigma',
      swarmPassphrase: _swarm.text,
      bootstrapPeers: [_peer.text],
    );
    final devices = await pairedDevices();
    setState(() {
      _info = info;
      _devices = devices;
      _speechModel = SpeechModelStore(Directory('${dir.path}/sigma/speech'));
    });
    _toast('Agent online, ${info.restoredSkills} skill(s) restored');
  });

  Future<void> _showPairingCode() => _guard(() async {
    final code = await createPairingCode();
    if (!mounted) return;
    await showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Pair another device'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          spacing: 8,
          children: [
            const Text(
              'Enter this code on your other device within 10 minutes. '
              'It works once.',
            ),
            SelectableText(
              code,
              style: const TextStyle(fontFamily: 'monospace'),
            ),
          ],
        ),
        actions: [
          TextButton.icon(
            onPressed: () => Clipboard.setData(ClipboardData(text: code)),
            icon: const Icon(Icons.copy),
            label: const Text('Copy'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Done'),
          ),
        ],
      ),
    );
    final devices = await pairedDevices();
    setState(() => _devices = devices);
  });

  Future<void> _pair() => _guard(() async {
    final device = await pairDevice(code: _pairCode.text);
    _pairCode.clear();
    final devices = await pairedDevices();
    setState(() => _devices = devices);
    _toast('Paired with ${_short(device)} and synced');
  });

  Future<void> _syncNow() => _guard(() async {
    final reached = await syncNow();
    final devices = await pairedDevices();
    setState(() => _devices = devices);
    _toast('Synced with $reached of ${devices.length} device(s)');
  });

  Future<void> _unpair(String hex) => _guard(() async {
    await unpairDevice(hexPubkey: hex);
    final devices = await pairedDevices();
    setState(() => _devices = devices);
    _toast('Unpaired ${_short(hex)}');
  });

  Future<void> _run() => _guard(() async {
    final evidence = _evidence.text
        .split(',')
        .map((e) => e.split('='))
        .where((kv) => kv.length == 2 && kv[0].trim().isNotEmpty)
        .map(
          (kv) => Evidence(
            predicate: kv[0].trim(),
            value: int.tryParse(kv[1].trim()) ?? 0,
          ),
        )
        .toList();
    await _cycle(evidence, [_goal.text]);
  });

  /// One cognitive cycle without new observations or goals (the avatar's own pace).
  Future<void> _think() => _guard(() => _cycle(const [], const []));

  Future<void> _cycle(List<Evidence> evidence, List<String> goals) async {
    final step = await runCognitiveStep(evidence: evidence, activeGoals: goals);
    if (!mounted) return;
    setState(() {
      _step = step;
      _pending = step.awaitingConfirmation;
      _undoStack = step.undoStack;
    });
  }

  // ---- Voice ---------------------------------------------------------------

  /// Tap to talk: hears one utterance on this device, then acts on it.
  Future<void> _listen() async {
    if (_listening) return _speech?.stop();
    final model = _speechModel;
    if (model == null) return;
    if (!model.installed && !await _installSpeechModel(model)) return;
    final speech = _speech ??= SherpaSpeechInput(model);
    setState(() {
      _listening = true;
      _heard = '';
      _suggestions = const [];
    });
    try {
      await for (final h in speech.listen()) {
        if (!mounted) return;
        setState(() => _heard = h.text);
        if (h.isFinal) {
          setState(() => _listening = false);
          // The recognizer reports no confidence of its own; speech counts a bit below text.
          if (h.text.isNotEmpty) await _understand(h.text, 0.9);
          if (h.text.isEmpty) _toast("I didn't hear anything.");
        }
      }
    } catch (e) {
      _toast('$e');
    } finally {
      if (mounted) setState(() => _listening = false);
    }
  }

  Future<bool> _installSpeechModel(SpeechModelStore model) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Talk to Sigma'),
        content: Text(
          'Speech is recognized on this device. Sigma needs to download a '
          'speech model once (about ${model.spec.downloadMegabytes} MB, '
          'checked against a pinned checksum). Your voice never leaves '
          'this device.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Not now'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Download'),
          ),
        ],
      ),
    );
    if (ok != true) return false;
    setState(() => _modelProgress = 0);
    try {
      await model.install(
        onProgress: (p) {
          if (mounted) setState(() => _modelProgress = p);
        },
      );
      return true;
    } catch (e) {
      _toast('$e');
      return false;
    } finally {
      if (mounted) setState(() => _modelProgress = null);
    }
  }

  /// Typed or spoken: the language layer interprets it against Sigma's own predicates.
  Future<void> _understand(String text, double recognitionConfidence) =>
      _guard(() async {
        final i = await interpretUtterance(
          text: text,
          recognitionConfidence: recognitionConfidence,
        );
        final reply = await _voice.handle(i);
        setState(() {
          _lastUtterance = i.normalized;
          _suggestions = reply.suggestions;
        });
        _reply(reply);
      });

  /// The user picked a "did you mean" reading: remember the phrase, then do it.
  Future<void> _pickSuggestion(SpokenIntent intent) => _guard(() async {
    final phrase = _lastUtterance;
    setState(() => _suggestions = const []);
    if (phrase != null && phrase.isNotEmpty) {
      await teachPhrase(phrase: phrase, intent: intent);
    }
    _reply(await _voice.run(intent));
  });

  void _reply(VoiceReply reply) {
    final pending = _pending;
    final text = pending != null && !reply.text.contains('Confirm')
        ? '${reply.text} I need your OK to ${humanize(pending.action)}: '
              'tap Confirm.'
        : reply.text;
    _toast(text);
  }

  @override
  String? get pendingAction => _pending?.action;

  @override
  bool get canUndo => _undoStack.isNotEmpty;

  @override
  Future<void> setGoal(String goal) async {
    _goal.text = goal;
    await _cycle(const [], [goal]);
  }

  @override
  Future<void> observe(
    String predicate,
    int value, {
    required double confidence,
  }) async {
    if (confidence >= hardEvidenceConfidence) {
      return _cycle([Evidence(predicate: predicate, value: value)], const []);
    }
    // Unsure what was said: soft evidence for one cycle, leaning the way it was heard.
    final lean = 0.5 + 0.5 * confidence.clamp(0, 1);
    await perceive(
      predicate: predicate,
      probability: value == 1 ? lean : 1 - lean,
    );
    await _cycle(const [], const []);
  }

  @override
  Future<void> think() => _cycle(const [], const []);

  @override
  Future<void> feedback(double reward) => provideFeedback(reward: reward);

  @override
  Future<void> undo() async {
    await undoLastAction();
    setState(() => _undoStack = _undoStack.sublist(0, _undoStack.length - 1));
  }

  @override
  Future<void> cancel() async {
    await cancelPendingAction();
    setState(() => _pending = null);
  }

  @override
  Future<void> teach() async {
    // After the current voice step finishes, so the dialog is not under its busy state.
    Timer.run(() => _avatar ? _teachFromAvatar() : _teach());
  }

  @override
  String explain() {
    final step = _step;
    if (step == null) return "I haven't thought about anything yet.";
    final bubble = AvatarMood.of(step, online: _info != null).bubble;
    final goals = step.goalStack.map(humanize).join(', then ');
    return [
      bubble ?? "I'm not doing anything right now.",
      if (goals.isNotEmpty) 'My goals: $goals.',
    ].join(' ');
  }

  // The token comes only from this button: engine and skill data never carry it.
  Future<void> _confirm(ConfirmationRequest req) => _guard(() async {
    setState(() => _pending = null);
    _toast('Confirmed ${await confirmAction(token: req.token)}');
  });

  Future<void> _cancelPending() => _guard(() async {
    await cancelPendingAction();
    setState(() => _pending = null);
    _toast('Cancelled');
  });

  Future<void> _undo() => _guard(() async {
    final r = await undoLastAction();
    setState(() => _undoStack = _undoStack.sublist(0, _undoStack.length - 1));
    _toast(
      r.reverseAction == null
          ? 'Withdrew ${r.action}'
          : 'Undid ${r.action} with ${r.reverseAction}',
    );
  });

  Future<void> _editRules() async {
    final rules = await actionRules();
    if (!mounted) return;
    final text = await showDialog<String>(
      context: context,
      builder: (_) => RulesDialog(
        initial: rules
            .map(
              (r) => r.irreversible
                  ? '!${r.action}'
                  : '${r.action} > ${r.reverse ?? ''}',
            )
            .join('\n'),
      ),
    );
    if (text == null) return;
    await _guard(() async {
      await setActionRules(rules: parseActionRules(text));
      _toast('Action rules saved');
    });
  }

  Future<void> _feedback(double reward) => _guard(() async {
    final r = await provideFeedback(reward: reward);
    if (_avatar) {
      return _say(
        reward > 0
            ? "Thanks! I'll do that again."
            : "Sorry. I'll do that less.",
      );
    }
    final reps = r.reputations
        .map(
          (p) =>
              '${_short(p.authorPubkey)}: ${p.reputation.toStringAsFixed(2)}',
        )
        .join(', ');
    _toast(
      'Reputation $reps'
      '${r.evictedSkills.isEmpty ? '' : ' · evicted ${r.evictedSkills.join(', ')}'}',
    );
  });

  Future<void> _trustAuthor() => _guard(() async {
    await trustPeerAuthor(hexPubkey: _trust.text);
    _toast('Author trusted');
  });

  Future<void> _teach() async {
    final demo = await showDialog<(String, List<PredicateDemo>)>(
      context: context,
      builder: (_) => TeachDialog(goal: _step?.impasse ?? _goal.text),
    );
    if (demo == null) return;
    await _guard(() async {
      final id = await teachAndShareSkill(
        skillName: demo.$1,
        predicates: demo.$2,
      );
      _toast('Signed and shared $id');
    });
    await _run();
  }

  Widget _buildAvatar() {
    final avatar = SigmaAvatar(
      mood: AvatarMood.of(_step, online: _info != null),
      listening: _listening,
      onListen: _info == null ? null : _listen,
      note:
          (_listening ? (_heard.isEmpty ? 'Listening…' : '“$_heard”') : null) ??
          (_modelProgress == null
              ? null
              : 'Getting my ears… ${(_modelProgress! * 100).round()}%') ??
          _note ??
          (_pending == null
              ? null
              : 'May I ${humanize(_pending!.action)}? '
                    'Double-click me to confirm.'),
      busy: _busy,
      onThink: _info == null ? null : (_step == null ? _run : _think),
      onOpen: _exitAvatar,
      onFeedback: _info == null || _step == null ? null : _feedback,
      onTeach: _info == null ? null : _teachFromAvatar,
      onDragStart: AvatarWindow.supported ? _avatarWindow.startDragging : null,
      autoThink: _info == null ? null : _autoThink,
      onAutoThinkChanged: (v) {
        setState(() => _autoThink = v);
        _scheduleHeartbeat();
      },
      alwaysShowControls: !AvatarWindow.supported,
    );
    if (AvatarWindow.supported) {
      return Material(
        type: MaterialType.transparency,
        child: Center(child: avatar),
      );
    }
    return Scaffold(
      appBar: AppBar(
        title: const Text('Sigma'),
        leading: IconButton(
          tooltip: 'Back',
          onPressed: _exitAvatar,
          icon: const Icon(Icons.close),
        ),
      ),
      body: Center(child: avatar),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_avatar) return _buildAvatar();
    final step = _step;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Sigma cognitive agent'),
        actions: [
          if (_info != null)
            IconButton(
              key: const Key('voice-listen'),
              tooltip: _listening ? 'Stop listening' : 'Talk to Sigma',
              onPressed: _modelProgress != null ? null : _listen,
              isSelected: _listening,
              icon: const Icon(Icons.mic_none),
              selectedIcon: const Icon(Icons.mic),
            ),
          IconButton(
            tooltip: AvatarWindow.supported
                ? 'Put Sigma on the desktop'
                : 'Show Sigma',
            onPressed: _enterAvatar,
            icon: const Icon(Icons.face),
          ),
        ],
        bottom: _busy
            ? const PreferredSize(
                preferredSize: Size.fromHeight(4),
                child: LinearProgressIndicator(),
              )
            : null,
      ),
      body: ListView(
        padding: const EdgeInsets.all(12),
        children: [
          _section('Identity', [
            if (_info == null) ...[
              _field(_swarm, 'Swarm passphrase'),
              _field(_peer, 'Home NAS / bootstrap peer EndpointId'),
              FilledButton.icon(
                onPressed: _busy ? null : _start,
                icon: const Icon(Icons.power_settings_new),
                label: const Text('Start agent'),
              ),
            ] else ...[
              _copyRow('EndpointId', _info!.endpointId),
              _copyRow('Author Pubkey', _info!.authorPubkey),
              Row(
                children: [
                  Expanded(
                    child: _field(
                      _trust,
                      'Trust my device (author pubkey hex)',
                    ),
                  ),
                  IconButton(
                    onPressed: _busy ? null : _trustAuthor,
                    icon: const Icon(Icons.verified_user),
                  ),
                ],
              ),
            ],
          ]),
          if (_info != null)
            _section('My devices', [
              const Text(
                'Paired devices keep memories, reputations, learned weights '
                'and goals in sync, end-to-end encrypted. Nothing private '
                'goes to the swarm.',
              ),
              for (final d in _devices)
                ListTile(
                  dense: true,
                  contentPadding: EdgeInsets.zero,
                  leading: const Icon(Icons.devices),
                  title: Text(
                    _short(d),
                    style: const TextStyle(fontFamily: 'monospace'),
                  ),
                  trailing: IconButton(
                    tooltip: 'Unpair',
                    onPressed: _busy ? null : () => _unpair(d),
                    icon: const Icon(Icons.link_off),
                  ),
                ),
              if (_devices.isEmpty) const Text('No paired devices yet'),
              Row(
                children: [
                  Expanded(
                    child: _field(_pairCode, 'Pairing code from my device'),
                  ),
                  IconButton(
                    tooltip: 'Pair',
                    onPressed: _busy ? null : _pair,
                    icon: const Icon(Icons.add_link),
                  ),
                ],
              ),
              Wrap(
                spacing: 8,
                children: [
                  OutlinedButton.icon(
                    onPressed: _busy ? null : _showPairingCode,
                    icon: const Icon(Icons.qr_code),
                    label: const Text('Show pairing code'),
                  ),
                  OutlinedButton.icon(
                    onPressed: _busy ? null : _syncNow,
                    icon: const Icon(Icons.sync),
                    label: const Text('Sync now'),
                  ),
                ],
              ),
            ]),
          if (_info != null)
            _section('Talk to Sigma', [
              Row(
                spacing: 8,
                children: [
                  Expanded(
                    child: TextField(
                      key: const Key('voice-text'),
                      controller: _utterance,
                      decoration: const InputDecoration(
                        labelText: 'Say or type, e.g. "back up my files"',
                        border: OutlineInputBorder(),
                      ),
                      onSubmitted: (t) {
                        _utterance.clear();
                        _understand(t, 1);
                      },
                    ),
                  ),
                  IconButton.filled(
                    tooltip: _listening ? 'Stop listening' : 'Speak',
                    onPressed: _modelProgress != null ? null : _listen,
                    icon: Icon(_listening ? Icons.stop : Icons.mic),
                  ),
                ],
              ),
              if (_modelProgress case final p?)
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  spacing: 4,
                  children: [
                    const Text('Downloading the on-device speech model…'),
                    LinearProgressIndicator(value: p),
                  ],
                ),
              if (_listening)
                ListTile(
                  leading: const Icon(Icons.hearing),
                  title: Text(_heard.isEmpty ? 'Listening…' : _heard),
                  subtitle: const Text('Recognized on this device'),
                ),
              if (_suggestions.isNotEmpty)
                Wrap(
                  spacing: 8,
                  crossAxisAlignment: WrapCrossAlignment.center,
                  children: [
                    const Text('Did you mean'),
                    for (final s in _suggestions)
                      ActionChip(
                        label: Text(describeIntent(s)),
                        onPressed: _busy ? null : () => _pickSuggestion(s),
                      ),
                  ],
                ),
            ]),
          if (_info != null)
            _section('Goal', [
              _field(_goal, 'Top-level goal'),
              _field(_evidence, 'Observations, e.g. SubgoalMountNAS=1'),
              Wrap(
                spacing: 8,
                children: [
                  FilledButton.icon(
                    onPressed: _busy ? null : _run,
                    icon: const Icon(Icons.play_arrow),
                    label: const Text('Run cognitive step'),
                  ),
                  OutlinedButton.icon(
                    onPressed: _busy ? null : _editRules,
                    icon: const Icon(Icons.shield),
                    label: const Text('Action rules'),
                  ),
                ],
              ),
            ]),
          if (step != null) ...[
            _section('Goal stack', [
              Wrap(
                crossAxisAlignment: WrapCrossAlignment.center,
                children: [
                  for (final (i, g) in step.goalStack.indexed) ...[
                    if (i > 0) const Icon(Icons.chevron_right),
                    Chip(label: Text(g)),
                  ],
                  if (step.goalStack.isEmpty) const Text('All goals satisfied'),
                ],
              ),
            ]),
            _section('Decision', [
              ListTile(
                leading: const Icon(Icons.bolt),
                title: Text(step.selectedAction ?? 'No action selected'),
                subtitle: const Text('Rate the outcome to reinforce the chain'),
                trailing: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    IconButton(
                      onPressed: _busy ? null : () => _feedback(1),
                      icon: const Icon(Icons.thumb_up),
                    ),
                    IconButton(
                      onPressed: _busy ? null : () => _feedback(-1),
                      icon: const Icon(Icons.thumb_down),
                    ),
                  ],
                ),
              ),
              if (_pending case final req?)
                Card(
                  color: Theme.of(context).colorScheme.tertiaryContainer,
                  child: ListTile(
                    leading: const Icon(Icons.warning_amber),
                    title: Text('Confirm ${req.action}?'),
                    subtitle: const Text('This action cannot be undone.'),
                    trailing: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        TextButton(
                          onPressed: _busy ? null : _cancelPending,
                          child: const Text('Cancel'),
                        ),
                        FilledButton(
                          onPressed: _busy ? null : () => _confirm(req),
                          child: const Text('Confirm'),
                        ),
                      ],
                    ),
                  ),
                ),
              if (_undoStack.isNotEmpty)
                Align(
                  alignment: Alignment.centerLeft,
                  child: OutlinedButton.icon(
                    onPressed: _busy ? null : _undo,
                    icon: const Icon(Icons.undo),
                    label: Text('Undo ${_undoStack.last}'),
                  ),
                ),
              if (step.impasse != null)
                Card(
                  color: Theme.of(context).colorScheme.errorContainer,
                  child: ListTile(
                    leading: const Icon(Icons.link_off),
                    title: Text('Impasse: no skill bridges ${step.impasse}'),
                    subtitle: const Text(
                      'Asked the swarm. You can also demonstrate it.',
                    ),
                    trailing: TextButton(
                      onPressed: _teach,
                      child: const Text('Teach'),
                    ),
                  ),
                ),
            ]),
            _section('Beliefs', [
              for (final b in step.beliefs)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Row(
                    children: [
                      SizedBox(
                        width: 170,
                        child: Text(
                          b.predicate,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                      Expanded(
                        child: LinearProgressIndicator(
                          value: b.distribution.length > 1
                              ? b.distribution[1]
                              : 0,
                        ),
                      ),
                      SizedBox(
                        width: 48,
                        child: Text(
                          b.distribution.length > 1
                              ? b.distribution[1].toStringAsFixed(2)
                              : '-',
                          textAlign: TextAlign.end,
                        ),
                      ),
                    ],
                  ),
                ),
            ]),
            _section('Skill provenance', [
              for (final p in step.provenance)
                ListTile(
                  dense: true,
                  leading: Icon(
                    p.reputation >= 0.25 ? Icons.verified : Icons.block,
                  ),
                  title: Text(p.skillName),
                  subtitle: Text('author ${_short(p.authorPubkey)}'),
                  trailing: Text('rep ${p.reputation.toStringAsFixed(2)}'),
                ),
            ]),
          ],
        ],
      ),
      floatingActionButton: _info == null
          ? null
          : FloatingActionButton.extended(
              onPressed: _busy ? null : _teach,
              icon: const Icon(Icons.school),
              label: const Text('Teach skill'),
            ),
    );
  }

  Widget _section(String title, List<Widget> children) => Card(
    child: Padding(
      padding: const EdgeInsets.all(12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        spacing: 8,
        children: [
          Text(title, style: Theme.of(context).textTheme.titleMedium),
          ...children,
        ],
      ),
    ),
  );

  Widget _copyRow(String label, String value) => ListTile(
    contentPadding: EdgeInsets.zero,
    title: Text(label),
    subtitle: SelectableText(
      value,
      style: const TextStyle(fontFamily: 'monospace'),
    ),
    trailing: IconButton(
      icon: const Icon(Icons.copy),
      onPressed: () async {
        await Clipboard.setData(ClipboardData(text: value));
        _toast('$label copied');
      },
    ),
  );
}

Widget _field(TextEditingController c, String label) => TextField(
  controller: c,
  decoration: InputDecoration(
    labelText: label,
    border: const OutlineInputBorder(),
  ),
);

String _short(String hex) => hex.length > 12 ? '${hex.substring(0, 12)}…' : hex;

/// Collects a demonstration: which predicates were involved, their roles and values.
class TeachDialog extends StatefulWidget {
  const TeachDialog({super.key, required this.goal});
  final String goal;

  @override
  State<TeachDialog> createState() => _TeachDialogState();
}

class _TeachDialogState extends State<TeachDialog> {
  final _name = TextEditingController();
  late final List<(TextEditingController, Role, bool)> _rows = [
    (TextEditingController(text: widget.goal), Role.goal, true),
    (TextEditingController(text: 'Action'), Role.action, true),
  ];

  @override
  Widget build(BuildContext context) => AlertDialog(
    title: const Text('Demonstrate a skill'),
    content: SizedBox(
      width: 480,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        spacing: 8,
        children: [
          _field(_name, 'Skill name'),
          for (final (i, (c, role, value)) in _rows.indexed)
            Row(
              spacing: 8,
              children: [
                Expanded(child: _field(c, 'Predicate')),
                DropdownButton<Role>(
                  value: role,
                  items: [
                    for (final r in Role.values)
                      DropdownMenuItem(value: r, child: Text(r.name)),
                  ],
                  onChanged: (r) => setState(() => _rows[i] = (c, r!, value)),
                ),
                Switch(
                  value: value,
                  onChanged: (v) => setState(() => _rows[i] = (c, role, v)),
                ),
              ],
            ),
          TextButton.icon(
            onPressed: () => setState(
              () => _rows.add((TextEditingController(), Role.context, true)),
            ),
            icon: const Icon(Icons.add),
            label: const Text('Predicate'),
          ),
        ],
      ),
    ),
    actions: [
      TextButton(
        onPressed: () => Navigator.pop(context),
        child: const Text('Cancel'),
      ),
      FilledButton(
        onPressed: () => Navigator.pop(context, (
          _name.text.isEmpty ? 'skill-${widget.goal}' : _name.text,
          [
            for (final (c, role, value) in _rows)
              if (c.text.trim().isNotEmpty)
                PredicateDemo(
                  name: c.text.trim(),
                  role: role,
                  value: value ? 1 : 0,
                ),
          ],
        )),
        child: const Text('Sign & share'),
      ),
    ],
  );
}

/// Parses one rule per line: `Action > Reverse` (undoable) or `!Action` (irreversible).
List<ActionRule> parseActionRules(String text) => [
  for (final line in text.split('\n').map((l) => l.trim()))
    if (line.startsWith('!'))
      ActionRule(action: line.substring(1).trim(), irreversible: true)
    else if (line.contains('>'))
      ActionRule(
        action: line.split('>')[0].trim(),
        reverse: line.split('>')[1].trim(),
        irreversible: false,
      ),
];

class RulesDialog extends StatefulWidget {
  const RulesDialog({super.key, required this.initial});
  final String initial;

  @override
  State<RulesDialog> createState() => _RulesDialogState();
}

class _RulesDialogState extends State<RulesDialog> {
  late final _text = TextEditingController(text: widget.initial);

  @override
  Widget build(BuildContext context) => AlertDialog(
    title: const Text('Action rules'),
    content: TextField(
      controller: _text,
      minLines: 4,
      maxLines: 10,
      decoration: const InputDecoration(
        helperText:
            'One per line. ActionMountNAS > ActionUnmountNAS makes an '
            'action undoable; !ActionDeleteBackups always asks you first.',
        helperMaxLines: 3,
        border: OutlineInputBorder(),
      ),
    ),
    actions: [
      TextButton(
        onPressed: () => Navigator.pop(context),
        child: const Text('Cancel'),
      ),
      FilledButton(
        onPressed: () => Navigator.pop(context, _text.text),
        child: const Text('Save'),
      ),
    ],
  );
}
