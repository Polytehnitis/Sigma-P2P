import 'dart:math' as math;

import 'package:flutter/scheduler.dart';
import 'package:material_ui/material_ui.dart';

import 'avatar_mood.dart';
import 'avatar_painter.dart';

/// Sigma's animated character with its speech bubble and quick controls.
///
/// Click: think (run a cognitive step). Double-click: open the full app.
/// Drag: move the window. Right-click or long-press: menu. The mic button listens.
class SigmaAvatar extends StatefulWidget {
  const SigmaAvatar({
    super.key,
    required this.mood,
    this.note,
    this.busy = false,
    this.onThink,
    this.onOpen,
    this.onFeedback,
    this.onTeach,
    this.onDragStart,
    this.autoThink,
    this.onAutoThinkChanged,
    this.alwaysShowControls = false,
    this.onListen,
    this.listening = false,
  });

  final AvatarMood mood;

  /// A short message that replaces the mood's bubble, such as feedback or an error.
  final String? note;
  final bool busy;
  final VoidCallback? onThink;
  final VoidCallback? onOpen;
  final ValueChanged<double>? onFeedback;
  final VoidCallback? onTeach;
  final VoidCallback? onDragStart;
  final bool? autoThink;
  final ValueChanged<bool>? onAutoThinkChanged;

  /// Shows the rating buttons without hovering (touch screens, tests).
  final bool alwaysShowControls;

  /// Starts (or stops) listening for one spoken utterance.
  final VoidCallback? onListen;

  /// Sigma is listening: the mic button lights up and the character looks at the user.
  final bool listening;

  @override
  State<SigmaAvatar> createState() => _SigmaAvatarState();
}

class _SigmaAvatarState extends State<SigmaAvatar>
    with SingleTickerProviderStateMixin {
  late final Ticker _ticker;
  Duration _last = Duration.zero;
  double _t = 0;
  late AvatarPose _pose = AvatarPose.of(
    widget.mood.expression,
    widget.mood.gesture,
  );
  Offset _gaze = Offset.zero;
  bool _hover = false;

  @override
  void initState() {
    super.initState();
    _ticker = createTicker(_tick)..start();
  }

  void _tick(Duration now) {
    final dt = (now - _last).inMicroseconds / 1e6;
    _last = now;
    final m = widget.mood;
    var target = AvatarPose.of(m.expression, m.gesture);
    // Living gestures on top of the target pose.
    final wobble = switch (m.gesture) {
      Gesture.wave => (0.0, 0.45 * math.sin(_t * 7)),
      Gesture.cheer => (0.25 * math.sin(_t * 9), 0.25 * math.sin(_t * 9 + 1)),
      Gesture.ponder => (0.0, 0.12 * math.sin(_t * 5)),
      Gesture.shrug => (0.15 * math.sin(_t * 2), 0.15 * math.sin(_t * 2)),
      _ => (0.0, 0.0),
    };
    target = AvatarPose(
      mouthCurve: target.mouthCurve,
      mouthOpen: target.mouthOpen,
      browRaise: target.browRaise,
      browWorry: target.browWorry,
      eyeOpen: target.eyeOpen,
      eyeSmile: target.eyeSmile,
      leftArm: target.leftArm + wobble.$1,
      rightArm: target.rightArm + wobble.$2,
      blush: target.blush,
      antennaDroop: target.antennaDroop,
    );
    final ease = 1 - math.exp(-dt.clamp(0, 0.1) * 8);
    setState(() {
      _t += dt.clamp(0, 0.1);
      _pose = AvatarPose.lerp(_pose, target, ease);
      _gaze = Offset.lerp(
        _gaze,
        widget.listening ? Offset.zero : m.gaze,
        ease,
      )!;
    });
  }

  @override
  void dispose() {
    _ticker.dispose();
    super.dispose();
  }

  double get _bob {
    final e = widget.mood.energy;
    final speed = 1.6 + e * 4;
    final height = 1.5 + e * 7;
    final b = math.sin(_t * speed);
    // Energetic moods hop (only upward), calm moods breathe (both ways).
    return e > 0.6 ? -b.abs() * height : b * height * 0.5;
  }

  double get _blink {
    if (widget.mood.expression == Expression.asleep) return 0;
    final phase = _t % 3.7;
    return phase < 0.14 ? math.sin(phase / 0.14 * math.pi) : 0;
  }

  Future<void> _menu(Offset at) async {
    final m = widget.mood;
    final choice = await showMenu<String>(
      context: context,
      position: RelativeRect.fromRect(
        at & Size.zero,
        Offset.zero & MediaQuery.sizeOf(context),
      ),
      items: [
        if (widget.onThink != null)
          const PopupMenuItem(value: 'think', child: Text('Think now')),
        if (widget.onListen != null)
          PopupMenuItem(
            value: 'listen',
            child: Text(widget.listening ? 'Stop listening' : 'Listen'),
          ),
        if (m.canRate && widget.onFeedback != null) ...[
          const PopupMenuItem(value: 'up', child: Text('Good choice')),
          const PopupMenuItem(value: 'down', child: Text('Bad choice')),
        ],
        if (widget.onTeach != null)
          const PopupMenuItem(value: 'teach', child: Text('Teach a skill…')),
        if (widget.autoThink != null)
          CheckedPopupMenuItem(
            value: 'auto',
            checked: widget.autoThink!,
            child: const Text('Think on my own'),
          ),
        if (widget.onOpen != null)
          const PopupMenuItem(value: 'open', child: Text('Open Sigma')),
      ],
    );
    switch (choice) {
      case 'think':
        widget.onThink?.call();
      case 'listen':
        widget.onListen?.call();
      case 'up':
        widget.onFeedback?.call(1);
      case 'down':
        widget.onFeedback?.call(-1);
      case 'teach':
        widget.onTeach?.call();
      case 'auto':
        widget.onAutoThinkChanged?.call(!widget.autoThink!);
      case 'open':
        widget.onOpen?.call();
    }
  }

  @override
  Widget build(BuildContext context) {
    final m = widget.mood;
    final bubble = widget.note ?? m.bubble;
    final showRate =
        m.canRate &&
        widget.onFeedback != null &&
        (_hover || widget.alwaysShowControls);
    return MouseRegion(
      onEnter: (_) => setState(() => _hover = true),
      onExit: (_) => setState(() => _hover = false),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            height: 84,
            child: Align(
              alignment: Alignment.bottomCenter,
              child: bubble == null
                  ? null
                  : _SpeechBubble(
                      key: const Key('avatar-bubble'),
                      text: bubble,
                    ),
            ),
          ),
          GestureDetector(
            key: const Key('avatar-body'),
            behavior: HitTestBehavior.opaque,
            onTap: widget.busy ? null : widget.onThink,
            onDoubleTap: widget.onOpen,
            onPanStart: widget.onDragStart == null
                ? null
                : (_) => widget.onDragStart!(),
            onSecondaryTapUp: (d) => _menu(d.globalPosition),
            onLongPressStart: (d) => _menu(d.globalPosition),
            child: Semantics(
              label: 'Sigma avatar, feeling ${m.expression.name}',
              child: CustomPaint(
                size: const Size(180, 216),
                painter: AvatarPainter(
                  pose: _pose,
                  gaze: _gaze,
                  glow: m.glow,
                  bob: _bob,
                  blink: _blink,
                ),
              ),
            ),
          ),
          SizedBox(
            height: 40,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              spacing: 4,
              children: [
                if (widget.busy)
                  const SizedBox.square(
                    dimension: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
                if (m.focus != null && !showRate && !m.needsTeaching)
                  Flexible(
                    child: _Pill(
                      icon: Icons.center_focus_strong,
                      text: humanize(m.focus!),
                    ),
                  ),
                if (showRate) ...[
                  _RoundButton(
                    key: const Key('avatar-thumb-up'),
                    icon: Icons.thumb_up,
                    tooltip: 'Good choice',
                    onPressed: () => widget.onFeedback!(1),
                  ),
                  _RoundButton(
                    key: const Key('avatar-thumb-down'),
                    icon: Icons.thumb_down,
                    tooltip: 'Bad choice',
                    onPressed: () => widget.onFeedback!(-1),
                  ),
                ],
                if (widget.onListen != null &&
                    (widget.listening || _hover || widget.alwaysShowControls))
                  _RoundButton(
                    key: const Key('avatar-listen'),
                    icon: widget.listening ? Icons.mic : Icons.mic_none,
                    tooltip: widget.listening ? 'Stop listening' : 'Talk to me',
                    onPressed: widget.onListen!,
                  ),
                if (m.needsTeaching && widget.onTeach != null)
                  _RoundButton(
                    key: const Key('avatar-teach'),
                    icon: Icons.school,
                    tooltip: 'Teach me',
                    onPressed: widget.onTeach!,
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _SpeechBubble extends StatelessWidget {
  const _SpeechBubble({super.key, required this.text});
  final String text;

  @override
  Widget build(BuildContext context) => CustomPaint(
    painter: _BubblePainter(),
    child: Padding(
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 16),
      child: Text(
        text,
        maxLines: 3,
        overflow: TextOverflow.ellipsis,
        textAlign: TextAlign.center,
        style: const TextStyle(
          color: Color(0xFF203040),
          fontSize: 13,
          height: 1.25,
        ),
      ),
    ),
  );
}

class _BubblePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final r = RRect.fromRectAndRadius(
      Rect.fromLTWH(0, 0, size.width, size.height - 8),
      const Radius.circular(14),
    );
    final tail = Path()
      ..moveTo(size.width / 2 - 8, size.height - 9)
      ..lineTo(size.width / 2, size.height)
      ..lineTo(size.width / 2 + 8, size.height - 9)
      ..close();
    final path = Path()
      ..addRRect(r)
      ..addPath(tail, Offset.zero);
    canvas.drawShadow(path, const Color(0xFF000000), 3, false);
    canvas.drawPath(path, Paint()..color = const Color(0xFFFFFFFF));
  }

  @override
  bool shouldRepaint(_BubblePainter oldDelegate) => false;
}

class _Pill extends StatelessWidget {
  const _Pill({required this.icon, required this.text});
  final IconData icon;
  final String text;

  @override
  Widget build(BuildContext context) => DecoratedBox(
    decoration: BoxDecoration(
      color: const Color(0xE6FFFFFF),
      borderRadius: BorderRadius.circular(12),
    ),
    child: Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        spacing: 4,
        children: [
          Icon(icon, size: 14, color: const Color(0xFF2F8F97)),
          Flexible(
            child: Text(
              text,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(fontSize: 12, color: Color(0xFF203040)),
            ),
          ),
        ],
      ),
    ),
  );
}

class _RoundButton extends StatelessWidget {
  const _RoundButton({
    super.key,
    required this.icon,
    required this.tooltip,
    required this.onPressed,
  });
  final IconData icon;
  final String tooltip;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) => IconButton.filledTonal(
    tooltip: tooltip,
    iconSize: 18,
    visualDensity: VisualDensity.compact,
    onPressed: onPressed,
    icon: Icon(icon),
  );
}
