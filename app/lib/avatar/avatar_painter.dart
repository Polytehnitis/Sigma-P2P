import 'dart:math' as math;
import 'dart:ui' show lerpDouble;

import 'package:flutter/rendering.dart';

import 'avatar_mood.dart';

/// Every continuous parameter of the character's face and body. The widget eases
/// the current pose toward the mood's target pose each frame.
class AvatarPose {
  const AvatarPose({
    this.mouthCurve = 0.3,
    this.mouthOpen = 0,
    this.browRaise = 0,
    this.browWorry = 0,
    this.eyeOpen = 1,
    this.eyeSmile = 0,
    this.leftArm = 0.25,
    this.rightArm = 0.25,
    this.blush = 0,
    this.antennaDroop = 0,
  });

  /// -1 frown … 1 smile.
  final double mouthCurve;

  /// 0 closed … 1 wide open.
  final double mouthOpen;

  /// -1 low … 1 raised.
  final double browRaise;

  /// 0 flat … 1 inner corners lifted (worry).
  final double browWorry;

  /// 0 closed … 1 open.
  final double eyeOpen;

  /// 0 round eyes … 1 happy arcs.
  final double eyeSmile;

  /// Arm angles from hanging down (0) to straight up (pi), outward from the body.
  final double leftArm;
  final double rightArm;
  final double blush;
  final double antennaDroop;

  factory AvatarPose.of(Expression e, Gesture g) {
    final (
      mouthCurve,
      mouthOpen,
      browRaise,
      browWorry,
      eyeOpen,
      eyeSmile,
      blush,
      droop,
    ) = switch (e) {
      Expression.asleep => (0.1, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 1.0),
      Expression.idle => (0.35, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
      Expression.happy => (1.0, 0.55, 0.5, 0.0, 1.0, 1.0, 0.8, 0.0),
      Expression.sad => (-0.9, 0.0, -0.2, 1.0, 0.75, 0.0, 0.0, 0.9),
      Expression.surprised => (0.0, 0.9, 1.0, 0.2, 1.0, 0.0, 0.2, -0.3),
      Expression.hopeful => (0.7, 0.15, 0.4, 0.0, 1.0, 0.3, 0.4, -0.2),
      Expression.worried => (-0.5, 0.2, 0.3, 1.0, 1.0, 0.0, 0.0, 0.5),
      Expression.thinking => (-0.1, 0.0, 0.2, 0.3, 0.85, 0.0, 0.0, 0.2),
    };
    final (l, r) = switch (g) {
      Gesture.rest => (0.25, 0.25),
      Gesture.wave => (0.25, 2.5),
      Gesture.cheer => (2.6, 2.6),
      Gesture.droop => (0.05, 0.05),
      Gesture.startle => (1.9, 1.9),
      Gesture.point => (0.25, 1.6),
      Gesture.ponder => (0.25, 3.3),
      Gesture.shrug => (1.2, 1.2),
    };
    return AvatarPose(
      mouthCurve: mouthCurve,
      mouthOpen: mouthOpen,
      browRaise: browRaise,
      browWorry: browWorry,
      eyeOpen: eyeOpen,
      eyeSmile: eyeSmile,
      leftArm: l,
      rightArm: r,
      blush: blush,
      antennaDroop: droop,
    );
  }

  static AvatarPose lerp(AvatarPose a, AvatarPose b, double t) {
    double f(double x, double y) => lerpDouble(x, y, t)!;
    return AvatarPose(
      mouthCurve: f(a.mouthCurve, b.mouthCurve),
      mouthOpen: f(a.mouthOpen, b.mouthOpen),
      browRaise: f(a.browRaise, b.browRaise),
      browWorry: f(a.browWorry, b.browWorry),
      eyeOpen: f(a.eyeOpen, b.eyeOpen),
      eyeSmile: f(a.eyeSmile, b.eyeSmile),
      leftArm: f(a.leftArm, b.leftArm),
      rightArm: f(a.rightArm, b.rightArm),
      blush: f(a.blush, b.blush),
      antennaDroop: f(a.antennaDroop, b.antennaDroop),
    );
  }
}

/// Draws Sigma's character: a round, bean-shaped creature with a light-bulb antenna
/// and a Σ on its belly. Designed for this app; drawn entirely in code.
class AvatarPainter extends CustomPainter {
  AvatarPainter({
    required this.pose,
    required this.gaze,
    required this.glow,
    required this.bob,
    required this.blink,
  });

  final AvatarPose pose;
  final Offset gaze;
  final Color glow;

  /// Vertical offset of the breathing/bouncing motion, in logical pixels at 200 wide.
  final double bob;

  /// 0 eyes as posed … 1 fully shut (blinking).
  final double blink;

  static const _body = Color(0xFF4FB9B4);
  static const _bodyDark = Color(0xFF2F8F97);
  static const _belly = Color(0xFFD8F3EE);
  static const _ink = Color(0xFF203040);

  @override
  void paint(Canvas canvas, Size size) {
    // Designed on a 200 x 240 canvas; scale uniformly to fit.
    final s = math.min(size.width / 200, size.height / 240);
    canvas.save();
    canvas.translate((size.width - 200 * s) / 2, (size.height - 240 * s) / 2);
    canvas.scale(s);

    const cx = 100.0;
    final cy = 140.0 + bob;
    final stretch = 1 + bob * -0.004;

    // Ground shadow shrinks as the body rises.
    canvas.drawOval(
      Rect.fromCenter(
        center: const Offset(cx, 222),
        width: 96 - bob * 2,
        height: 12,
      ),
      Paint()..color = const Color(0x33000000),
    );

    // Feet.
    final feet = Paint()..color = _bodyDark;
    for (final dx in [-24.0, 24.0]) {
      canvas.drawOval(
        Rect.fromCenter(
          center: Offset(cx + dx, 212 + bob * 0.3),
          width: 30,
          height: 16,
        ),
        feet,
      );
    }

    _antenna(canvas, cx, cy - 66 * stretch);

    // Body.
    final body = Rect.fromCenter(
      center: Offset(cx, cy),
      width: 124,
      height: 138 * stretch,
    );
    canvas.drawOval(
      body,
      Paint()
        ..shader = const LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [_body, _bodyDark],
        ).createShader(body),
    );
    canvas.drawOval(
      Rect.fromCenter(center: Offset(cx, cy + 34), width: 70, height: 58),
      Paint()..color = _belly,
    );
    _sigma(canvas, Offset(cx, cy + 36));

    _arm(canvas, Offset(cx - 56, cy + 6), pose.leftArm, -1);
    _arm(canvas, Offset(cx + 56, cy + 6), pose.rightArm, 1);

    _eyes(canvas, cx, cy - 26);
    _mouth(canvas, Offset(cx, cy + 4));

    if (pose.blush > 0.01) {
      final p = Paint()
        ..color = Color.fromRGBO(255, 120, 140, 0.45 * pose.blush.clamp(0, 1));
      for (final dx in [-38.0, 38.0]) {
        canvas.drawOval(
          Rect.fromCenter(
            center: Offset(cx + dx, cy - 4),
            width: 16,
            height: 9,
          ),
          p,
        );
      }
    }
    canvas.restore();
  }

  void _antenna(Canvas canvas, double x, double top) {
    final droop = pose.antennaDroop;
    final tip = Offset(x + 16 + droop * 12, top - 34 + droop * 22);
    final stalk = Path()
      ..moveTo(x, top + 4)
      ..quadraticBezierTo(x - 6, top - 22, tip.dx, tip.dy);
    canvas.drawPath(
      stalk,
      Paint()
        ..color = _bodyDark
        ..style = PaintingStyle.stroke
        ..strokeWidth = 5
        ..strokeCap = StrokeCap.round,
    );
    canvas.drawCircle(
      tip,
      15,
      Paint()
        ..color = glow.withValues(alpha: 0.45)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 8),
    );
    canvas.drawCircle(tip, 8, Paint()..color = glow);
    canvas.drawCircle(
      tip.translate(-2.5, -2.5),
      2.5,
      Paint()..color = const Color(0xCCFFFFFF),
    );
  }

  void _sigma(Canvas canvas, Offset c) {
    final p = Path()
      ..moveTo(c.dx + 9, c.dy - 11)
      ..lineTo(c.dx - 9, c.dy - 11)
      ..lineTo(c.dx + 2, c.dy)
      ..lineTo(c.dx - 9, c.dy + 11)
      ..lineTo(c.dx + 9, c.dy + 11);
    canvas.drawPath(
      p,
      Paint()
        ..color = _bodyDark
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3.5
        ..strokeJoin = StrokeJoin.round
        ..strokeCap = StrokeCap.round,
    );
  }

  /// A stubby arm with a round hand; `side` is -1 for the left arm, 1 for the right.
  void _arm(Canvas canvas, Offset shoulder, double angle, double side) {
    const len = 40.0;
    final hand =
        shoulder + Offset(side * math.sin(angle) * len, math.cos(angle) * len);
    canvas.drawLine(
      shoulder,
      hand,
      Paint()
        ..color = _bodyDark
        ..strokeWidth = 15
        ..strokeCap = StrokeCap.round,
    );
    canvas.drawCircle(hand, 10, Paint()..color = _body);
    canvas.drawCircle(
      hand,
      10,
      Paint()
        ..color = _bodyDark
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );
  }

  void _eyes(Canvas canvas, double cx, double y) {
    final open = (pose.eyeOpen * (1 - blink)).clamp(0.0, 1.0);
    final smile = pose.eyeSmile.clamp(0.0, 1.0);
    final ink = Paint()
      ..color = _ink
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.5
      ..strokeCap = StrokeCap.round;
    for (final side in [-1.0, 1.0]) {
      final c = Offset(cx + side * 23, y);
      if (open < 0.15 || smile > 0.6) {
        // Closed (asleep or blinking) eyes curve down; happy eyes curve up.
        final up = smile > 0.6 && open >= 0.15;
        final arc = Path()
          ..moveTo(c.dx - 10, c.dy)
          ..quadraticBezierTo(c.dx, c.dy + (up ? -12 : 8), c.dx + 10, c.dy);
        canvas.drawPath(arc, ink);
      } else {
        final h = 26 * open;
        canvas.drawOval(
          Rect.fromCenter(center: c, width: 21, height: h),
          Paint()..color = const Color(0xFFFFFFFF),
        );
        final pupil = c + Offset(gaze.dx * 4.5, gaze.dy * 5 * open);
        canvas.save();
        canvas.clipPath(
          Path()..addOval(Rect.fromCenter(center: c, width: 21, height: h)),
        );
        canvas.drawCircle(pupil, 7, Paint()..color = _ink);
        canvas.drawCircle(
          pupil.translate(-2.5, -2.5),
          2.4,
          Paint()..color = const Color(0xFFFFFFFF),
        );
        canvas.restore();
      }
      // Brow: lifted, lowered and (for worry) tilted with the inner end up.
      final by = y - 21 - pose.browRaise * 5;
      final inner = Offset(c.dx - side * 9, by - pose.browWorry * 6);
      final outer = Offset(c.dx + side * 9, by + pose.browWorry * 2);
      canvas.drawLine(inner, outer, ink..strokeWidth = 3);
    }
  }

  void _mouth(Canvas canvas, Offset c) {
    final curve = pose.mouthCurve.clamp(-1.0, 1.0);
    final open = pose.mouthOpen.clamp(0.0, 1.0);
    final w = 13 + open * 3;
    if (open < 0.08) {
      final p = Path()
        ..moveTo(c.dx - w, c.dy)
        ..quadraticBezierTo(c.dx, c.dy + curve * 11, c.dx + w, c.dy);
      canvas.drawPath(
        p,
        Paint()
          ..color = _ink
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3.5
          ..strokeCap = StrokeCap.round,
      );
      return;
    }
    if (open > 0.6 && curve.abs() < 0.3) {
      // A round "oh!".
      canvas.drawOval(
        Rect.fromCenter(
          center: c.translate(0, 4),
          width: 13,
          height: 8 + open * 10,
        ),
        Paint()..color = _ink,
      );
      return;
    }
    // An open mouth: upper lip follows the curve, lower lip drops with `open`.
    final top = c.dy - (curve < 0 ? -curve * 4 : 0);
    final p = Path()
      ..moveTo(c.dx - w, top)
      ..quadraticBezierTo(c.dx, top + curve * 5, c.dx + w, top)
      ..quadraticBezierTo(c.dx, top + 4 + open * 18 + curve * 6, c.dx - w, top)
      ..close();
    canvas.drawPath(p, Paint()..color = _ink);
    canvas.save();
    canvas.clipPath(p);
    canvas.drawOval(
      Rect.fromCenter(
        center: Offset(c.dx, top + 4 + open * 12),
        width: w * 1.1,
        height: 10,
      ),
      Paint()..color = const Color(0xFFFF7D8C),
    );
    canvas.restore();
  }

  @override
  bool shouldRepaint(AvatarPainter old) =>
      old.pose != pose ||
      old.gaze != gaze ||
      old.glow != glow ||
      old.bob != bob ||
      old.blink != blink;
}
