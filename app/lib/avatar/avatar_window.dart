import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:material_ui/material_ui.dart';
import 'package:window_manager/window_manager.dart';

/// Turns the app's own window into a small, frameless, transparent, always-on-top
/// companion and back. Only desktop platforms have such a window; elsewhere the
/// avatar is shown full-screen inside the app.
class AvatarWindow {
  static const size = Size(260, 360);

  static bool get supported =>
      !kIsWeb && (Platform.isLinux || Platform.isMacOS || Platform.isWindows);

  Rect? _restore;
  Size? _minimum;

  Window? get _window => supported ? WindowManager.instance.getCurrent() : null;

  /// Shrinks the window to the companion. Returns an error message when the window
  /// could not be changed, in which case the avatar still shows inside it.
  String? enter() {
    final w = _window;
    if (w == null) {
      return supported ? 'No window to turn into a companion' : null;
    }
    try {
      _restore = w.bounds.toRect();
      _minimum = w.minimumSize.toSize();
      final bottomRight = _restore!.bottomRight;
      w.setFrameless(true);
      w.hasShadow = false;
      w.backgroundColor = const Color(0x00000000).toNative();
      w.isResizable = false;
      w.minimumSize = size.toNative();
      w.setSize(size.toNative(), false);
      // Keep the companion where the main window's bottom-right corner was.
      w.position = (bottomRight - Offset(size.width, size.height)).toNative();
      w.isAlwaysOnTop = true;
      w.isVisibleOnAllWorkspaces = true;
      return null;
    } catch (e) {
      return 'Could not float the avatar: $e';
    }
  }

  /// Restores the regular app window.
  String? exit() {
    final w = _window;
    if (w == null) return null;
    try {
      w.isAlwaysOnTop = false;
      w.isVisibleOnAllWorkspaces = false;
      w.titleBarStyle = TitleBarStyle.normal;
      w.setFrameless(false);
      w.hasShadow = true;
      w.backgroundColor = const Color(0xFF000000).toNative();
      w.isResizable = true;
      w.minimumSize = (_minimum ?? Size.zero).toNative();
      final r = _restore;
      if (r != null) w.bounds = r.toNative();
      w.focus();
      return null;
    } catch (e) {
      return 'Could not restore the window: $e';
    }
  }

  /// Lets the user move the frameless companion by dragging it.
  void startDragging() {
    try {
      _window?.startDragging();
    } catch (e) {
      debugPrint('avatar drag failed: $e');
    }
  }
}
