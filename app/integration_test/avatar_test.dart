import 'package:app/avatar/avatar_view.dart';
import 'package:app/main.dart';
import 'package:app/src/rust/frb_generated.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:material_ui/material_ui.dart';
import 'package:window_manager/window_manager.dart';

Future<void> frames(WidgetTester tester, [int n = 10]) async {
  for (var i = 0; i < n; i++) {
    await tester.pump(const Duration(milliseconds: 100));
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(() async => await RustLib.init());

  testWidgets('the window becomes a floating avatar and back', (tester) async {
    await tester.pumpWidget(const SigmaApp());
    await frames(tester);
    final window = WindowManager.instance.getCurrent()!;
    final before = window.size.toSize();

    await tester.tap(find.byTooltip('Put Sigma on the desktop'));
    await frames(tester);
    expect(find.byType(SigmaAvatar), findsOneWidget);
    expect(find.textContaining('Zzz'), findsOneWidget);
    // Always-on-top is a window manager hint, which bare Xvfb in CI has none to honour.
    expect(window.isFrameless, isTrue);
    await frames(tester, 20);
    expect(window.size.toSize(), const Size(260, 360));

    // Double-click the character to open the full app again.
    final body = find.byKey(const Key('avatar-body'));
    await tester.tap(body);
    await tester.pump(const Duration(milliseconds: 50));
    await tester.tap(body);
    await frames(tester, 20);
    expect(find.byType(SigmaAvatar), findsNothing);
    expect(find.text('Sigma cognitive agent'), findsOneWidget);
    expect(window.isFrameless, isFalse);
    expect(window.size.toSize(), before);
  });
}
