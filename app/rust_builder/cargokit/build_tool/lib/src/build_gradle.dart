/// This is copied from Cargokit (which is the official way to use it currently)
/// Details: https://fzyzcjy.github.io/flutter_rust_bridge/manual/integrate/builtin

import 'dart:io';

import 'package:logging/logging.dart';
import 'package:path/path.dart' as path;

import 'artifacts_provider.dart';
import 'builder.dart';
import 'environment.dart';
import 'options.dart';
import 'target.dart';

final log = Logger('build_gradle');

class BuildGradle {
  BuildGradle({required this.userOptions});

  final CargokitUserOptions userOptions;

  Future<void> build() async {
    final targets = Environment.targetPlatforms
        .map((arch) {
          final target = Target.forFlutterName(arch);
          if (target == null) {
            throw Exception(
              "Unknown darwin target or platform: $arch, ${Environment.darwinPlatformName}",
            );
          }
          return target;
        })
        .toSet()
        .toList();

    final environment = BuildEnvironment.fromEnvironment(isAndroid: true);
    final provider = ArtifactProvider(
      environment: environment,
      userOptions: userOptions,
    );
    final artifacts = await provider.getArtifacts(targets);

    for (final target in targets) {
      final libs = artifacts[target]!;
      final outputDir = path.join(Environment.outputDir, target.android!);
      Directory(outputDir).createSync(recursive: true);

      for (final lib in libs) {
        if (lib.type == AritifactType.dylib) {
          File(lib.path).copySync(path.join(outputDir, lib.finalFileName));
        }
      }
      _copyCxxRuntime(target, outputDir);
    }
  }

  /// Sigma: whisper.cpp (in the Rust library) links the NDK's shared C++ runtime, which
  /// Android does not provide, so it ships next to the library.
  void _copyCxxRuntime(Target target, String outputDir) {
    final triple = switch (target.android) {
      'arm64-v8a' => 'aarch64-linux-android',
      'armeabi-v7a' => 'arm-linux-androideabi',
      'x86_64' => 'x86_64-linux-android',
      'x86' => 'i686-linux-android',
      _ => null,
    };
    if (triple == null) return;
    final prebuilt = Directory(path.join(Environment.sdkPath, 'ndk',
        Environment.ndkVersion, 'toolchains', 'llvm', 'prebuilt'));
    if (!prebuilt.existsSync()) {
      throw Exception('Android NDK not found at ${prebuilt.path}');
    }
    for (final host in prebuilt.listSync().whereType<Directory>()) {
      final lib = File(path.join(
          host.path, 'sysroot', 'usr', 'lib', triple, 'libc++_shared.so'));
      if (lib.existsSync()) {
        lib.copySync(path.join(outputDir, 'libc++_shared.so'));
        return;
      }
    }
    throw Exception('libc++_shared.so for $triple not found in ${prebuilt.path}');
  }
}
