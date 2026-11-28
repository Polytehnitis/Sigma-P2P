import 'dart:async';
import 'dart:io';

import 'package:crypto/crypto.dart';

/// The on-device speech recognition model: an English Whisper model for whisper.cpp (ggml,
/// quantized). It is downloaded once, checked against a pinned SHA-256 digest, and then
/// everything runs offline: audio never leaves the device.
class SpeechModelSpec {
  const SpeechModelSpec({
    required this.file,
    required this.url,
    required this.sha256,
    required this.bytes,
  });

  /// The model's file name, also its name on disk.
  final String file;
  final String url;
  final String sha256;
  final int bytes;

  static const _repo =
      'https://huggingface.co/ggerganov/whisper.cpp/resolve/main';

  /// Whisper small (English), 5-bit: much better than base at short commands and accents,
  /// and fast enough on a desktop CPU.
  static const desktop = SpeechModelSpec(
    file: 'ggml-small.en-q5_1.bin',
    url: '$_repo/ggml-small.en-q5_1.bin',
    sha256: 'bfdff4894dcb76bbf647d56263ea2a96645423f1669176f4844a1bf8e478ad30',
    bytes: 190098681,
  );

  /// Whisper base (English), 5-bit: a third of the size and CPU of small, for phones.
  static const mobile = SpeechModelSpec(
    file: 'ggml-base.en-q5_1.bin',
    url: '$_repo/ggml-base.en-q5_1.bin',
    sha256: '4baf70dd0d7c4247ba2b81fafd9c01005ac77c2f9ef064e00dcf195d0e2fdd2f',
    bytes: 59721011,
  );

  /// The model for this device.
  static SpeechModelSpec get forThisDevice =>
      Platform.isAndroid || Platform.isIOS ? mobile : desktop;

  /// Megabytes downloaded once, for the consent prompt.
  int get downloadMegabytes => (bytes / 1e6).round();
}

/// Where the model lives on this device and how it gets there.
class SpeechModelStore {
  SpeechModelStore(this.root, {SpeechModelSpec? spec})
    : spec = spec ?? SpeechModelSpec.forThisDevice;

  /// Usually `<documents>/sigma/speech`.
  final Directory root;
  final SpeechModelSpec spec;

  /// The model file. It only appears once its digest matched.
  File get file => File('${root.path}/${spec.file}');

  bool get installed => file.existsSync();

  /// Downloads and verifies the model. `onProgress` gets values in [0, 1].
  Future<void> install({void Function(double progress)? onProgress}) async {
    await root.create(recursive: true);
    final part = File('${file.path}.part');
    try {
      await _download(part, onProgress);
      final digest = await sha256.bind(part.openRead()).first;
      if (digest.toString() != spec.sha256) {
        throw const SpeechModelException(
          'The downloaded speech model does not match its pinned checksum.',
        );
      }
      await part.rename(file.path);
      _forgetOldModels();
    } finally {
      if (part.existsSync()) part.deleteSync();
    }
  }

  /// Earlier versions used a sherpa-onnx model; it is no longer needed.
  void _forgetOldModels() {
    for (final e in root.listSync()) {
      final name = e.path.split(Platform.pathSeparator).last.split('/').last;
      if (e is Directory && name.startsWith('sherpa-onnx-')) {
        try {
          e.deleteSync(recursive: true);
        } on FileSystemException {
          // Best effort: it only takes space.
        }
      }
    }
  }

  Future<void> _download(
    File into,
    void Function(double progress)? onProgress,
  ) async {
    final client = HttpClient();
    try {
      final request = await client.getUrl(Uri.parse(spec.url));
      final response = await request.close();
      if (response.statusCode != HttpStatus.ok) {
        throw SpeechModelException(
          'Speech model download failed (HTTP ${response.statusCode}).',
        );
      }
      final sink = into.openWrite();
      var received = 0;
      try {
        await for (final chunk in response) {
          received += chunk.length;
          if (received > spec.bytes) {
            throw const SpeechModelException('Speech model is too large.');
          }
          sink.add(chunk);
          onProgress?.call((received / spec.bytes).clamp(0, 1).toDouble());
        }
      } finally {
        await sink.close();
      }
    } finally {
      client.close();
    }
  }
}

class SpeechModelException implements Exception {
  const SpeechModelException(this.message);
  final String message;

  @override
  String toString() => message;
}
