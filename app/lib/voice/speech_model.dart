import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:archive/archive_io.dart';
import 'package:crypto/crypto.dart';

/// The on-device speech recognition model: a small streaming English zipformer (int8) for
/// sherpa-onnx. It is downloaded once, checked against pinned SHA-256 digests, and
/// then everything runs offline: audio never leaves the device.
class SpeechModelSpec {
  const SpeechModelSpec({
    required this.name,
    required this.url,
    required this.archiveSha256,
    required this.archiveBytes,
    required this.files,
  });

  final String name;
  final String url;
  final String archiveSha256;
  final int archiveBytes;

  /// Files kept from the archive, by role (encoder, decoder, joiner, tokens), with digests.
  final Map<String, (String file, String sha256)> files;

  static const english = SpeechModelSpec(
    name: 'sherpa-onnx-streaming-zipformer-en-20M-2023-02-17',
    url:
        'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/'
        'sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2',
    archiveSha256:
        '9c559283e8498d3fe95913c79ca1cb454bb26281ac2b102b41306c7d752765d9',
    archiveBytes: 127887156,
    files: {
      'encoder': (
        'encoder-epoch-99-avg-1.int8.onnx',
        '3810755ce7c3ab26b42a8bcf39d191308fa27fb0f53358823ba46141d03b7eb3',
      ),
      'decoder': (
        'decoder-epoch-99-avg-1.int8.onnx',
        '21e2a2acd961b3ac72f55be2f10f1a285e1b0b0ba010d7c0b6eab141411b163c',
      ),
      'joiner': (
        'joiner-epoch-99-avg-1.int8.onnx',
        'e085d73b593cf9b0707f370dbd656d58327d3fe36d80d849202ef81df02cb01e',
      ),
      'tokens': (
        'tokens.txt',
        '49e3c2646595fd907228b3c6787069658f67b17377c60aeb8619c4551b2316fb',
      ),
    },
  );

  /// Megabytes downloaded once (the archive), for the consent prompt.
  int get downloadMegabytes => (archiveBytes / 1e6).round();
}

/// Where the model lives on this device and how it gets there.
class SpeechModelStore {
  SpeechModelStore(this.root, {this.spec = SpeechModelSpec.english});

  /// Usually `<documents>/sigma/speech`.
  final Directory root;
  final SpeechModelSpec spec;

  Directory get dir => Directory('${root.path}/${spec.name}');

  String path(String role) => '${dir.path}/${spec.files[role]!.$1}';

  /// The model is complete (a marker is written only after every digest matched).
  bool get installed => File('${dir.path}/.verified').existsSync();

  /// Downloads, verifies and unpacks the model. `onProgress` gets values in [0, 1].
  Future<void> install({void Function(double progress)? onProgress}) async {
    await root.create(recursive: true);
    final archive = File('${root.path}/${spec.name}.tar.bz2.part');
    try {
      await _download(archive, onProgress);
      final digest = await sha256.bind(archive.openRead()).first;
      if (digest.toString() != spec.archiveSha256) {
        throw const SpeechModelException(
          'The downloaded speech model does not match its pinned checksum.',
        );
      }
      final (from, to, files) = (archive.path, dir.path, spec);
      await Isolate.run(() => _unpack(from, to, files));
      File('${dir.path}/.verified').writeAsStringSync(spec.archiveSha256);
    } finally {
      if (archive.existsSync()) archive.deleteSync();
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
      final total = response.contentLength > 0
          ? response.contentLength
          : spec.archiveBytes;
      final sink = into.openWrite();
      var received = 0;
      try {
        await for (final chunk in response) {
          received += chunk.length;
          if (received > spec.archiveBytes * 2) {
            throw const SpeechModelException('Speech model is too large.');
          }
          sink.add(chunk);
          onProgress?.call((received / total).clamp(0, 1).toDouble());
        }
      } finally {
        await sink.close();
      }
    } finally {
      client.close();
    }
  }
}

/// Runs in a background isolate: bzip2 -> tar -> only the files the spec names, each
/// checked against its digest.
Future<void> _unpack(
  String archivePath,
  String outDir,
  SpeechModelSpec spec,
) async {
  final tarPath = '$archivePath.tar';
  final input = InputFileStream(archivePath);
  final output = OutputFileStream(tarPath);
  try {
    BZip2Decoder().decodeStream(input, output);
  } finally {
    await input.close();
    await output.close();
  }
  final tarInput = InputFileStream(tarPath);
  try {
    final tar = TarDecoder().decodeStream(tarInput);
    final out = Directory(outDir)..createSync(recursive: true);
    final wanted = {for (final f in spec.files.values) f.$1: f.$2};
    for (final entry in tar.files) {
      // Only exact, known file names: nothing in the archive picks a path.
      final name = entry.name.split('/').last;
      final expected = wanted.remove(name);
      if (expected == null || !entry.isFile) continue;
      final bytes = entry.readBytes() ?? const <int>[];
      if (sha256.convert(bytes).toString() != expected) {
        throw SpeechModelException('Speech model file $name is corrupt.');
      }
      File('${out.path}/$name').writeAsBytesSync(bytes);
    }
    if (wanted.isNotEmpty) {
      throw SpeechModelException(
        'Speech model is missing ${wanted.keys.join(', ')}.',
      );
    }
  } finally {
    await tarInput.close();
    File(tarPath).deleteSync();
  }
}

class SpeechModelException implements Exception {
  const SpeechModelException(this.message);
  final String message;

  @override
  String toString() => message;
}
