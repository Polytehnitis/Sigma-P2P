#!/usr/bin/env bash
# Copies the parts of whisper.cpp that Sigma builds (whisper, parakeet, ggml with its CPU and
# BLAS backends) from a checkout of https://github.com/ggml-org/whisper.cpp into
# crates/sigma-voice/whisper.cpp. Usage: vendor-whisper.sh <whisper.cpp checkout> <tag>
set -euo pipefail
src=$(cd "$1" && pwd)
tag=$2
dst=$(cd "$(dirname "$0")/.." && pwd)/whisper.cpp
git -C "$src" checkout -q "$tag"
commit=$(git -C "$src" rev-parse HEAD)
rm -rf "$dst"
mkdir -p "$dst/src" "$dst/ggml/src"
cp "$src"/{CMakeLists.txt,LICENSE} "$dst/"
cp -r "$src"/{cmake,include} "$dst/"
cp "$src"/src/{CMakeLists.txt,*.cpp,*.h} "$dst/src/"
cp -r "$src"/src/aneforge "$dst/src/"
cp "$src"/ggml/CMakeLists.txt "$dst/ggml/"
cp -r "$src"/ggml/{cmake,include} "$dst/ggml/"
cp "$src"/ggml/src/{CMakeLists.txt,*.c,*.cpp,*.h,*.in} "$dst/ggml/src/"
cp -r "$src"/ggml/src/{ggml-cpu,ggml-blas} "$dst/ggml/src/"
cat > "$dst/VENDORED.md" <<MD
# whisper.cpp, vendored

From https://github.com/ggml-org/whisper.cpp at \`$tag\` (\`$commit\`), MIT licensed (see LICENSE).
Only whisper, parakeet and ggml's CPU and BLAS backends; GPU backends, examples, bindings
and tests are left out. Update with \`crates/sigma-voice/scripts/vendor-whisper.sh <checkout> <tag>\`.
MD
du -sh "$dst"
