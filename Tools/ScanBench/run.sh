#!/bin/sh
# Builds and runs ScanBench (DepthEngine on a Mac, no iPhone needed). Run from anywhere:
#   Tools/ScanBench/run.sh                        synthetic scenarios with ground truth
#   ONLY=face Tools/ScanBench/run.sh              one scenario
#   Tools/ScanBench/run.sh replay <capture dir>   a raw capture shared from the app (unzipped)
set -e
cd "$(dirname "$0")/../.."
E=DepthEngine
xcrun swiftc -O -o "${TMPDIR:-/tmp}/scanbench" Tools/ScanBench/*.swift \
    $E/Registration/*.swift $E/Fusion/*.swift $E/Export/*.swift $E/DepthEngineError.swift $E/ScanMode.swift
exec "${TMPDIR:-/tmp}/scanbench" "$@"
