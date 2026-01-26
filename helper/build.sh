#!/bin/bash
# Build and sign the ANE helper binary
# Usage: ./build.sh [signing-identity]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Default signing identity (ad-hoc if not specified)
IDENTITY="${1:--}"

echo "=== Building ANE Helper ==="

# Compile
clang -framework Foundation \
      -F/System/Library/PrivateFrameworks \
      -framework AppleNeuralEngine \
      -o ane_helper \
      ane_helper.m

echo "Compiled ane_helper"

# Sign with entitlements
echo "Signing with identity: $IDENTITY"
codesign -f -s "$IDENTITY" \
         --entitlements ane_helper.entitlements \
         ane_helper

echo "Signed ane_helper"

# Verify
echo ""
echo "=== Verification ==="
codesign -d --entitlements - ane_helper 2>&1 | head -20

echo ""
echo "=== Testing ==="
echo '{"cmd": "status"}' | ./ane_helper || echo "Helper exited with code: $?"
