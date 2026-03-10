#!/usr/bin/env bash
# Build and create Shader Playground.app macOS bundle
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
APP_NAME="Shader Playground"
BUNDLE_ID="com.chadfurman.shader-playground"
BUNDLE_DIR="$PROJECT_DIR/target/${APP_NAME}.app"

echo "Building release binary..."
cargo build --release --manifest-path "$PROJECT_DIR/Cargo.toml"

echo "Creating app bundle..."
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR/Contents/MacOS"
mkdir -p "$BUNDLE_DIR/Contents/Resources"

# Copy binary
cp "$PROJECT_DIR/target/release/shader-playground" "$BUNDLE_DIR/Contents/MacOS/"

# Copy runtime resources (shaders, config, weights)
for f in playground.wgsl flame_compute.wgsl accumulation.wgsl histogram_cdf.wgsl weights.json params.json; do
    if [ -f "$PROJECT_DIR/$f" ]; then
        cp "$PROJECT_DIR/$f" "$BUNDLE_DIR/Contents/Resources/"
    fi
done

# Copy genomes directory if it exists
if [ -d "$PROJECT_DIR/genomes" ]; then
    cp -r "$PROJECT_DIR/genomes" "$BUNDLE_DIR/Contents/Resources/"
fi

# Write Info.plist
cat > "$BUNDLE_DIR/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleVersion</key>
    <string>0.4.1</string>
    <key>CFBundleShortVersionString</key>
    <string>0.4.1</string>
    <key>CFBundleExecutable</key>
    <string>shader-playground</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Shader Playground uses audio input for real-time visualization.</string>
</dict>
</plist>
PLIST

echo ""
echo "Bundle created: $BUNDLE_DIR"
echo ""
echo "To install: cp -r \"$BUNDLE_DIR\" /Applications/"
echo "To run:     open \"$BUNDLE_DIR\""
