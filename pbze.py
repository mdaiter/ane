# PBZE format decoder for Apple's compressed espresso.net files
"""
PBZE Format (Protobuf-Zipped-Espresso?)
=======================================

Binary format used for some espresso.net model files.

Structure:
  - Bytes 0-3:   b'pbze' magic
  - Bytes 4-7:   Version/flags (usually 0x00000000)
  - Bytes 8-15:  Unknown (possibly header size as u64 LE)
  - Bytes 16-19: Uncompressed size (big endian u32!)
  - Bytes 20-23: Unknown
  - Bytes 24-27: Padding/unknown  
  - Bytes 28+:   LZFSE compressed data (starts with b'bvx2' marker)

The compressed data is standard LZFSE which can be decoded using
Apple's libcompression.dylib with COMPRESSION_LZFSE algorithm.

LZFSE Block Markers:
  - bvx2: Compressed block type 2
  - bvx1: Compressed block type 1
  - bvx-: Raw/uncompressed block
  - bvxn: Uncompressed block

After decompression, the data is standard espresso.net JSON.
"""
from __future__ import annotations
import ctypes
import struct
import json
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass


# LZFSE compression constant
COMPRESSION_LZFSE = 0x801

# Lazy-loaded compression library
_compression: Optional[ctypes.CDLL] = None


def _load_compression() -> ctypes.CDLL:
  """Load Apple's libcompression.dylib."""
  global _compression
  if _compression is None:
    _compression = ctypes.CDLL("/usr/lib/libcompression.dylib")
  return _compression


def _decompress_lzfse(data: bytes, uncompressed_size: int) -> bytes:
  """Decompress LZFSE data using Apple's compression library."""
  lib = _load_compression()
  
  compression_decode_buffer = lib.compression_decode_buffer
  compression_decode_buffer.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_int
  ]
  compression_decode_buffer.restype = ctypes.c_size_t
  
  # Allocate output buffer with some extra space
  dst = ctypes.create_string_buffer(uncompressed_size + 1024)
  src = ctypes.create_string_buffer(data)
  
  result = compression_decode_buffer(
    dst, uncompressed_size + 1024,
    src, len(data),
    None, COMPRESSION_LZFSE
  )
  
  if result == 0:
    raise ValueError("LZFSE decompression failed")
  
  return dst.raw[:result]


@dataclass
class PBZEHeader:
  """Parsed PBZE header."""
  magic: bytes
  version: int
  field1: int  # Unknown purpose
  uncompressed_size: int
  field3: int  # Unknown purpose
  compressed_offset: int = 28  # Where compressed data starts


def parse_pbze_header(data: bytes) -> PBZEHeader:
  """Parse PBZE file header."""
  if len(data) < 32:
    raise ValueError("Data too short for PBZE header")
  
  magic = data[0:4]
  if magic != b'pbze':
    raise ValueError(f"Invalid PBZE magic: {magic}")
  
  version = struct.unpack('<I', data[4:8])[0]
  field1 = struct.unpack('<Q', data[8:16])[0]
  # Uncompressed size is big-endian u32 at offset 16
  uncompressed_size = struct.unpack('>I', data[16:20])[0]
  field3 = struct.unpack('<I', data[20:24])[0]
  
  return PBZEHeader(
    magic=magic,
    version=version,
    field1=field1,
    uncompressed_size=uncompressed_size,
    field3=field3,
  )


def is_pbze(data: bytes) -> bool:
  """Check if data is PBZE format."""
  return len(data) >= 4 and data[:4] == b'pbze'


def is_pbze_file(path: str | Path) -> bool:
  """Check if file is PBZE format."""
  with open(path, 'rb') as f:
    magic = f.read(4)
  return magic == b'pbze'


def decode_pbze(data: bytes) -> bytes:
  """Decode PBZE compressed data to raw JSON bytes."""
  header = parse_pbze_header(data)
  
  # Verify bvx2 marker
  if data[28:32] != b'bvx2':
    raise ValueError(f"Expected bvx2 marker at offset 28, got {data[28:32]}")
  
  compressed = data[header.compressed_offset:]
  return _decompress_lzfse(compressed, header.uncompressed_size)


def decode_pbze_file(path: str | Path) -> bytes:
  """Decode PBZE file to raw JSON bytes."""
  with open(path, 'rb') as f:
    data = f.read()
  return decode_pbze(data)


def load_pbze_json(path: str | Path) -> dict[str, Any]:
  """Load PBZE file and parse as JSON."""
  raw = decode_pbze_file(path)
  return json.loads(raw.decode('utf-8'))


def decode_espresso_net(path: str | Path) -> dict[str, Any]:
  """Load espresso.net file (handles both JSON and PBZE formats)."""
  path = Path(path)
  
  with open(path, 'rb') as f:
    header = f.read(4)
  
  if header == b'pbze':
    return load_pbze_json(path)
  else:
    # Assume JSON
    with open(path) as f:
      return json.load(f)


# Statistics about PBZE compression
def get_pbze_stats(path: str | Path) -> dict[str, Any]:
  """Get compression statistics for a PBZE file."""
  with open(path, 'rb') as f:
    data = f.read()
  
  header = parse_pbze_header(data)
  compressed_size = len(data) - header.compressed_offset
  
  return {
    "file_size": len(data),
    "compressed_size": compressed_size,
    "uncompressed_size": header.uncompressed_size,
    "compression_ratio": header.uncompressed_size / compressed_size if compressed_size > 0 else 0,
    "header_version": header.version,
  }


if __name__ == "__main__":
  import sys
  from pathlib import Path
  
  print("=== PBZE Format Decoder ===\n")
  
  # Find a PBZE file to test
  test_file = Path("/System/Library/ImagingNetworks/hairnet-v1.macOS.espresso.net")
  
  if not test_file.exists():
    print("Test file not found, searching for PBZE files...")
    import subprocess
    result = subprocess.run(
      ["find", "/System/Library", "-name", "*.espresso.net"],
      capture_output=True, text=True, timeout=60
    )
    for line in result.stdout.strip().split("\n"):
      if line and is_pbze_file(line):
        test_file = Path(line)
        break
  
  if not test_file.exists():
    print("No PBZE files found")
    sys.exit(1)
  
  print(f"Testing with: {test_file}")
  
  # Get stats
  stats = get_pbze_stats(test_file)
  print(f"\nCompression stats:")
  print(f"  File size: {stats['file_size']} bytes")
  print(f"  Compressed: {stats['compressed_size']} bytes")
  print(f"  Uncompressed: {stats['uncompressed_size']} bytes")
  print(f"  Ratio: {stats['compression_ratio']:.2f}x")
  
  # Decode
  print("\nDecoding...")
  data = load_pbze_json(test_file)
  
  print(f"\nDecoded JSON:")
  print(f"  Format version: {data.get('format_version')}")
  print(f"  Storage: {data.get('storage')}")
  print(f"  Layer count: {len(data.get('layers', []))}")
  
  # Show layer types
  layer_types: dict[str, int] = {}
  for layer in data.get('layers', []):
    lt = layer.get('type', 'unknown')
    layer_types[lt] = layer_types.get(lt, 0) + 1
  
  print(f"  Layer types: {dict(sorted(layer_types.items(), key=lambda x: -x[1]))}")
