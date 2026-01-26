# espresso.net format documentation and utilities
"""
Espresso Model Format
=====================

Apple's neural network format used by ANE and Core ML.

File Types
----------
- .espresso.net    - Network graph (JSON or pbze binary)
- .espresso.weights - Binary weights
- .espresso.shape   - Shape metadata

JSON Format (format_version 200-300)
------------------------------------
{
  "format_version": 200|300,
  "storage": "model.espresso.weights",
  "layers": [...],
  "metadata_in_weights": bool,
  "analyses": {...},
  "properties": {...},
  "shape": {...},
  "subnetworks": [...]
}

Layer Types (discovered from system models)
-------------------------------------------
Compute:
  - inner_product: Dense/FC layer, supports quantization
  - convolution: 2D convolution
  - batch_matmul: Batched matrix multiplication
  - elementwise: Binary/unary ops (operation codes 0-120+)
  - activation: relu, gelu, tanh, etc.
  - softmax: Softmax normalization
  - reduce: Reduction ops (sum, mean, max)

Memory/Shape:
  - reshape: Tensor reshape
  - transpose: Permute dimensions
  - concat: Concatenate tensors
  - general_concat: N-D concatenation
  - split_nd: Split tensor along axis
  - general_slice: Slice tensor
  - expand_dims: Add dimension
  - load_constant: Load constant tensor

Quantization:
  - dynamic_quantize: Runtime quantization
  - dynamic_dequantize: Runtime dequantization

Special:
  - instancenorm_1d: Instance normalization
  - get_shape: Get tensor shape
  - nonzero: Find nonzero indices
  - scatter_nd: Scatter operation
  - tile: Tile tensor

CPU vs ANE Differences
----------------------
ANE models tend to:
1. Have MORE layers (decomposed operations)
2. Use convolution instead of reshape where possible
3. Use quantization_mode:2 on inner_product
4. Use is_lookup:1 for embedding layers
5. Use split_nd/concat for tensor manipulation

Binary Format (pbze)
--------------------
Header: b'pbze' (4 bytes)
Appears to be compressed/encoded JSON.
Found in system imaging models.

Layer Operation Codes (elementwise)
-----------------------------------
0: add
1: sub
2: mul
3: div
101: select (ternary)
105: less_than
107: not_equal
117: floor
118: ceil
(Many more undocumented)
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class EspressoLayer:
  """A layer in an espresso.net model."""
  name: str
  type: str
  bottom: list[str]  # Input blobs
  top: str           # Output blob
  attributes: dict[str, Any] = field(default_factory=dict)
  
  @classmethod
  def from_dict(cls, d: dict) -> "EspressoLayer":
    bottom = d.get("bottom", "")
    if isinstance(bottom, str):
      bottom = [b.strip() for b in bottom.split(",") if b.strip()]
    return cls(
      name=d.get("name", ""),
      type=d.get("type", ""),
      bottom=bottom,
      top=d.get("top", ""),
      attributes={k: v for k, v in d.items() if k not in ["name", "type", "bottom", "top", "weights"]},
    )


@dataclass 
class EspressoNet:
  """An espresso.net model."""
  format_version: int
  storage: Optional[str]
  layers: list[EspressoLayer]
  properties: dict[str, Any] = field(default_factory=dict)
  
  @classmethod
  def from_file(cls, path: str | Path) -> "EspressoNet":
    """Load from .espresso.net file."""
    path = Path(path)
    with open(path) as f:
      content = f.read()
    
    # Check for binary format
    if content.startswith("pbze"):
      raise ValueError("Binary pbze format not yet supported")
    
    data = json.loads(content)
    return cls(
      format_version=data.get("format_version", 0),
      storage=data.get("storage"),
      layers=[EspressoLayer.from_dict(l) for l in data.get("layers", [])],
      properties=data.get("properties", {}),
    )
  
  def layer_type_counts(self) -> dict[str, int]:
    """Count layers by type."""
    counts: dict[str, int] = {}
    for layer in self.layers:
      counts[layer.type] = counts.get(layer.type, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))
  
  def find_layers(self, layer_type: str) -> list[EspressoLayer]:
    """Find all layers of a given type."""
    return [l for l in self.layers if l.type == layer_type]
  
  def get_inner_product_info(self) -> list[dict]:
    """Get info about inner_product layers (for quantization analysis)."""
    results = []
    for layer in self.find_layers("inner_product"):
      results.append({
        "name": layer.name,
        "nB": layer.attributes.get("nB"),
        "nC": layer.attributes.get("nC"),
        "quantization_mode": layer.attributes.get("quantization_mode", 0),
        "is_lookup": layer.attributes.get("is_lookup", 0),
        "has_biases": layer.attributes.get("has_biases", 0),
      })
    return results


def compare_cpu_ane_models(cpu_path: str, ane_path: str) -> dict:
  """Compare CPU and ANE versions of the same model."""
  cpu = EspressoNet.from_file(cpu_path)
  ane = EspressoNet.from_file(ane_path)
  
  cpu_types = set(cpu.layer_type_counts().keys())
  ane_types = set(ane.layer_type_counts().keys())
  
  return {
    "cpu_layer_count": len(cpu.layers),
    "ane_layer_count": len(ane.layers),
    "cpu_format_version": cpu.format_version,
    "ane_format_version": ane.format_version,
    "cpu_only_types": cpu_types - ane_types,
    "ane_only_types": ane_types - cpu_types,
    "common_types": cpu_types & ane_types,
    "cpu_type_counts": cpu.layer_type_counts(),
    "ane_type_counts": ane.layer_type_counts(),
  }


def find_system_models() -> list[Path]:
  """Find espresso.net models in system frameworks."""
  import subprocess
  result = subprocess.run(
    ["find", "/System/Library", "-name", "*.espresso.net"],
    capture_output=True, text=True, timeout=60
  )
  paths = [Path(p) for p in result.stdout.strip().split("\n") if p]
  return paths


def is_json_format(path: str | Path) -> bool:
  """Check if espresso.net file is JSON (vs binary pbze)."""
  with open(path, "rb") as f:
    header = f.read(4)
  return header != b"pbze"


if __name__ == "__main__":
  print("=== Espresso Format Analysis ===\n")
  
  # Find system models
  print("Finding system models...")
  models = find_system_models()
  print(f"Found {len(models)} espresso.net files")
  
  # Categorize by format
  json_models = [m for m in models if is_json_format(m)]
  binary_models = [m for m in models if not is_json_format(m)]
  print(f"  JSON format: {len(json_models)}")
  print(f"  Binary (pbze): {len(binary_models)}")
  
  # Load a sample JSON model
  if json_models:
    sample = json_models[0]
    print(f"\n=== Sample: {sample.name} ===")
    try:
      net = EspressoNet.from_file(sample)
      print(f"Format version: {net.format_version}")
      print(f"Layer count: {len(net.layers)}")
      print(f"Layer types: {net.layer_type_counts()}")
      
      # Check for quantization
      ip_layers = net.get_inner_product_info()
      if ip_layers:
        print(f"\nInner product layers ({len(ip_layers)}):")
        for ip in ip_layers[:3]:
          print(f"  {ip['name']}: {ip['nB']}x{ip['nC']}, quant={ip['quantization_mode']}, lookup={ip['is_lookup']}")
    except Exception as e:
      print(f"Error loading: {e}")
  
  # Try CPU vs ANE comparison
  cpu_path = "/System/Library/LinguisticData/RequiredAssets_en.bundle/AssetData/en.lm/unilm.bundle/cpu_embeddings.espresso.net"
  ane_path = "/System/Library/LinguisticData/RequiredAssets_en.bundle/AssetData/en.lm/unilm.bundle/ane_embeddings.espresso.net"
  
  if Path(cpu_path).exists() and Path(ane_path).exists():
    print("\n=== CPU vs ANE Comparison ===")
    comparison = compare_cpu_ane_models(cpu_path, ane_path)
    print(f"CPU layers: {comparison['cpu_layer_count']}")
    print(f"ANE layers: {comparison['ane_layer_count']}")
    print(f"CPU-only types: {comparison['cpu_only_types']}")
    print(f"ANE-only types: {comparison['ane_only_types']}")
