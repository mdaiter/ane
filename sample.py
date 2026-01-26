#!/usr/bin/env python3
"""
Sample: Using ANECompiler private API to build neural network layer descriptors.

This demonstrates the ANE (Apple Neural Engine) compiler's internal data structures
for representing neural network operations. While we can't yet compile/run models
(that requires figuring out ANECCompile's signature), we CAN:

1. Initialize layer descriptors with proper default values
2. Understand the struct layouts for each layer type
3. Prepare for future integration with CoreML/MPS

Supported layers (that we've reverse-engineered):
- Convolution (ANECConvLayerDesc) - 176 bytes
- Pooling (ANECPoolLayerDesc) - 96 bytes  
- Linear/Dense (ANECLinearLayerDesc) - 64 bytes
- Matrix Multiply (ANECMatrixMultLayerDesc) - 16 bytes
- Softmax (ANECSoftmaxLayerDesc) - 48 bytes
- SDPA/Attention (ANECSDPALayerDesc) - 8 bytes (transformers!)
- Reshape, Transpose, Concat, etc.
"""
from __future__ import annotations
import ctypes
from dataclasses import dataclass
from typing import Optional

from .compiler import (
  ANECompiler,
  ANECKernelSize,
  ANECStep,
  ANECPadding,
  ANECTensorDims,
  ANECConvLayerDesc,
  ANECPoolLayerDesc,
  ANECLinearLayerDesc,
  ANECMatrixMultLayerDesc,
  ANECSoftmaxLayerDesc,
  ANECSDPALayerDesc,
  STRUCT_SIZES,
)


@dataclass
class ANELayer:
  """A layer in an ANE computation graph."""
  name: str
  op_type: str
  desc_size: int
  input_dims: ANECTensorDims
  output_dims: Optional[ANECTensorDims] = None
  kernel_size: Optional[ANECKernelSize] = None
  stride: Optional[ANECStep] = None
  padding: Optional[ANECPadding] = None


class SimpleANEGraph:
  """Build a simple ANE computation graph (for demonstration)."""
  
  def __init__(self):
    self.ane = ANECompiler()
    self.layers: list[ANELayer] = []
  
  def add_conv2d(
    self,
    name: str,
    input_dims: tuple[int, int, int, int],  # N, C, H, W
    out_channels: int,
    kernel_size: int | tuple[int, int] = 3,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
  ) -> "SimpleANEGraph":
    """Add a 2D convolution layer."""
    n, c, h, w = input_dims
    
    # Parse kernel/stride/padding
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (padding, padding) if isinstance(padding, int) else padding
    
    # Calculate output dimensions
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    
    # Create ANE structs
    ks = self.ane.init_kernel_size(d=1, h=kh, w=kw)
    step = self.ane.init_step(d=1, h=sh, w=sw)
    pad = self.ane.init_padding(hf=ph, hb=ph, wf=pw, wb=pw)
    in_dims = self.ane.init_tensor_dims(n=n, c=c, h=h, w=w)
    out_dims = self.ane.init_tensor_dims(n=n, c=out_channels, h=out_h, w=out_w)
    
    # Initialize conv descriptor
    conv_desc = self.ane.init_conv_layer_desc()
    
    self.layers.append(ANELayer(
      name=name,
      op_type="conv2d",
      desc_size=conv_desc.size,
      input_dims=in_dims,
      output_dims=out_dims,
      kernel_size=ks,
      stride=step,
      padding=pad,
    ))
    
    return self
  
  def add_pool2d(
    self,
    name: str,
    input_dims: tuple[int, int, int, int],
    kernel_size: int = 2,
    stride: int = 2,
  ) -> "SimpleANEGraph":
    """Add a 2D pooling layer."""
    n, c, h, w = input_dims
    
    out_h = h // stride
    out_w = w // stride
    
    ks = self.ane.init_kernel_size(h=kernel_size, w=kernel_size)
    step = self.ane.init_step(h=stride, w=stride)
    in_dims = self.ane.init_tensor_dims(n=n, c=c, h=h, w=w)
    out_dims = self.ane.init_tensor_dims(n=n, c=c, h=out_h, w=out_w)
    
    pool_desc = self.ane.init_pool_layer_desc()
    
    self.layers.append(ANELayer(
      name=name,
      op_type="pool2d",
      desc_size=pool_desc.size,
      input_dims=in_dims,
      output_dims=out_dims,
      kernel_size=ks,
      stride=step,
    ))
    
    return self
  
  def add_linear(
    self,
    name: str,
    input_features: int,
    output_features: int,
    batch_size: int = 1,
  ) -> "SimpleANEGraph":
    """Add a linear/dense layer."""
    in_dims = self.ane.init_tensor_dims(n=batch_size, c=input_features)
    out_dims = self.ane.init_tensor_dims(n=batch_size, c=output_features)
    
    linear_desc = self.ane.init_linear_layer_desc()
    
    self.layers.append(ANELayer(
      name=name,
      op_type="linear",
      desc_size=linear_desc.size,
      input_dims=in_dims,
      output_dims=out_dims,
    ))
    
    return self
  
  def add_matmul(self, name: str, m: int, k: int, n: int, batch: int = 1) -> "SimpleANEGraph":
    """Add a matrix multiplication: (batch, m, k) @ (batch, k, n) -> (batch, m, n)."""
    in_dims = self.ane.init_tensor_dims(n=batch, c=m, h=k)
    out_dims = self.ane.init_tensor_dims(n=batch, c=m, h=n)
    
    matmul_desc = self.ane.init_matmul_layer_desc()
    
    self.layers.append(ANELayer(
      name=name,
      op_type="matmul",
      desc_size=matmul_desc.size,
      input_dims=in_dims,
      output_dims=out_dims,
    ))
    
    return self
  
  def add_softmax(self, name: str, dims: tuple[int, ...], axis: int = -1) -> "SimpleANEGraph":
    """Add a softmax layer."""
    in_dims = self.ane.init_tensor_dims(
      n=dims[0] if len(dims) > 0 else 1,
      c=dims[1] if len(dims) > 1 else 1,
      h=dims[2] if len(dims) > 2 else 1,
      w=dims[3] if len(dims) > 3 else 1,
    )
    
    softmax_desc = self.ane.init_softmax_layer_desc()
    
    self.layers.append(ANELayer(
      name=name,
      op_type="softmax",
      desc_size=softmax_desc.size,
      input_dims=in_dims,
      output_dims=in_dims,  # Same shape
    ))
    
    return self
  
  def add_sdpa(
    self,
    name: str,
    batch: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
  ) -> "SimpleANEGraph":
    """Add Scaled Dot-Product Attention (transformer attention).
    
    Computes: softmax(Q @ K^T / sqrt(d)) @ V
    
    Args:
      batch: Batch size
      num_heads: Number of attention heads
      seq_len: Sequence length
      head_dim: Dimension per head
    """
    # Input: Q, K, V each are (batch, num_heads, seq_len, head_dim)
    in_dims = self.ane.init_tensor_dims(
      n=batch,
      c=num_heads,
      h=seq_len,
      w=head_dim,
    )
    
    sdpa_desc = self.ane.init_sdpa_layer_desc()
    
    self.layers.append(ANELayer(
      name=name,
      op_type="sdpa",
      desc_size=sdpa_desc.size,
      input_dims=in_dims,
      output_dims=in_dims,  # Output same shape as Q
    ))
    
    return self
  
  def summary(self) -> str:
    """Print a summary of the graph."""
    lines = ["ANE Computation Graph", "=" * 60]
    
    total_desc_bytes = 0
    for layer in self.layers:
      total_desc_bytes += layer.desc_size
      
      in_shape = f"({layer.input_dims.n}, {layer.input_dims.c}, {layer.input_dims.h}, {layer.input_dims.w})"
      out_shape = f"({layer.output_dims.n}, {layer.output_dims.c}, {layer.output_dims.h}, {layer.output_dims.w})" if layer.output_dims else "?"
      
      lines.append(f"\n{layer.name} ({layer.op_type})")
      lines.append(f"  Input:  {in_shape}")
      lines.append(f"  Output: {out_shape}")
      lines.append(f"  Desc:   {layer.desc_size} bytes")
      
      if layer.kernel_size:
        lines.append(f"  Kernel: {layer.kernel_size.height}x{layer.kernel_size.width}")
      if layer.stride:
        lines.append(f"  Stride: {layer.stride.height}x{layer.stride.width}")
      if layer.padding:
        lines.append(f"  Pad:    {layer.padding.height_front},{layer.padding.width_front}")
    
    lines.append("\n" + "=" * 60)
    lines.append(f"Total layers: {len(self.layers)}")
    lines.append(f"Total descriptor bytes: {total_desc_bytes}")
    
    return "\n".join(lines)


def build_simple_cnn() -> SimpleANEGraph:
  """Build a simple CNN (like a tiny image classifier)."""
  graph = SimpleANEGraph()
  
  # Input: (1, 3, 224, 224) - batch=1, RGB, 224x224
  graph.add_conv2d("conv1", (1, 3, 224, 224), out_channels=64, kernel_size=7, stride=2, padding=3)
  # Output: (1, 64, 112, 112)
  
  graph.add_pool2d("pool1", (1, 64, 112, 112), kernel_size=3, stride=2)
  # Output: (1, 64, 56, 56)
  
  graph.add_conv2d("conv2", (1, 64, 56, 56), out_channels=128, kernel_size=3, padding=1)
  # Output: (1, 128, 56, 56)
  
  graph.add_conv2d("conv3", (1, 128, 56, 56), out_channels=256, kernel_size=3, padding=1)
  # Output: (1, 256, 56, 56)
  
  graph.add_pool2d("pool2", (1, 256, 56, 56), kernel_size=2, stride=2)
  # Output: (1, 256, 28, 28)
  
  # Global average pooling would flatten to (1, 256)
  graph.add_linear("fc1", input_features=256*28*28, output_features=1024)
  graph.add_linear("fc2", input_features=1024, output_features=1000)
  graph.add_softmax("softmax", (1, 1000))
  
  return graph


def build_transformer_attention() -> SimpleANEGraph:
  """Build a transformer attention block."""
  graph = SimpleANEGraph()
  
  batch = 1
  num_heads = 8
  seq_len = 512
  head_dim = 64
  embed_dim = num_heads * head_dim  # 512
  
  # Project to Q, K, V (in practice these are separate linear layers)
  graph.add_linear("proj_qkv", input_features=embed_dim, output_features=embed_dim*3, batch_size=seq_len)
  
  # Scaled Dot-Product Attention
  graph.add_sdpa("attention", batch=batch, num_heads=num_heads, seq_len=seq_len, head_dim=head_dim)
  
  # Output projection
  graph.add_linear("proj_out", input_features=embed_dim, output_features=embed_dim, batch_size=seq_len)
  
  return graph


if __name__ == "__main__":
  print("=" * 60)
  print("Example 1: Simple CNN")
  print("=" * 60)
  cnn = build_simple_cnn()
  print(cnn.summary())
  
  print("\n")
  print("=" * 60)
  print("Example 2: Transformer Attention Block")
  print("=" * 60)
  transformer = build_transformer_attention()
  print(transformer.summary())
  
  print("\n")
  print("=" * 60)
  print("All Available Layer Types")
  print("=" * 60)
  
  # Show all layer types we discovered
  from .compiler import probe_all_layer_desc_sizes
  all_sizes = probe_all_layer_desc_sizes()
  
  print(f"\nTotal layer types: {len(all_sizes)}")
  print("\nBy category:")
  
  categories = {
    "Attention/Transformer": ["SDPA"],
    "Convolution": ["Conv", "CrossCorrelation"],
    "Pooling": ["Pool"],
    "Normalization": ["Norm", "LRN"],
    "Linear/Matrix": ["Linear", "MatrixMult"],
    "Activation": ["Neuron", "Softmax", "Dropout"],
    "Reshape/Layout": ["Reshape", "Transpose", "Flatten", "Unflatten", "Concat", "Tile"],
    "Spatial": ["Resize", "Pad", "CropResize", "Resample", "AffineTransform"],
    "Reduction": ["Reduction", "TopK", "Sort", "NMS"],
    "Other": ["Gather", "Shape", "Random", "RingBuffer", "InputView"],
  }
  
  for cat, keywords in categories.items():
    matching = [name for name in all_sizes.keys() if any(kw in name for kw in keywords)]
    if matching:
      print(f"\n{cat}:")
      for name in matching:
        print(f"  {name}: {all_sizes[name]} bytes")
