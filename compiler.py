# ANECompiler ctypes wrapper
# Provides Python bindings to Apple's private ANE (Apple Neural Engine) compiler framework
from __future__ import annotations
import ctypes
import struct
from dataclasses import dataclass, field
from typing import Optional, Any
from pathlib import Path

# Framework paths
_FRAMEWORKS = {
  "CoreFoundation": "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation",
  "CoreGraphics": "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics",
  "IOKit": "/System/Library/Frameworks/IOKit.framework/IOKit",
  "Metal": "/System/Library/Frameworks/Metal.framework/Metal",
  "ANECompiler": "/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler",
}

# Lazy-loaded framework handles
_libs: dict[str, ctypes.CDLL] = {}

def _load_frameworks() -> ctypes.CDLL:
  """Load ANECompiler and its dependencies. Returns ANECompiler handle."""
  if "ANECompiler" in _libs:
    return _libs["ANECompiler"]

  # Load in dependency order
  for name in ["CoreFoundation", "CoreGraphics", "IOKit", "Metal", "ANECompiler"]:
    path = _FRAMEWORKS[name]
    try:
      _libs[name] = ctypes.CDLL(path)
    except OSError as e:
      raise RuntimeError(f"Failed to load {name}: {e}") from e

  return _libs["ANECompiler"]


# ============================================================================
# ANE Struct Definitions (inferred from runtime probing)
# ============================================================================

@dataclass
class ANECKernelSize:
  """Kernel size for convolution/pooling (D, H, W). 24 bytes."""
  depth: int = 1
  height: int = 1
  width: int = 1

  def to_bytes(self) -> bytes:
    # Each dimension is u64 (possibly u32 + padding)
    return struct.pack("<QQQ", self.depth, self.height, self.width)

  @classmethod
  def from_bytes(cls, data: bytes) -> "ANECKernelSize":
    d, h, w = struct.unpack("<QQQ", data[:24])
    return cls(depth=d, height=h, width=w)


@dataclass
class ANECStep:
  """Stride for convolution/pooling (D, H, W). 12 bytes."""
  depth: int = 1
  height: int = 1
  width: int = 1

  def to_bytes(self) -> bytes:
    return struct.pack("<III", self.depth, self.height, self.width)

  @classmethod
  def from_bytes(cls, data: bytes) -> "ANECStep":
    d, h, w = struct.unpack("<III", data[:12])
    return cls(depth=d, height=h, width=w)


@dataclass
class ANECPadding:
  """Padding (6 values: front/back for each of D, H, W). 24 bytes."""
  depth_front: int = 0
  depth_back: int = 0
  height_front: int = 0
  height_back: int = 0
  width_front: int = 0
  width_back: int = 0

  def to_bytes(self) -> bytes:
    return struct.pack("<IIIIII",
      self.depth_front, self.depth_back,
      self.height_front, self.height_back,
      self.width_front, self.width_back)

  @classmethod
  def from_bytes(cls, data: bytes) -> "ANECPadding":
    df, db, hf, hb, wf, wb = struct.unpack("<IIIIII", data[:24])
    return cls(depth_front=df, depth_back=db, height_front=hf, height_back=hb, width_front=wf, width_back=wb)


@dataclass
class ANECTensorDims:
  """Tensor dimensions (N, C, H, W, D). 40 bytes (5 x u64)."""
  n: int = 1  # batch
  c: int = 1  # channels
  h: int = 1  # height
  w: int = 1  # width
  d: int = 1  # depth

  def to_bytes(self) -> bytes:
    return struct.pack("<QQQQQ", self.n, self.c, self.h, self.w, self.d)

  @classmethod
  def from_bytes(cls, data: bytes) -> "ANECTensorDims":
    n, c, h, w, d = struct.unpack("<QQQQQ", data[:40])
    return cls(n=n, c=c, h=h, w=w, d=d)


@dataclass
class ANECTensorDesc:
  """Tensor descriptor. 64 bytes.
  
  Fields (inferred):
    - ptr0: pointer to type info or name (8 bytes)
    - dims: 6 dimension values as u64 (48 bytes) 
    - flags: 8 bytes
  """
  _raw: bytes = field(default_factory=lambda: b'\x00' * 64, repr=False)
  dims: ANECTensorDims = field(default_factory=ANECTensorDims)

  def to_bytes(self) -> bytes:
    return self._raw

  @classmethod
  def from_bytes(cls, data: bytes) -> "ANECTensorDesc":
    # Parse dimensions starting at offset 8
    dims = ANECTensorDims.from_bytes(data[8:48])
    return cls(_raw=data[:64], dims=dims)


# ============================================================================
# ANE Layer Descriptors
# ============================================================================

@dataclass
class ANECConvLayerDesc:
  """Convolution layer descriptor. 176 bytes.
  
  Contains embedded structs for kernel size, stride, padding, etc.
  """
  _raw: bytes = field(default_factory=lambda: b'\x00' * 176, repr=False)
  kernel_size: ANECKernelSize = field(default_factory=ANECKernelSize)
  stride: ANECStep = field(default_factory=ANECStep)
  padding: ANECPadding = field(default_factory=ANECPadding)

  @property
  def size(self) -> int:
    return 176


@dataclass
class ANECPoolLayerDesc:
  """Pooling layer descriptor. 96 bytes."""
  _raw: bytes = field(default_factory=lambda: b'\x00' * 96, repr=False)
  kernel_size: ANECKernelSize = field(default_factory=ANECKernelSize)
  stride: ANECStep = field(default_factory=ANECStep)

  @property
  def size(self) -> int:
    return 96


@dataclass  
class ANECLinearLayerDesc:
  """Linear/Dense layer descriptor. 64 bytes."""
  _raw: bytes = field(default_factory=lambda: b'\x00' * 64, repr=False)

  @property
  def size(self) -> int:
    return 64


@dataclass
class ANECMatrixMultLayerDesc:
  """Matrix multiplication layer descriptor. 16 bytes."""
  _raw: bytes = field(default_factory=lambda: b'\x00' * 16, repr=False)
  transpose_a: bool = False
  transpose_b: bool = False

  @property
  def size(self) -> int:
    return 16


@dataclass
class ANECSoftmaxLayerDesc:
  """Softmax layer descriptor. 48 bytes."""
  _raw: bytes = field(default_factory=lambda: b'\x00' * 48, repr=False)
  axis: int = -1

  @property
  def size(self) -> int:
    return 48


@dataclass
class ANECSDPALayerDesc:
  """Scaled Dot-Product Attention layer descriptor. 8 bytes.
  
  This is the transformer attention mechanism!
  """
  _raw: bytes = field(default_factory=lambda: b'\x00' * 8, repr=False)

  @property
  def size(self) -> int:
    return 8


# ============================================================================
# ANE Runtime API
# ============================================================================

class ANECompiler:
  """Python wrapper for ANECompiler private framework."""

  def __init__(self):
    self._ane = _load_frameworks()
    self._cf = _libs["CoreFoundation"]
    self._setup_functions()

  def _setup_functions(self):
    """Setup ctypes function signatures."""

    self._get_mps_version = None
    self._get_spi_version = None
    self._get_net_version = None
    self._get_analytics_size = None

    # Version getters
    try:
      self._get_mps_version = self._ane.ANECGetMPSDialectSupportedVersion
      self._get_mps_version.restype = ctypes.c_uint64
      self._get_mps_version.argtypes = []
    except AttributeError:
      pass

    try:
      self._get_spi_version = self._ane.ANECGetMPSSPIDialectSupportedVersion
      self._get_spi_version.restype = ctypes.c_uint64
      self._get_spi_version.argtypes = []
    except AttributeError:
      pass
    
    try:
      self._get_net_version = self._ane.ANECGetValidateNetworkSupportedVersion
      self._get_net_version.restype = ctypes.c_uint64
      self._get_net_version.argtypes = []
    except AttributeError:
      pass

    try:
      self._get_analytics_size = self._ane.ANECGetAnalyticsBufferSize
      self._get_analytics_size.restype = ctypes.c_size_t
      self._get_analytics_size.argtypes = []
    except AttributeError:
      pass

    # Struct initializers (all take void* and return void)
    self._init_funcs: dict[str, Any] = {}
    init_names = [
      "ANECKernelSizeInitialize",
      "ANECStepInitialize",
      "ANECPaddingInitialize",
      "ANECTensorDimsInitialize",
      "ANECTensorDescInitialize",
      "ANECConvLayerDescInitialize",
      "ANECPoolLayerDescInitialize",
      "ANECLinearLayerDescInitialize",
      "ANECMatrixMultLayerDescInitialize",
      "ANECSoftmaxLayerDescInitialize",
      "ANECSDPALayerDescInitialize",
      "ANECNeuronLayerDescInitialize",
      "ANECReductionLayerDescInitialize",
      "ANECReshapeLayerDescInitialize",
      "ANECTransposeLayerDescInitialize",
      "ANECConcatLayerDescInitialize",
      "ANECGatherLayerDescInitialize",
    ]
    for name in init_names:
      try:
        func = getattr(self._ane, name)
        func.restype = None
        func.argtypes = [ctypes.c_void_p]
        self._init_funcs[name] = func
      except AttributeError:
        pass  # Some functions may not exist on all OS versions

  # --- Version Info ---

  @property
  def mps_dialect_version(self) -> int:
    """Get MPS dialect supported version."""
    return self._get_mps_version() if self._get_mps_version else None

  @property
  def mps_spi_dialect_version(self) -> int:
    """Get MPS SPI dialect supported version."""
    return self._get_spi_version() if self._get_spi_version else None

  @property
  def validate_network_version(self) -> int:
    """Get validate network supported version."""
    return self._get_net_version() if self._get_net_version else None

  @property
  def analytics_buffer_size(self) -> int:
    """Get analytics buffer size."""
    return self._get_analytics_size()

  # --- Struct Initialization ---

  def init_kernel_size(self, d: int = 1, h: int = 1, w: int = 1) -> ANECKernelSize | None:
    """Create and initialize a kernel size struct."""
    if "ANECKernelSizeInitialize" not in self._init_funcs:
      return None
    buf = ctypes.create_string_buffer(24)
    self._init_funcs["ANECKernelSizeInitialize"](ctypes.cast(buf, ctypes.c_void_p))
    ks = ANECKernelSize.from_bytes(buf.raw)
    ks.depth, ks.height, ks.width = d, h, w
    return ks

  def init_step(self, d: int = 1, h: int = 1, w: int = 1) -> ANECStep | None:
    """Create and initialize a stride/step struct."""
    if "ANECStepInitialize" not in self._init_funcs:
      return None
    buf = ctypes.create_string_buffer(12)
    self._init_funcs["ANECStepInitialize"](ctypes.cast(buf, ctypes.c_void_p))
    step = ANECStep.from_bytes(buf.raw)
    step.depth, step.height, step.width = d, h, w
    return step

  def init_padding(self, df: int = 0, db: int = 0, hf: int = 0, hb: int = 0, wf: int = 0, wb: int = 0) -> ANECPadding | None:
    """Create and initialize a padding struct."""
    if "ANECPaddingInitialize" not in self._init_funcs:
      return None
    buf = ctypes.create_string_buffer(24)
    self._init_funcs["ANECPaddingInitialize"](ctypes.cast(buf, ctypes.c_void_p))
    pad = ANECPadding.from_bytes(buf.raw)
    pad.depth_front, pad.depth_back = df, db
    pad.height_front, pad.height_back = hf, hb
    pad.width_front, pad.width_back = wf, wb
    return pad

  def init_tensor_dims(self, n: int = 1, c: int = 1, h: int = 1, w: int = 1, d: int = 1) -> ANECTensorDims | None:
    """Create and initialize tensor dimensions."""
    if "ANECTensorDimsInitialize" not in self._init_funcs:
      return None
    buf = ctypes.create_string_buffer(40)
    self._init_funcs["ANECTensorDimsInitialize"](ctypes.cast(buf, ctypes.c_void_p))
    dims = ANECTensorDims.from_bytes(buf.raw)
    dims.n, dims.c, dims.h, dims.w, dims.d = n, c, h, w, d
    return dims

  def init_tensor_desc(self) -> ANECTensorDesc | None:
    """Create and initialize a tensor descriptor."""
    if "ANECTensorDescInitialize" not in self._init_funcs:
      return None
    buf = ctypes.create_string_buffer(64)
    self._init_funcs["ANECTensorDescInitialize"](ctypes.cast(buf, ctypes.c_void_p))
    return ANECTensorDesc.from_bytes(buf.raw)

  def init_conv_layer_desc(self) -> ANECConvLayerDesc | None:
    """Create and initialize a convolution layer descriptor."""
    if "ANECConvLayerDescInitialize" not in self._init_funcs:
      return None
    buf = ctypes.create_string_buffer(176)
    self._init_funcs["ANECConvLayerDescInitialize"](ctypes.cast(buf, ctypes.c_void_p))
    return ANECConvLayerDesc(_raw=buf.raw)

  def init_pool_layer_desc(self) -> ANECPoolLayerDesc | None:
    """Create and initialize a pooling layer descriptor."""
    if "ANECPoolLayerDescInitialize" not in self._init_funcs:
      return None
    buf = ctypes.create_string_buffer(96)
    self._init_funcs["ANECPoolLayerDescInitialize"](ctypes.cast(buf, ctypes.c_void_p))
    return ANECPoolLayerDesc(_raw=buf.raw)

  def init_linear_layer_desc(self) -> ANECLinearLayerDesc | None:
    """Create and initialize a linear layer descriptor."""
    if "ANECLinearLayerDescInitialize" not in self._init_funcs:
      return None
    buf = ctypes.create_string_buffer(64)
    self._init_funcs["ANECLinearLayerDescInitialize"](ctypes.cast(buf, ctypes.c_void_p))
    return ANECLinearLayerDesc(_raw=buf.raw)

  def init_matmul_layer_desc(self) -> ANECMatrixMultLayerDesc | None:
    """Create and initialize a matrix multiplication layer descriptor."""
    if "ANECMatrixMultLayerDescInitialize" not in self._init_funcs:
      return None
    buf = ctypes.create_string_buffer(16)
    self._init_funcs["ANECMatrixMultLayerDescInitialize"](ctypes.cast(buf, ctypes.c_void_p))
    return ANECMatrixMultLayerDesc(_raw=buf.raw)

  def init_softmax_layer_desc(self) -> ANECSoftmaxLayerDesc | None:
    """Create and initialize a softmax layer descriptor."""
    if "ANECSoftmaxLayerDescInitialize" not in self._init_funcs:
      return None
    buf = ctypes.create_string_buffer(48)
    self._init_funcs["ANECSoftmaxLayerDescInitialize"](ctypes.cast(buf, ctypes.c_void_p))
    return ANECSoftmaxLayerDesc(_raw=buf.raw)

  def init_sdpa_layer_desc(self) -> ANECSDPALayerDesc | None:
    """Create and initialize a scaled dot-product attention layer descriptor."""
    if "ANECSDPALayerDescInitialize" not in self._init_funcs:
      return None
    buf = ctypes.create_string_buffer(8)
    self._init_funcs["ANECSDPALayerDescInitialize"](ctypes.cast(buf, ctypes.c_void_p))
    return ANECSDPALayerDesc(_raw=buf.raw)

# ============================================================================
# Struct Size Constants (for reference)
# ============================================================================

STRUCT_SIZES = {
  "ANECKernelSize": 24,
  "ANECStep": 12,
  "ANECPadding": 24,
  "ANECTensorDims": 40,
  "ANECTensorDesc": 64,
  "ANECConvLayerDesc": 176,
  "ANECPoolLayerDesc": 96,
  "ANECLinearLayerDesc": 64,
  "ANECMatrixMultLayerDesc": 16,
  "ANECSoftmaxLayerDesc": 48,
  "ANECSDPALayerDesc": 8,
}


# ============================================================================
# Convenience Functions
# ============================================================================

def get_ane_info() -> dict[str, Any]:
  """Get ANE compiler information."""
  try:
    ane = ANECompiler()
    return {
      "available": True,
      "mps_dialect_version": ane.mps_dialect_version,
      "mps_spi_dialect_version": ane.mps_spi_dialect_version,
      "validate_network_version": ane.validate_network_version,
      "analytics_buffer_size": ane.analytics_buffer_size,
      "struct_sizes": STRUCT_SIZES,
    }
  except Exception as e:
    return {
      "available": False,
      "error": str(e),
    }


def probe_all_layer_desc_sizes() -> dict[str, int]:
  """Probe sizes of all LayerDesc structs by calling their Initialize functions."""
  ane_lib = _load_frameworks()
  
  # Find all LayerDescInitialize functions
  import subprocess
  result = subprocess.run(
    ["dyld_info", "-exports", _FRAMEWORKS["ANECompiler"]],
    capture_output=True, text=True, timeout=30
  )
  
  sizes = {}
  for line in result.stdout.split("\n"):
    if "LayerDescInitialize" in line:
      parts = line.strip().split()
      if len(parts) >= 2:
        name = parts[1].lstrip("_")
        struct_name = name.replace("Initialize", "").replace("ANEC", "ANEC")
        
        # Probe size
        try:
          func = getattr(ane_lib, name)
          func.restype = None
          func.argtypes = [ctypes.c_void_p]
          
          buf = ctypes.create_string_buffer(1024)
          for i in range(1024):
            buf[i] = 0xAA
          
          func(ctypes.cast(buf, ctypes.c_void_p))
          
          # Find last modified byte
          raw = buf.raw
          last = 0
          for i in range(1023, -1, -1):
            if raw[i] != 0xAA:
              last = i + 1
              break
          
          sizes[struct_name] = last
        except Exception:
          pass
  
  return sizes


if __name__ == "__main__":
    print("ANE Compiler Info:")
    info = get_ane_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
    if info.get("available"):
        print("\nTesting struct initialization...")
        ane = ANECompiler()
        tests = [
            ("KernelSize", lambda: ane.init_kernel_size(3, 3, 3)),
            ("Step", lambda: ane.init_step(2, 2, 2)),
            ("Padding", lambda: ane.init_padding(1, 1, 2, 2, 3, 3)),
            ("TensorDims", lambda: ane.init_tensor_dims(1, 3, 224, 224, 1)),
            ("TensorDesc", lambda: ane.init_tensor_desc()),
            ("ConvLayerDesc", lambda: ane.init_conv_layer_desc()),
            ("PoolLayerDesc", lambda: ane.init_pool_layer_desc()),
            ("LinearLayerDesc", lambda: ane.init_linear_layer_desc()),
            ("MatMulLayerDesc", lambda: ane.init_matmul_layer_desc()),
            ("SoftmaxLayerDesc", lambda: ane.init_softmax_layer_desc()),
            ("SDPALayerDesc", lambda: ane.init_sdpa_layer_desc()),
        ]
        RED = '\033[91m'
        for name, init_func in tests:
            result = init_func()
            if result:
                print(f"  {name}: {result}")
            else:
                print(f"  {name}: {RED}Not available on this macOS version")
