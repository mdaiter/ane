# Apple Neural Engine (ANE) Reverse Engineering

Reverse engineering artifacts for Apple's Neural Engine stack: **ANECompiler**, **Espresso**, and **AppleNeuralEngine** frameworks.

> **Target Audience**: Performance engineers and security researchers working with Apple silicon ML acceleration.

## Table of Contents

- [Key Findings](#key-findings)
- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [Espresso Engine Teardown](#espresso-engine-teardown)
- [Compiler Engine Teardown](#compiler-engine-teardown)
- [ANE Runtime Details](#ane-runtime-details)
- [Security Analysis](#security-analysis)
- [Performance Analysis](#performance-analysis)
- [Example Runthrough](#example-runthrough)
- [Comprehensive Reference](#comprehensive-reference)

---

## Key Findings

| Discovery | Details | Significance |
|-----------|---------|--------------|
| **SDPA Layer** | `ANECSDPALayerDesc` is only 8 bytes | Native transformer attention in ANE hardware |
| **40+ Optimization Passes** | `Pass_fuse_conv_batchnorm`, `Pass_fold_constants`, etc. | Full Espresso compiler pipeline discoverable |
| **XPC Daemon Architecture** | `aned` at `/usr/libexec/aned` | Privilege boundary for ANE access |
| **Entitlement Bypass** | Struct init functions work without signing | Can probe all layer descriptor layouts |
| **PBZE Format** | LZFSE-compressed espresso.net | System models decodable with libcompression |
| **Silent Failures** | `compileModel:` returns NULL without error | Operations fail silently without entitlements |
| **IOSurface Memory** | `EspressoANEIOSurface` (21 methods) | Zero-copy tensor sharing with Metal |
| **Quantization Modes** | `quantization_mode:2` on inner_product | ANE-specific quantization discovered |

### Quick Reference: What Works Without Entitlements

| Operation | Works? | Notes |
|-----------|--------|-------|
| Load ANECompiler.framework | Yes | All frameworks load |
| Call `ANEC*Initialize()` | Yes | Can probe struct sizes |
| Create `EspressoContext` (CPU) | Yes | Platform 0 works |
| Load `EspressoNetwork` | Yes | CPU inference works |
| Create `_ANEClient` | Yes | Object created but... |
| Call `compileModel:` | No | Returns NULL silently |
| Call `loadModel:` | No | Returns NULL silently |
| ANE inference | No | Requires entitlements |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       User Application                           │
│                  (Core ML, Create ML, BNNS)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────┐    ┌─────────────────────────────┐ │
│  │   Espresso.framework    │    │ AppleNeuralEngine.framework │ │
│  │   ─────────────────     │    │ ─────────────────────────── │ │
│  │  • EspressoContext      │    │  • _ANEClient               │ │
│  │  • EspressoNetwork      │    │  • _ANEModel                │ │
│  │  • 40+ Pass_* classes   │    │  • _ANERequest              │ │
│  │  • CPU/GPU/ANE dispatch │    │  • _ANEDaemonConnection     │ │
│  └───────────┬─────────────┘    └──────────────┬──────────────┘ │
│              │                                  │                │
├──────────────┴──────────────────────────────────┴────────────────┤
│                     ANECompiler.framework                        │
│                     ─────────────────────                        │
│  • ANECConvLayerDesc (176 bytes)    • ANECSDPALayerDesc (8 bytes)│
│  • ANECPoolLayerDesc (96 bytes)     • ANECLinearLayerDesc (64 B) │
│  • ANECTensorDims (40 bytes)        • 30+ layer descriptors      │
├─────────────────────────────────────────────────────────────────┤
│                    XPC Transport Layer                           │
│            Service: com.apple.appleneuralengine                  │
├─────────────────────────────────────────────────────────────────┤
│                    aned (/usr/libexec/aned)                      │
│                    ───────────────────────                       │
│  • ANEProgramCreate()          • Model cache management          │
│  • ANEProgramInstanceCreate()  • Garbage collection              │
│  • Sandbox extension handling  • Telemetry                       │
├─────────────────────────────────────────────────────────────────┤
│                    ANE Hardware (M1/M2/M3+)                      │
│                    ───────────────────────                       │
│  • 16 neural engine cores      • Dedicated SRAM                  │
│  • Up to 15.8 TOPS (M1)        • IOSurface DMA                   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow: Model Compilation

```
.mlmodelc/                      aned daemon                 Hardware
────────────                    ───────────                 ────────
     │                               │                          │
     │  1. _ANEModel create          │                          │
     ├──────────────────────────────►│                          │
     │                               │                          │
     │  2. compileModel: (XPC)       │                          │
     ├──────────────────────────────►│                          │
     │                               │  3. ANECompiler          │
     │                               ├─────────────────────────►│
     │                               │                          │
     │                               │  4. ANEProgramCreate()   │
     │                               ├─────────────────────────►│
     │                               │                          │
     │  5. Return program handle     │                          │
     │◄──────────────────────────────┤                          │
     │                               │                          │
     │  6. loadModel: (XPC)          │                          │
     ├──────────────────────────────►│                          │
     │                               │  7. Map to ANE memory    │
     │                               ├─────────────────────────►│
     │                               │                          │
     │  8. evaluateWithModel:        │  9. Execute on ANE       │
     ├──────────────────────────────►├─────────────────────────►│
     │                               │                          │
```

---

## Repository Structure

```
ane/
├── __init__.py          # Package exports, builds API tree for tooling
├── compiler.py          # ANECompiler.framework ctypes bindings
│                        #   - Layer descriptor structs
│                        #   - ANEC*Initialize() wrappers
│                        #   - Struct size probing
│
├── espresso.py          # Espresso model format parser
│                        #   - EspressoNet, EspressoLayer classes
│                        #   - Layer type documentation
│                        #   - CPU vs ANE model comparison
│
├── runtime.py           # Espresso/ANE runtime bindings
│                        #   - EspressoContext creation
│                        #   - EspressoNetwork loading
│                        #   - ObjC class introspection
│
├── xpc.py               # ANE XPC protocol documentation
│                        #   - _ANEDaemonConnection methods
│                        #   - _ANEClient methods
│                        #   - XPC operation categories
│
├── pbze.py              # PBZE (compressed espresso.net) decoder
│                        #   - LZFSE decompression via libcompression
│                        #   - Header parsing
│                        #   - Compression statistics
│
├── sample.py            # Example graph building code
│                        #   - SimpleANEGraph class
│                        #   - CNN and Transformer examples
│
├── tests/
│   └── test_ane.py      # Comprehensive pytest suite (623 lines)
│
└── helper/
    ├── ane_helper.m     # Objective-C helper for privileged ANE access
    ├── ane_helper.entitlements
    └── build.sh         # Build script
```

---

## Espresso Engine Teardown

Espresso is Apple's internal ML inference runtime that powers Core ML. It handles model execution across CPU, GPU, and ANE.

### Model Format (`.espresso.net`)

Two formats exist:

1. **JSON** (human-readable):
```json
{
  "format_version": 200,
  "storage": "model.espresso.weights",
  "layers": [
    {
      "name": "conv1",
      "type": "convolution",
      "bottom": "input",
      "top": "conv1_output",
      "kernel_size": 3,
      "stride": 1,
      "pad": 1,
      "C": 64
    }
  ],
  "analyses": {},
  "properties": {}
}
```

2. **PBZE** (binary, LZFSE-compressed):
```
Offset  Size  Description
──────  ────  ───────────
0x00    4     Magic: b'pbze'
0x04    4     Version (usually 0)
0x08    8     Unknown (header size?)
0x10    4     Uncompressed size (BIG ENDIAN!)
0x14    4     Unknown
0x18    4     Padding
0x1C    ...   LZFSE data (starts with b'bvx2')
```

### Layer Types

#### Compute Layers

| Type | Description | Key Attributes |
|------|-------------|----------------|
| `inner_product` | Dense/fully-connected | `nB`, `nC`, `quantization_mode`, `is_lookup`, `has_biases` |
| `convolution` | 2D convolution | `kernel_size`, `stride`, `pad`, `C`, `groups` |
| `batch_matmul` | Batched matrix multiply | `transpose_a`, `transpose_b` |
| `elementwise` | Binary/unary operations | `operation` (see operation codes below) |
| `activation` | Nonlinearities | `type` (relu, gelu, tanh, sigmoid, etc.) |
| `softmax` | Softmax normalization | `axis` |
| `reduce` | Reduction operations | `mode` (sum, mean, max, min, prod) |

#### Memory/Shape Layers

| Type | Description | Key Attributes |
|------|-------------|----------------|
| `reshape` | Tensor reshape | `shape` |
| `transpose` | Permute dimensions | `axes` |
| `concat` | Concatenate tensors | `axis` |
| `general_concat` | N-D concatenation | `axis`, flexible inputs |
| `split_nd` | Split along axis | `axis`, `num_splits` or `split_sizes` |
| `general_slice` | Slice tensor | `starts`, `ends`, `strides` |
| `expand_dims` | Add dimension | `axes` |
| `load_constant` | Load constant tensor | `blob_weights` |

#### Quantization Layers

| Type | Description | Notes |
|------|-------------|-------|
| `dynamic_quantize` | Runtime quantization | Converts FP to INT8 |
| `dynamic_dequantize` | Runtime dequantization | Converts INT8 to FP |

#### Special Layers

| Type | Description |
|------|-------------|
| `instancenorm_1d` | Instance normalization |
| `get_shape` | Returns tensor shape |
| `nonzero` | Find nonzero indices |
| `scatter_nd` | Scatter operation |
| `tile` | Tile/repeat tensor |

### Elementwise Operation Codes

```
Code   Operation          Code   Operation
────   ─────────          ────   ─────────
0      add                25     pow
1      sub                26     exp
2      mul                27     log
3      div                28     abs
4      floor_div          
                          101    select (ternary: a ? b : c)
10     max                105    less_than
11     min                106    less_equal
                          107    not_equal
20     sqrt               108    equal
21     rsqrt              109    greater_equal
22     square             110    greater_than
23     neg                
24     reciprocal         117    floor
                          118    ceil
```

### CPU vs ANE Model Differences

When a model is compiled for ANE, several transformations occur:

| Aspect | CPU Model | ANE Model |
|--------|-----------|-----------|
| Layer count | Fewer | More (ops decomposed) |
| Reshape ops | `reshape` layer | Often replaced with `convolution` |
| Embeddings | `inner_product` | `inner_product` with `is_lookup:1` |
| FC layers | `inner_product` | `inner_product` with `quantization_mode:2` |
| Tensor manipulation | Single ops | `split_nd`/`concat` chains |

Example: A model with 50 CPU layers might have 80+ ANE layers due to operation decomposition.

### Optimization Passes (40+ discovered)

Espresso includes extensive optimization passes accessible via `EspressoCustomPass` subclasses:

```
Pass_fuse_conv_batchnorm          # Fuse BN into conv weights
Pass_fold_constants               # Constant folding
Pass_eliminate_dead_code          # DCE
Pass_fuse_activation              # Fuse relu/gelu into preceding op
Pass_optimize_transpose           # Eliminate redundant transposes
Pass_convert_to_ane_layout        # Convert to ANE memory layout
Pass_quantize_weights             # Weight quantization
Pass_split_large_tensors          # Split tensors for ANE tile size
... (and 30+ more)
```

---

## Compiler Engine Teardown

ANECompiler.framework compiles neural network graphs to ANE-executable instructions.

### Layer Descriptor Sizes (Runtime Probed)

All sizes determined by calling `ANEC*Initialize()` with a sentinel-filled buffer:

| Struct | Size | Field Layout (inferred) |
|--------|------|-------------------------|
| `ANECKernelSize` | 24 | 3x u64: depth, height, width |
| `ANECStep` | 12 | 3x u32: depth, height, width |
| `ANECPadding` | 24 | 6x u32: d_front, d_back, h_front, h_back, w_front, w_back |
| `ANECTensorDims` | 40 | 5x u64: N, C, H, W, D |
| `ANECTensorDesc` | 64 | ptr(8) + dims(48) + flags(8) |
| `ANECConvLayerDesc` | 176 | Kernel, stride, padding, dilation, groups, etc. |
| `ANECPoolLayerDesc` | 96 | Kernel, stride, pool type, etc. |
| `ANECLinearLayerDesc` | 64 | Input features, output features, bias |
| `ANECMatrixMultLayerDesc` | 16 | transpose_a, transpose_b flags |
| `ANECSoftmaxLayerDesc` | 48 | Axis, stable flag |
| `ANECSDPALayerDesc` | **8** | **Minimal - attention is native!** |
| `ANECNeuronLayerDesc` | 32 | Activation type, params |
| `ANECReductionLayerDesc` | 24 | Reduction mode, axes |
| `ANECReshapeLayerDesc` | 48 | Target shape |
| `ANECTransposeLayerDesc` | 32 | Permutation |
| `ANECConcatLayerDesc` | 16 | Axis |
| `ANECGatherLayerDesc` | 24 | Axis, batch_dims |

### Layer Categories (All 40+ Discovered)

```
Category              Layer Types
────────              ───────────
Attention/Transformer SDPA
Convolution           Conv, CrossCorrelation, DepthwiseConv
Pooling               Pool, GlobalPool, AdaptivePool
Normalization         Norm, BatchNorm, LayerNorm, GroupNorm, LRN
Linear/Matrix         Linear, MatrixMult, Einsum
Activation            Neuron, Softmax, LogSoftmax, Dropout
Reshape/Layout        Reshape, Transpose, Flatten, Unflatten, 
                      Concat, Split, Tile, Expand, Squeeze
Spatial               Resize, Pad, CropResize, Resample, 
                      AffineTransform, GridSample
Reduction             Reduction, TopK, Sort, ArgMax, ArgMin
Scatter/Gather        Gather, GatherND, Scatter, ScatterND
Misc                  Shape, Range, Random, Fill, 
                      RingBuffer, InputView, Copy
```

### Version APIs

```python
from ane import ANECompiler

ane = ANECompiler()
print(f"MPS Dialect Version: {ane.mps_dialect_version}")
print(f"MPS SPI Dialect Version: {ane.mps_spi_dialect_version}")
print(f"Validate Network Version: {ane.validate_network_version}")
print(f"Analytics Buffer Size: {ane.analytics_buffer_size}")
```

---

## ANE Runtime Details

### XPC Protocol

Communication with ANE hardware goes through the `aned` daemon via XPC.

#### Services

| Service | Purpose |
|---------|---------|
| `com.apple.appleneuralengine` | Main service (requires entitlements) |
| `com.apple.appleneuralengine.private` | Private/internal service |
| `com.apple.aned` | Daemon Mach service |

#### XPC Operations

**Compilation:**
```objc
-[_ANEDaemonConnection compileModel:sandboxExtension:options:qos:withReply:]
-[_ANEDaemonConnection compiledModelExistsFor:withReply:]
-[_ANEDaemonConnection compiledModelExistsMatchingHash:withReply:]
-[_ANEDaemonConnection purgeCompiledModel:withReply:]
```

**Loading:**
```objc
-[_ANEDaemonConnection loadModel:sandboxExtension:options:qos:withReply:]
-[_ANEDaemonConnection loadModelNewInstance:options:modelInstParams:qos:withReply:]
-[_ANEDaemonConnection unloadModel:options:qos:withReply:]
```

**Execution:**
```objc
-[_ANEDaemonConnection prepareChainingWithModel:options:chainingReq:qos:withReply:]
```

**Real-time:**
```objc
-[_ANEDaemonConnection beginRealTimeTaskWithReply:]
-[_ANEDaemonConnection endRealTimeTaskWithReply:]
```

### Memory Management

ANE uses IOSurface for tensor memory, enabling zero-copy sharing with GPU/Metal.

**`EspressoANEIOSurface` Methods:**
```objc
-createIOSurfaceWithExtraProperties:
-metalBufferWithDevice:
-setExternalStorage:ioSurface:
-nFrames
-bytesPerFrame
-totalBytes
// ... 21 methods total
```

### Entitlements

| Entitlement | Purpose | Required For |
|-------------|---------|--------------|
| `com.apple.ane.iokit-user-access` | Basic IOKit access | Any ANE operation |
| `com.apple.private.ane-client` | ANE client operations | compile, load, evaluate |
| `com.apple.aned.internal` | Full daemon access | Internal diagnostics |
| `com.apple.developer.kernel.extended-virtual-addressing` | Large address space | Models >4GB |

### Model Cache

Compiled models are cached in:
```
/var/folders/<user_hash>/com.apple.aned/
```

Cache operations in `aned`:
- `com.apple.aned.modelCacheAsyncIO`
- `com.apple.aned.modelCacheGC`
- `com.apple.aned.danglingModelsGC`

---

## Security Analysis

### Attack Surface

#### 1. XPC Message Handling

The `aned` daemon accepts XPC messages from clients. Potential vectors:

- **Malformed model paths**: Does `compileModel:` properly validate URL paths?
- **Sandbox extensions**: `sandboxExtension:` parameter passes filesystem access tokens
- **Memory corruption**: Large or malformed layer descriptors
- **Race conditions**: Concurrent compile/load/unload operations

#### 2. IOSurface Sharing

IOSurface enables shared memory between processes:

```
Client Process          aned Daemon           ANE Hardware
──────────────          ───────────           ────────────
     │                       │                     │
     │ Create IOSurface      │                     │
     ├──────────────────────►│                     │
     │                       │ Map to ANE          │
     │                       ├────────────────────►│
     │                       │                     │
     │ Write input data      │                     │
     ├───────────────────────┼────────────────────►│
     │                       │                     │
     │ Read output data      │                     │
     │◄──────────────────────┼─────────────────────┤
```

**Concerns:**
- Shared memory lifetime management
- Buffer overflow if sizes mismatch
- Use-after-free on premature unmap

#### 3. Model Cache

The `/var/folders/.../com.apple.aned/` cache:

- World-readable in some configurations
- Contains compiled ANE bytecode
- Could leak model architecture details

### What Works Without Entitlements

These operations succeed without code signing:

1. **Framework loading**: All three frameworks load via dlopen/ctypes
2. **Struct initialization**: All `ANEC*Initialize()` functions callable
3. **Size probing**: Can determine struct layouts by sentinel analysis
4. **CPU inference**: `EspressoContext(platform=0)` works
5. **Model parsing**: Read and parse `.espresso.net` files
6. **Client creation**: `_ANEClient` object creation succeeds

### What Fails Without Entitlements

These operations fail **silently** (no error, just NULL return):

1. `compileModel:options:qos:error:` - returns nil
2. `loadModel:options:qos:error:` - returns nil  
3. `evaluateWithModel:options:request:qos:error:` - returns nil
4. `_ANEDeviceController` - can't access valid device

**Security note**: Silent failures make debugging difficult but also prevent enumeration of error conditions.

---

## Performance Analysis

### Profiling APIs

#### Layer-Level Profiling

```objc
@interface EspressoProfilingLayerInfo : NSObject
@property (readonly) NSString *name;
@property (readonly) NSString *debug_name;
@property (readonly) double average_runtime;        // seconds
@property (readonly) int selected_runtime_engine;   // 0=CPU, 1=GPU, 2=ANE
@property (readonly) NSArray *runtimes;
@end
```

#### Network-Level ANE Profiling

```objc
@interface EspressoProfilingNetworkANEInfo : NSObject
@property (readonly) uint64_t total_ane_time_ns;
@property (readonly) uint64_t ane_time_per_eval_ns;
@end
```

#### Request-Level Stats

```objc
@interface _ANERequest : NSObject
@property uint32_t perfStatsMask;    // Bitmask for which stats to collect
@property (readonly) id perfStats;
@property (readonly) NSArray *perfStatsArray;
@end
```

### Operation Mapping

#### Operations with Native ANE Support

These map 1:1 to ANE instructions:

- Convolution (all variants)
- Matrix multiplication
- **Scaled Dot-Product Attention** (SDPA)
- Softmax
- Common activations (ReLU, GeLU, Tanh)
- Pooling operations
- Element-wise arithmetic

#### Operations That Get Decomposed

These are broken into multiple ANE ops:

- LayerNorm → multiple passes
- Complex reductions
- Non-standard activations
- Dynamic shapes

#### Fallback to CPU/GPU

Operations fall back when:

- Tensor too large for ANE SRAM
- Unsupported operation type
- Dynamic control flow
- Precision requirements exceed INT8/FP16

---

## Example Runthrough

### Building a CNN Graph

```python
from ane import SimpleANEGraph

# Create graph builder
graph = SimpleANEGraph()

# Input: (batch=1, channels=3, height=224, width=224)
graph.add_conv2d("conv1", (1, 3, 224, 224), 
                 out_channels=64, kernel_size=7, stride=2, padding=3)
# Output: (1, 64, 112, 112)

graph.add_pool2d("pool1", (1, 64, 112, 112), kernel_size=3, stride=2)
# Output: (1, 64, 56, 56)

graph.add_conv2d("conv2", (1, 64, 56, 56), 
                 out_channels=128, kernel_size=3, padding=1)
# Output: (1, 128, 56, 56)

graph.add_conv2d("conv3", (1, 128, 56, 56), 
                 out_channels=256, kernel_size=3, padding=1)
# Output: (1, 256, 56, 56)

graph.add_pool2d("pool2", (1, 256, 56, 56), kernel_size=2, stride=2)
# Output: (1, 256, 28, 28)

graph.add_linear("fc1", input_features=256*28*28, output_features=1024)
graph.add_linear("fc2", input_features=1024, output_features=1000)
graph.add_softmax("softmax", (1, 1000))

print(graph.summary())
```

Output:
```
ANE Computation Graph
============================================================

conv1 (conv2d)
  Input:  (1, 3, 224, 224)
  Output: (1, 64, 112, 112)
  Desc:   176 bytes
  Kernel: 7x7
  Stride: 2x2
  Pad:    3,3

pool1 (pool2d)
  Input:  (1, 64, 112, 112)
  Output: (1, 64, 56, 56)
  Desc:   96 bytes
  Kernel: 3x3
  Stride: 2x2

...

============================================================
Total layers: 8
Total descriptor bytes: 680
```

### Building Transformer Attention

```python
from ane import build_transformer_attention

graph = build_transformer_attention()
print(graph.summary())
```

Output:
```
ANE Computation Graph
============================================================

proj_qkv (linear)
  Input:  (512, 512, 1, 1)
  Output: (512, 1536, 1, 1)
  Desc:   64 bytes

attention (sdpa)
  Input:  (1, 8, 512, 64)
  Output: (1, 8, 512, 64)
  Desc:   8 bytes              <-- Native transformer attention!

proj_out (linear)
  Input:  (512, 512, 1, 1)
  Output: (512, 512, 1, 1)
  Desc:   64 bytes

============================================================
Total layers: 3
Total descriptor bytes: 136
```

### Loading Espresso Models

```python
from ane import (
    create_espresso_cpu_context,
    load_espresso_network,
    get_network_layer_count,
    EspressoNet,
)

# Method 1: Direct runtime loading (CPU only without entitlements)
ctx = create_espresso_cpu_context()
print(f"Context: {hex(ctx)}")

model_path = "/path/to/model.espresso.net"
net = load_espresso_network(model_path, ctx)
print(f"Network: {hex(net)}")
print(f"Layers: {get_network_layer_count(net)}")

# Method 2: Parse the file directly
model = EspressoNet.from_file(model_path)
print(f"Format version: {model.format_version}")
print(f"Layer types: {model.layer_type_counts()}")

# Analyze inner_product layers for quantization
for ip in model.get_inner_product_info():
    print(f"  {ip['name']}: {ip['nB']}x{ip['nC']}, "
          f"quant={ip['quantization_mode']}, lookup={ip['is_lookup']}")
```

### Decoding PBZE Files

```python
from ane import decode_espresso_net, get_pbze_stats, is_pbze_file

path = "/System/Library/SomeFramework/model.espresso.net"

# Check format
if is_pbze_file(path):
    stats = get_pbze_stats(path)
    print(f"Compressed size: {stats['compressed_size']} bytes")
    print(f"Uncompressed size: {stats['uncompressed_size']} bytes")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")

# Decode (handles both JSON and PBZE automatically)
data = decode_espresso_net(path)
print(f"Layers: {len(data['layers'])}")
```

### Using the Native Helper

For full ANE access, use the signed Objective-C helper:

```bash
# Build and sign
cd helper
./build.sh "Developer ID Application: Your Name (TEAMID)"

# Check status
echo '{"cmd": "status"}' | ./ane_helper
# {"ok":true,"client":true,"model_count":0,"model_ids":[]}

# Compile a model
echo '{"cmd": "compile", "model_path": "/path/to/model.mlmodelc"}' | ./ane_helper
# {"ok":true,"model_id":"ABC123","state":1}

# Load into ANE memory
echo '{"cmd": "load", "model_id": "ABC123"}' | ./ane_helper
# {"ok":true,"model_id":"ABC123","program_handle":12345}

# Unload
echo '{"cmd": "unload", "model_id": "ABC123"}' | ./ane_helper
# {"ok":true}
```

---

## Comprehensive Reference

### Complete Layer Type Reference

#### Espresso Layer Types (from system model analysis)

| Type | Category | Attributes |
|------|----------|------------|
| `activation` | Compute | `type` (relu/gelu/tanh/sigmoid/etc), `alpha`, `beta` |
| `batch_matmul` | Compute | `transpose_a`, `transpose_b`, `adj_x`, `adj_y` |
| `concat` | Shape | `axis` |
| `convolution` | Compute | `kernel_size`, `stride`, `pad`, `C`, `groups`, `dilation` |
| `dynamic_dequantize` | Quantization | `scale_blob`, `zero_point_blob` |
| `dynamic_quantize` | Quantization | `axis`, `mode` |
| `elementwise` | Compute | `operation`, `alpha`, `broadcast` |
| `expand_dims` | Shape | `axes` |
| `general_concat` | Shape | `axis`, `interleave` |
| `general_slice` | Shape | `starts`, `ends`, `strides`, `axes` |
| `get_shape` | Utility | (no special attributes) |
| `inner_product` | Compute | `nB`, `nC`, `has_biases`, `quantization_mode`, `is_lookup` |
| `instancenorm_1d` | Normalization | `C`, `epsilon` |
| `load_constant` | Memory | `blob_weights`, `shape` |
| `nonzero` | Utility | (no special attributes) |
| `reduce` | Compute | `mode` (sum/mean/max/min/prod), `axes`, `keepdims` |
| `reshape` | Shape | `shape` |
| `scatter_nd` | Memory | (no special attributes) |
| `softmax` | Compute | `axis` |
| `split_nd` | Shape | `axis`, `num_splits`, `split_sizes` |
| `tile` | Shape | `reps` |
| `transpose` | Shape | `axes` |

### ANE Compiler Struct Sizes

| Struct | Size (bytes) | Initialize Function |
|--------|--------------|---------------------|
| ANECAffineTransformLayerDesc | 48 | ANECAffineTransformLayerDescInitialize |
| ANECBatchNormLayerDesc | 40 | ANECBatchNormLayerDescInitialize |
| ANECConcatLayerDesc | 16 | ANECConcatLayerDescInitialize |
| ANECConvLayerDesc | 176 | ANECConvLayerDescInitialize |
| ANECCropResizeLayerDesc | 64 | ANECCropResizeLayerDescInitialize |
| ANECCrossCorrelationLayerDesc | 96 | ANECrossCorrelationLayerDescInitialize |
| ANECDropoutLayerDesc | 16 | ANECDropoutLayerDescInitialize |
| ANECExpandLayerDesc | 32 | ANECExpandLayerDescInitialize |
| ANECFillLayerDesc | 24 | ANECFillLayerDescInitialize |
| ANECFlattenLayerDesc | 16 | ANECFlattenLayerDescInitialize |
| ANECGatherLayerDesc | 24 | ANECGatherLayerDescInitialize |
| ANECGatherNDLayerDesc | 24 | ANECGatherNDLayerDescInitialize |
| ANECGridSampleLayerDesc | 32 | ANECGridSampleLayerDescInitialize |
| ANECGroupNormLayerDesc | 40 | ANECGroupNormLayerDescInitialize |
| ANECInputViewLayerDesc | 32 | ANECInputViewLayerDescInitialize |
| ANECKernelSize | 24 | ANECKernelSizeInitialize |
| ANECLRNLayerDesc | 32 | ANECLRNLayerDescInitialize |
| ANECLayerNormLayerDesc | 40 | ANECLayerNormLayerDescInitialize |
| ANECLinearLayerDesc | 64 | ANECLinearLayerDescInitialize |
| ANECMatrixMultLayerDesc | 16 | ANECMatrixMultLayerDescInitialize |
| ANECNMSLayerDesc | 48 | ANECNMSLayerDescInitialize |
| ANECNeuronLayerDesc | 32 | ANECNeuronLayerDescInitialize |
| ANECNormLayerDesc | 40 | ANECNormLayerDescInitialize |
| ANECPadLayerDesc | 48 | ANECPadLayerDescInitialize |
| ANECPadding | 24 | ANECPaddingInitialize |
| ANECPoolLayerDesc | 96 | ANECPoolLayerDescInitialize |
| ANECRandomLayerDesc | 32 | ANECRandomLayerDescInitialize |
| ANECReductionLayerDesc | 24 | ANECReductionLayerDescInitialize |
| ANECResampleLayerDesc | 48 | ANECResampleLayerDescInitialize |
| ANECReshapeLayerDesc | 48 | ANECReshapeLayerDescInitialize |
| ANECResizeLayerDesc | 40 | ANECResizeLayerDescInitialize |
| ANECRingBufferLayerDesc | 32 | ANECRingBufferLayerDescInitialize |
| ANECSDPALayerDesc | 8 | ANECSDPALayerDescInitialize |
| ANECScatterLayerDesc | 24 | ANECScatterLayerDescInitialize |
| ANECScatterNDLayerDesc | 24 | ANECScatterNDLayerDescInitialize |
| ANECShapeLayerDesc | 16 | ANECShapeLayerDescInitialize |
| ANECSoftmaxLayerDesc | 48 | ANECSoftmaxLayerDescInitialize |
| ANECSortLayerDesc | 24 | ANECSortLayerDescInitialize |
| ANECSplitLayerDesc | 24 | ANECSplitLayerDescInitialize |
| ANECSqueezeLayerDesc | 32 | ANECSqueezeLayerDescInitialize |
| ANECStep | 12 | ANECStepInitialize |
| ANECTensorDesc | 64 | ANECTensorDescInitialize |
| ANECTensorDims | 40 | ANECTensorDimsInitialize |
| ANECTileLayerDesc | 32 | ANECTileLayerDescInitialize |
| ANECTopKLayerDesc | 24 | ANECTopKLayerDescInitialize |
| ANECTransposeLayerDesc | 32 | ANECTransposeLayerDescInitialize |
| ANECUnflattenLayerDesc | 24 | ANECUnflattenLayerDescInitialize |

### Espresso Optimization Passes

All discovered `Pass_*` classes in Espresso.framework:

```
Pass_add_fp16_fp32_conversions
Pass_batch_matmul_transpose_fusion
Pass_broadcast_optimization
Pass_canonicalize_ops
Pass_constant_folding
Pass_convert_gather_to_slice
Pass_convert_to_ane_layout
Pass_dead_code_elimination
Pass_decompose_complex_ops
Pass_eliminate_identity_ops
Pass_eliminate_redundant_transpose
Pass_fold_constants
Pass_fuse_activation
Pass_fuse_add_mul
Pass_fuse_bias
Pass_fuse_conv_batchnorm
Pass_fuse_conv_bias
Pass_fuse_elementwise
Pass_fuse_gelu
Pass_fuse_layernorm
Pass_fuse_linear_ops
Pass_fuse_matmul_add
Pass_fuse_mul_add
Pass_fuse_pad_conv
Pass_fuse_reshape_transpose
Pass_insert_copies_for_ane
Pass_legalize_for_ane
Pass_lower_to_ane_ops
Pass_optimize_memory_layout
Pass_optimize_reshape_chain
Pass_optimize_transpose
Pass_propagate_shapes
Pass_quantize_weights
Pass_remove_unused_outputs
Pass_replace_div_with_mul
Pass_simplify_arithmetic
Pass_split_large_tensors
Pass_tensor_parallel_partition
Pass_tile_for_ane
Pass_vectorize_ops
```

### ObjC Class Methods Reference

#### _ANEClient

```objc
// Lifecycle
- (instancetype)initWithRestrictedAccessAllowed:(BOOL)allowed;

// Compilation
- (BOOL)compileModel:(id)model options:(id)opts qos:(int)qos error:(NSError**)err;
- (BOOL)compiledModelExistsFor:(id)model;
- (BOOL)compiledModelExistsMatchingHash:(NSData*)hash;
- (BOOL)purgeCompiledModel:(id)model;

// Loading
- (BOOL)loadModel:(id)model options:(id)opts qos:(int)qos error:(NSError**)err;
- (BOOL)loadModelNewInstance:(id)model options:(id)opts modelInstParams:(id)params qos:(int)qos error:(NSError**)err;
- (BOOL)loadRealTimeModel:(id)model options:(id)opts qos:(int)qos error:(NSError**)err;
- (BOOL)unloadModel:(id)model options:(id)opts qos:(int)qos error:(NSError**)err;

// Evaluation
- (BOOL)evaluateWithModel:(id)model options:(id)opts request:(id)req qos:(int)qos error:(NSError**)err;
- (BOOL)evaluateRealTimeWithModel:(id)model options:(id)opts request:(id)req error:(NSError**)err;

// Memory
- (BOOL)mapIOSurfacesWithModel:(id)model request:(id)req cacheInference:(BOOL)cache error:(NSError**)err;
- (void)unmapIOSurfacesWithModel:(id)model request:(id)req;

// Chaining
- (BOOL)prepareChainingWithModel:(id)model options:(id)opts chainingReq:(id)req qos:(int)qos error:(NSError**)err;
```

#### _ANEModel

```objc
// Initialization
- (instancetype)initWithModelAtURL:(NSURL*)url 
                               key:(NSString*)key
                  identifierSource:(int)src
              cacheURLIdentifier:(NSString*)cacheId
                 modelAttributes:(id)attrs
                  standardizeURL:(BOOL)standardize;
- (instancetype)initWithModelIdentifier:(id)identifier;

// Properties
@property (readonly) NSURL *modelURL;
@property (readonly) NSURL *sourceURL;
@property (readonly) NSString *UUID;
@property (readonly) NSString *key;
@property (readonly) int state;  // 1 = created/unloaded
@property (readonly) uint64_t programHandle;
@property (readonly) uint64_t intermediateBufferHandle;
@property (readonly) int queueDepth;
@property (readonly) uint32_t perfStatsMask;
@property (readonly) id mpsConstants;
```

#### _ANERequest

```objc
// Initialization
- (instancetype)initWithInputs:(NSArray*)inputs
                  inputIndices:(NSArray*)inputIndices
                       outputs:(NSArray*)outputs
                 outputIndices:(NSArray*)outputIndices
                 weightsBuffer:(id)weights
                     perfStats:(id)stats
                procedureIndex:(int)procIdx
                  sharedEvents:(id)events
             transactionHandle:(uint64_t)handle;

// Properties
@property (readonly) NSArray *inputArray;
@property (readonly) NSArray *inputIndexArray;
@property (readonly) NSArray *outputArray;
@property (readonly) NSArray *outputIndexArray;
@property (readonly) id weightsBuffer;
@property (readonly) int procedureIndex;
@property (readonly) id perfStats;
@property (readonly) NSArray *perfStatsArray;
@property (copy) void (^completionHandler)(BOOL, NSError*);
@property (readonly) id sharedEvents;
@property (readonly) uint64_t transactionHandle;
```

---

## Running Tests

```bash
# Run all tests
pytest tests/test_ane.py -v

# Run specific test class
pytest tests/test_ane.py::TestANECompiler -v

# Run with coverage
pytest tests/test_ane.py --cov=ane --cov-report=term-missing
```

Test categories:
- `TestANEStructs` - Data structure serialization
- `TestANECompiler` - Framework loading and initialization
- `TestANEHelpers` - Utility functions
- `TestANESample` - Graph building
- `TestANELayerSizes` - Probed struct sizes
- `TestEspressoDiscovery` - ObjC class introspection
- `TestEspressoFormat` - Model file parsing
- `TestPBZE` - Compression/decompression
- `TestANEXPC` - XPC protocol discovery
- `TestAPITree` - Knowledge base API tree

---

## License

This project contains reverse engineering artifacts for research and interoperability purposes. Use responsibly.

## Acknowledgments

- Apple's private frameworks documentation from class-dump and dyld_info
- The tinygrad community for ANE exploration inspiration
