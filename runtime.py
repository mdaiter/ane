# ANE Runtime exploration
# Documents the Espresso and AppleNeuralEngine frameworks for model loading/inference
"""
ANE Runtime Architecture
========================

Apple's Neural Engine stack has three main private frameworks:

1. ANECompiler.framework - Compiles neural network graphs to ANE instructions
   - ANECCompile*, ANECValidate* functions
   - Layer descriptor structs (ANECConvLayerDesc, etc.)
   - Works with "espresso.net" JSON format

2. Espresso.framework - ML inference runtime (CPU/GPU/ANE)
   - EspressoContext: platform context (0=CPU, 1=GPU, 2=ANE)
   - EspressoNetwork: loads espresso.net files
   - Handles model execution on selected platform

3. AppleNeuralEngine.framework - Low-level ANE hardware interface  
   - _ANEClient: XPC connection to ANE daemon
   - _ANEModel: represents a compiled model
   - _ANEDeviceController: hardware access
   - _ANEDaemonConnection: IPC to aned service

Model Formats
=============

espresso.net - JSON format describing network graph:
{
  "storage": "model.espresso.weights",  // weights file
  "format_version": 200,
  "layers": [
    {"type": "inner_product", "name": "fc1", "bottom": "input", "top": "output", ...},
    {"type": "conv", ...},
  ]
}

espresso.shape - Shape metadata
espresso.weights - Binary weights data

mlmodelc/ - CoreML compiled model directory:
  - model.espresso.net
  - model.espresso.shape
  - model.espresso.weights
  - metadata.json
  - coremldata.bin

Loading a Model
===============

Via Espresso (simpler, higher-level):

```objc
EspressoContext *ctx = [[EspressoContext alloc] initWithPlatform:0];  // 0=CPU
EspressoNetwork *net = [[EspressoNetwork alloc] 
    initWithJSFile:"/path/to/model.espresso.net"  // C string, not NSString!
    context:ctx 
    computePath:0];  // int flag, not path string
```

Via AppleNeuralEngine (lower-level, requires entitlements):

```objc
_ANEClient *client = [[_ANEClient alloc] initWithRestrictedAccessAllowed:YES];
_ANEModel *model = [[_ANEModel alloc] 
    initWithModelAtURL:modelURL 
    key:@"key" 
    identifierSource:0 
    cacheURLIdentifier:nil 
    modelAttributes:nil 
    standardizeURL:YES];

[client compileModel:model options:nil qos:0 error:&error];
[client loadModel:model options:nil qos:0 error:&error];
[client evaluateWithModel:model options:nil request:request qos:0 error:&error];
```

Key Findings
============

1. EspressoContext platforms:
   - 0: CPU (works)
   - 1: GPU (requires Metal setup, may hang)
   - 2: ANE (requires entitlements, may hang)

2. _ANEClient requires entitlements to fully work:
   - com.apple.aned.internal
   - com.apple.private.ane-client

3. initWithJSFile:context:computePath: signature:
   - Takes C string (const char*) for path, NOT NSString
   - computePath is int flag, not a path

4. Model state values (from _ANEModel.state):
   - 1: Created/unloaded
   - (others TBD)

5. ANEDeviceStruct layout:
   - 3x void* pointers
   - 1x char
   - 1x int  
   - 1x uint64

What Works Without Entitlements
================================

1. ANECompiler layer descriptor initialization:
   - All ANEC*Initialize functions work
   - Can probe struct sizes and layouts
   - Can build layer graphs in memory

2. Espresso CPU inference:
   - EspressoContext with platform 0
   - EspressoNetwork loading

3. _ANEModel creation:
   - Can create model objects
   - Can inspect URLs, UUIDs, state

4. _ANEClient creation:
   - Can create with initWithRestrictedAccessAllowed:YES
   - Gets valid connection object
   - compile/load calls return NULL (no error) without proper entitlements

What Requires Entitlements  
===========================

1. Actual ANE compilation
2. Model loading to ANE
3. ANE inference
4. _ANEDeviceController with valid device
"""
from __future__ import annotations
import ctypes
from dataclasses import dataclass
from typing import Optional, Callable, Any

# ObjC runtime bindings for exploring AppleNeuralEngine
_objc: Optional[ctypes.CDLL] = None
_ane: Optional[ctypes.CDLL] = None
_espresso: Optional[ctypes.CDLL] = None


def _load_objc():
  global _objc
  if _objc is None:
    _objc = ctypes.CDLL("/usr/lib/libobjc.dylib")
  return _objc


def _load_ane():
  global _ane
  if _ane is None:
    _ane = ctypes.CDLL("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine")
  return _ane


def _load_espresso():
  global _espresso
  if _espresso is None:
    _espresso = ctypes.CDLL("/System/Library/PrivateFrameworks/Espresso.framework/Espresso")
  return _espresso


def _make_msg_send(ret_type, arg_types):
  """Create an objc_msgSend function with specific signature."""
  objc = _load_objc()
  fn = ctypes.CFUNCTYPE(ret_type, *arg_types)(("objc_msgSend", objc))
  return fn


# Common message send variants
_msg_id_id: Optional[Callable] = None
_msg_id_id_id: Optional[Callable] = None
_msg_id_id_cstr: Optional[Callable] = None
_msg_cstr_id: Optional[Callable] = None
_msg_int_id: Optional[Callable] = None
_msg_bool_id: Optional[Callable] = None


def _init_msg_sends():
  global _msg_id_id, _msg_id_id_id, _msg_id_id_cstr, _msg_cstr_id, _msg_int_id, _msg_bool_id
  if _msg_id_id is not None:
    return
  
  _msg_id_id = _make_msg_send(ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p])
  _msg_id_id_id = _make_msg_send(ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p])
  _msg_id_id_cstr = _make_msg_send(ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p])
  _msg_cstr_id = _make_msg_send(ctypes.c_char_p, [ctypes.c_void_p, ctypes.c_void_p])
  _msg_int_id = _make_msg_send(ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p])
  _msg_bool_id = _make_msg_send(ctypes.c_bool, [ctypes.c_void_p, ctypes.c_void_p])


def _objc_class(name: bytes) -> int:
  """Get ObjC class by name."""
  objc = _load_objc()
  objc_getClass = objc.objc_getClass
  objc_getClass.argtypes = [ctypes.c_char_p]
  objc_getClass.restype = ctypes.c_void_p
  return objc_getClass(name)


def _sel(name: bytes) -> int:
  """Get selector by name."""
  objc = _load_objc()
  sel_registerName = objc.sel_registerName
  sel_registerName.argtypes = [ctypes.c_char_p]
  sel_registerName.restype = ctypes.c_void_p
  return sel_registerName(name)


@dataclass
class EspressoInfo:
  """Information about Espresso framework availability."""
  available: bool
  network_class: int = 0
  context_class: int = 0


@dataclass
class ANERuntimeInfo:
  """Information about AppleNeuralEngine framework."""
  available: bool
  client_class: int = 0
  model_class: int = 0
  device_controller_class: int = 0
  daemon_connection_class: int = 0


def get_espresso_info() -> EspressoInfo:
  """Check if Espresso framework is available and get class pointers."""
  try:
    _load_espresso()
    _init_msg_sends()
    
    network = _objc_class(b"EspressoNetwork")
    context = _objc_class(b"EspressoContext")
    
    return EspressoInfo(
      available=bool(network and context),
      network_class=network,
      context_class=context,
    )
  except Exception:
    return EspressoInfo(available=False)


def get_ane_runtime_info() -> ANERuntimeInfo:
  """Check if AppleNeuralEngine framework is available."""
  try:
    _load_ane()
    _init_msg_sends()
    
    client = _objc_class(b"_ANEClient")
    model = _objc_class(b"_ANEModel")
    device = _objc_class(b"_ANEDeviceController")
    daemon = _objc_class(b"_ANEDaemonConnection")
    
    return ANERuntimeInfo(
      available=bool(client and model),
      client_class=client,
      model_class=model,
      device_controller_class=device,
      daemon_connection_class=daemon,
    )
  except Exception:
    return ANERuntimeInfo(available=False)


def create_espresso_cpu_context() -> Optional[int]:
  """Create an Espresso context for CPU execution.
  
  Returns:
    Object pointer or None if failed
  """
  try:
    _load_espresso()
    _init_msg_sends()
    
    ctx_class = _objc_class(b"EspressoContext")
    if not ctx_class:
      return None
    
    # alloc
    ctx = _msg_id_id(ctx_class, _sel(b"alloc"))
    if not ctx:
      return None
    
    # initWithPlatform:0 (CPU)
    msg_init_platform = _make_msg_send(ctypes.c_void_p, [
      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
    ])
    ctx = msg_init_platform(ctx, _sel(b"initWithPlatform:"), 0)
    
    return ctx
  except Exception:
    return None


def load_espresso_network(espresso_net_path: str, context: int) -> Optional[int]:
  """Load an Espresso network from .espresso.net file.
  
  Args:
    espresso_net_path: Path to .espresso.net file (not directory)
    context: Espresso context from create_espresso_cpu_context()
    
  Returns:
    Network object pointer or None if failed
  """
  try:
    _load_espresso()
    _init_msg_sends()
    
    net_class = _objc_class(b"EspressoNetwork")
    if not net_class:
      return None
    
    # alloc
    net = _msg_id_id(net_class, _sel(b"alloc"))
    if not net:
      return None
    
    # initWithJSFile:context:computePath:
    # Note: takes C string, not NSString; computePath is int flag
    msg_init = _make_msg_send(ctypes.c_void_p, [
      ctypes.c_void_p, ctypes.c_void_p,
      ctypes.c_char_p,  # path as C string
      ctypes.c_void_p,  # context
      ctypes.c_int,     # computePath flag
    ])
    
    net = msg_init(net, _sel(b"initWithJSFile:context:computePath:"),
                   espresso_net_path.encode(), context, 0)
    
    return net
  except Exception:
    return None


def get_network_layer_count(network: int) -> int:
  """Get number of layers in Espresso network."""
  try:
    _init_msg_sends()
    msg_long = _make_msg_send(ctypes.c_long, [ctypes.c_void_p, ctypes.c_void_p])
    return msg_long(network, _sel(b"layers_size"))
  except Exception:
    return -1


# ============================================================================
# Espresso Class Discovery
# ============================================================================

def list_espresso_classes() -> list[str]:
  """Get all Espresso-prefixed ObjC classes."""
  _load_espresso()
  objc = _load_objc()
  
  objc_getClassList = objc.objc_getClassList
  objc_getClassList.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
  objc_getClassList.restype = ctypes.c_int
  
  class_getName = objc.class_getName
  class_getName.argtypes = [ctypes.c_void_p]
  class_getName.restype = ctypes.c_char_p
  
  count = objc_getClassList(None, 0)
  classes = (ctypes.c_void_p * count)()
  objc_getClassList(classes, count)
  
  espresso_classes = []
  for i in range(count):
    if classes[i]:
      name = class_getName(classes[i])
      if name and name.startswith(b"Espresso"):
        espresso_classes.append(name.decode())
  
  return sorted(espresso_classes)


def list_class_methods(class_name: str) -> list[tuple[str, str]]:
  """Get all methods of an ObjC class with their type encodings."""
  _load_espresso()
  objc = _load_objc()
  
  objc_getClass = objc.objc_getClass
  objc_getClass.argtypes = [ctypes.c_char_p]
  objc_getClass.restype = ctypes.c_void_p
  
  class_copyMethodList = objc.class_copyMethodList
  class_copyMethodList.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
  class_copyMethodList.restype = ctypes.POINTER(ctypes.c_void_p)
  
  method_getName = objc.method_getName
  method_getName.argtypes = [ctypes.c_void_p]
  method_getName.restype = ctypes.c_void_p
  
  sel_getName = objc.sel_getName
  sel_getName.argtypes = [ctypes.c_void_p]
  sel_getName.restype = ctypes.c_char_p
  
  method_getTypeEncoding = objc.method_getTypeEncoding
  method_getTypeEncoding.argtypes = [ctypes.c_void_p]
  method_getTypeEncoding.restype = ctypes.c_char_p
  
  cls = objc_getClass(class_name.encode())
  if not cls:
    return []
  
  count = ctypes.c_uint(0)
  methods = class_copyMethodList(cls, ctypes.byref(count))
  
  results = []
  for i in range(count.value):
    sel = method_getName(methods[i])
    name = sel_getName(sel)
    encoding = method_getTypeEncoding(methods[i])
    results.append((name.decode(), encoding.decode() if encoding else ""))
  
  return results


# Exported functions for quick testing
__all__ = [
  "get_espresso_info",
  "get_ane_runtime_info", 
  "create_espresso_cpu_context",
  "load_espresso_network",
  "get_network_layer_count",
  "list_espresso_classes",
  "list_class_methods",
  "EspressoInfo",
  "ANERuntimeInfo",
]


if __name__ == "__main__":
  print("=== Espresso Framework ===")
  esp_info = get_espresso_info()
  print(f"Available: {esp_info.available}")
  if esp_info.available:
    print(f"  EspressoNetwork: {hex(esp_info.network_class)}")
    print(f"  EspressoContext: {hex(esp_info.context_class)}")
  
  print("\n=== AppleNeuralEngine Framework ===")
  ane_info = get_ane_runtime_info()
  print(f"Available: {ane_info.available}")
  if ane_info.available:
    print(f"  _ANEClient: {hex(ane_info.client_class)}")
    print(f"  _ANEModel: {hex(ane_info.model_class)}")
    print(f"  _ANEDeviceController: {hex(ane_info.device_controller_class)}")
    print(f"  _ANEDaemonConnection: {hex(ane_info.daemon_connection_class)}")
  
  print("\n=== Test CPU Espresso ===")
  ctx = create_espresso_cpu_context()
  print(f"CPU Context: {hex(ctx) if ctx else 'NULL'}")
  
  if ctx:
    # Try loading a simple model
    model_path = "/System/Library/PrivateFrameworks/CoreSuggestionsInternals.framework/Versions/A/Resources/assets_130/model/model.mlmodelc/model.espresso.net"
    net = load_espresso_network(model_path, ctx)
    print(f"Network: {hex(net) if net else 'NULL'}")
    
    if net:
      layers = get_network_layer_count(net)
      print(f"Layers: {layers}")
