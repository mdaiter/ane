# ANE XPC Protocol Documentation and Client
"""
ANE XPC Protocol
================

Apple Neural Engine uses XPC (Cross-Process Communication) to communicate
between client applications and the aned daemon.

Services
--------
- com.apple.appleneuralengine: Main service (requires entitlements)
- com.apple.appleneuralengine.private: Private service  
- com.apple.aned: Daemon Mach service

Daemon: /usr/libexec/aned
Launch plist: /System/Library/LaunchDaemons/com.apple.aned.plist

Client Classes (AppleNeuralEngine.framework)
--------------------------------------------
_ANEDaemonConnection: XPC connection to aned
_ANEClient: High-level client API
_ANEModel: Model representation
_ANERequest: Inference request

XPC Operations (_ANEDaemonConnection methods)
---------------------------------------------
1. Model Compilation:
   - compileModel:sandboxExtension:options:qos:withReply:
   - compiledModelExistsFor:withReply:
   - compiledModelExistsMatchingHash:withReply:
   - purgeCompiledModel:withReply:

2. Model Loading:
   - loadModel:sandboxExtension:options:qos:withReply:
   - loadModelNewInstance:options:modelInstParams:qos:withReply:
   - unloadModel:options:qos:withReply:

3. Execution:
   - prepareChainingWithModel:options:chainingReq:qos:withReply:
   
4. Real-time:
   - beginRealTimeTaskWithReply:
   - endRealTimeTaskWithReply:

5. Utility:
   - echo:withReply:
   - reportTelemetryToPPS:playload:

_ANEClient High-Level API
-------------------------
Compile:
  - compileModel:options:qos:error:
  - compiledModelExistsFor:
  - compiledModelExistsMatchingHash:
  - purgeCompiledModel:
  - purgeCompiledModelMatchingHash:

Load:
  - loadModel:options:qos:error:
  - loadModelNewInstance:options:modelInstParams:qos:error:
  - loadRealTimeModel:options:qos:error:
  - unloadModel:options:qos:error:
  - unloadRealTimeModel:options:qos:error:

Evaluate:
  - evaluateWithModel:options:request:qos:error:
  - evaluateRealTimeWithModel:options:request:error:
  - evaluateDirectWithModel:options:request:qos:error: (internal)

Memory:
  - mapIOSurfacesWithModel:request:cacheInference:error:
  - unmapIOSurfacesWithModel:request:
  - buffersReadyWithModel:inputBuffers:options:qos:error:

Chaining:
  - prepareChainingWithModel:options:chainingReq:qos:error:
  - enqueueSetsWithModel:outputSet:options:qos:error:

_ANEModel
---------
Properties:
  - modelURL: URL to .mlmodelc or compiled model
  - sourceURL: Original model URL
  - UUID: Model unique identifier
  - key: Model key string
  - state: Model state (1=created/unloaded, etc.)
  - programHandle: Handle to loaded ANE program
  - intermediateBufferHandle: Handle to intermediate buffers
  - queueDepth: Command queue depth
  - perfStatsMask: Performance statistics mask
  - mpsConstants: MPS (Metal) constants for hybrid execution

Initialization:
  - initWithModelAtURL:key:identifierSource:cacheURLIdentifier:modelAttributes:standardizeURL:
  - initWithModelIdentifier:

_ANERequest
-----------
Properties:
  - inputArray: Array of input IOSurfaces
  - inputIndexArray: Input symbol indices
  - outputArray: Array of output IOSurfaces
  - outputIndexArray: Output symbol indices
  - weightsBuffer: Weights buffer (for dynamic weights)
  - procedureIndex: Which procedure to execute
  - perfStats: Performance statistics output
  - completionHandler: Async completion block
  - sharedEvents: Metal shared events for sync
  - transactionHandle: Transaction identifier

Required Entitlements
---------------------
Full ANE access requires these entitlements:
  - com.apple.aned.internal
  - com.apple.private.ane-client
  - com.apple.developer.kernel.extended-virtual-addressing (for large models)

Without entitlements:
  - _ANEClient can be created
  - compileModel returns NULL
  - loadModel returns NULL
  - No error returned, just fails silently

XPC Message Flow (inferred from strings)
----------------------------------------
1. Client creates _ANEDaemonConnection to "com.apple.appleneuralengine"
2. Client sends compileModel: with model URL and sandbox extension
3. Daemon (aned) calls ANECompiler to compile model
4. Daemon returns compiled model handle or caches it
5. Client sends loadModel: to load into ANE memory
6. Daemon returns programHandle and intermediateBufferHandle
7. Client creates _ANERequest with IOSurfaces
8. Client calls evaluateWithModel: (or real-time variant)
9. Daemon enqueues request to ANE hardware
10. Results written to output IOSurfaces

Key Internal Functions (from aned strings)
------------------------------------------
- ANEProgramCreate(): Creates ANE program from compiled model
- ANEProgramInstanceCreate(): Creates program instance for execution
- destroyProgramInstance: Cleanup
- prepareChainingRequest: Setup chained execution
- garbageCollectDanglingModels: Memory cleanup

Cache System
------------
Models are cached in:
  - MODELCACHEDIR environment variable
  - /var/folders/.../com.apple.aned/ (derived)
  
Cache operations:
  - com.apple.aned.modelCacheAsyncIO
  - com.apple.aned.modelCacheGC
  - com.apple.aned.danglingModelsGC

Performance Statistics
----------------------
perfStatsMask controls what stats are collected:
  - Execution time
  - Queue depth utilization
  - Memory usage
  - Power consumption (on capable devices)

Output available via:
  - _ANERequest.perfStats
  - _ANERequest.perfStatsArray
"""
from __future__ import annotations
import ctypes
from dataclasses import dataclass
from typing import Optional, Callable, Any

# ObjC runtime bindings
_objc: Optional[ctypes.CDLL] = None
_ane: Optional[ctypes.CDLL] = None


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


def _objc_class(name: bytes) -> int:
  objc = _load_objc()
  objc_getClass = objc.objc_getClass
  objc_getClass.argtypes = [ctypes.c_char_p]
  objc_getClass.restype = ctypes.c_void_p
  return objc_getClass(name)


def _sel(name: bytes) -> int:
  objc = _load_objc()
  sel_registerName = objc.sel_registerName
  sel_registerName.argtypes = [ctypes.c_char_p]
  sel_registerName.restype = ctypes.c_void_p
  return sel_registerName(name)


def _make_msg_send(ret_type, arg_types):
  objc = _load_objc()
  fn = ctypes.CFUNCTYPE(ret_type, *arg_types)(("objc_msgSend", objc))
  return fn


# Message send variants
_msg_id_id = None
_msg_id_id_bool = None


def _init_msg_sends():
  global _msg_id_id, _msg_id_id_bool
  if _msg_id_id is not None:
    return
  _msg_id_id = _make_msg_send(ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p])
  _msg_id_id_bool = _make_msg_send(ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool])


@dataclass
class ANEXPCInfo:
  """Information about ANE XPC availability."""
  daemon_connection_class: int
  client_class: int
  model_class: int
  request_class: int
  available: bool


def get_ane_xpc_info() -> ANEXPCInfo:
  """Check if ANE XPC classes are available."""
  try:
    _load_ane()
    _init_msg_sends()
    
    daemon = _objc_class(b"_ANEDaemonConnection")
    client = _objc_class(b"_ANEClient")
    model = _objc_class(b"_ANEModel")
    request = _objc_class(b"_ANERequest")
    
    return ANEXPCInfo(
      daemon_connection_class=daemon,
      client_class=client,
      model_class=model,
      request_class=request,
      available=bool(daemon and client and model and request),
    )
  except Exception:
    return ANEXPCInfo(0, 0, 0, 0, False)


def create_ane_client(restricted_access: bool = True) -> Optional[int]:
  """Create an ANE client.
  
  Args:
    restricted_access: If True, allows restricted access (required for most operations)
    
  Returns:
    Object pointer or None if failed
    
  Note: Even with a valid client, operations will fail without proper entitlements.
  """
  try:
    _load_ane()
    _init_msg_sends()
    
    client_class = _objc_class(b"_ANEClient")
    if not client_class:
      return None
    
    # alloc
    client = _msg_id_id(client_class, _sel(b"alloc"))
    if not client:
      return None
    
    # initWithRestrictedAccessAllowed:
    client = _msg_id_id_bool(client, _sel(b"initWithRestrictedAccessAllowed:"), restricted_access)
    
    return client
  except Exception:
    return None


def list_daemon_connection_methods() -> list[tuple[str, str]]:
  """List all _ANEDaemonConnection methods with type encodings."""
  _load_ane()
  objc = _load_objc()
  
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
  
  cls = _objc_class(b"_ANEDaemonConnection")
  if not cls:
    return []
  
  count = ctypes.c_uint(0)
  methods = class_copyMethodList(cls, ctypes.byref(count))
  
  results = []
  for i in range(count.value):
    sel = method_getName(methods[i])
    name = sel_getName(sel).decode()
    encoding = method_getTypeEncoding(methods[i])
    enc_str = encoding.decode() if encoding else ""
    results.append((name, enc_str))
  
  return results


def list_client_methods() -> list[tuple[str, str]]:
  """List all _ANEClient methods with type encodings."""
  _load_ane()
  objc = _load_objc()
  
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
  
  cls = _objc_class(b"_ANEClient")
  if not cls:
    return []
  
  count = ctypes.c_uint(0)
  methods = class_copyMethodList(cls, ctypes.byref(count))
  
  results = []
  for i in range(count.value):
    sel = method_getName(methods[i])
    name = sel_getName(sel).decode()
    encoding = method_getTypeEncoding(methods[i])
    enc_str = encoding.decode() if encoding else ""
    results.append((name, enc_str))
  
  return results


# XPC Operation Categories
XPC_OPERATIONS = {
  "compile": [
    "compileModel:sandboxExtension:options:qos:withReply:",
    "compiledModelExistsFor:withReply:",
    "compiledModelExistsMatchingHash:withReply:",
    "purgeCompiledModel:withReply:",
    "purgeCompiledModelMatchingHash:withReply:",
  ],
  "load": [
    "loadModel:sandboxExtension:options:qos:withReply:",
    "loadModelNewInstance:options:modelInstParams:qos:withReply:",
    "unloadModel:options:qos:withReply:",
  ],
  "execute": [
    "prepareChainingWithModel:options:chainingReq:qos:withReply:",
  ],
  "realtime": [
    "beginRealTimeTaskWithReply:",
    "endRealTimeTaskWithReply:",
  ],
  "utility": [
    "echo:withReply:",
    "reportTelemetryToPPS:playload:",
  ],
}

# Known entitlements
REQUIRED_ENTITLEMENTS = [
  "com.apple.aned.internal",
  "com.apple.private.ane-client",
  "com.apple.developer.kernel.extended-virtual-addressing",  # For large models
]


if __name__ == "__main__":
  print("=== ANE XPC Protocol Info ===\n")
  
  info = get_ane_xpc_info()
  print(f"XPC Available: {info.available}")
  print(f"  _ANEDaemonConnection: {hex(info.daemon_connection_class) if info.daemon_connection_class else 'N/A'}")
  print(f"  _ANEClient: {hex(info.client_class) if info.client_class else 'N/A'}")
  print(f"  _ANEModel: {hex(info.model_class) if info.model_class else 'N/A'}")
  print(f"  _ANERequest: {hex(info.request_class) if info.request_class else 'N/A'}")
  
  print("\n=== _ANEDaemonConnection Methods ===")
  for name, encoding in list_daemon_connection_methods():
    print(f"  {name}")
  
  print("\n=== XPC Operation Categories ===")
  for category, ops in XPC_OPERATIONS.items():
    print(f"\n{category}:")
    for op in ops:
      print(f"  - {op}")
  
  print("\n=== Required Entitlements ===")
  for ent in REQUIRED_ENTITLEMENTS:
    print(f"  - {ent}")
  
  print("\n=== Testing Client Creation ===")
  client = create_ane_client()
  print(f"Client: {hex(client) if client else 'NULL'}")
  if client:
    print("  (Note: Operations will fail without proper entitlements)")
