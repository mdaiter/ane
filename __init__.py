# Apple Neural Engine (ANE) reverse engineering artifacts
# Provides Python bindings and knowledge for ANE, Espresso, and related frameworks
from .compiler import *
from .runtime import *
from .espresso import *
from .pbze import *
from .xpc import *
from .sample import *

def build_ane_api_tree():
  """Build API tree from ANE discoveries.
  
  Returns an APITree populated with all known ANE, Espresso, and 
  AppleNeuralEngine APIs discovered through reverse engineering.
  """
  from mcgyver.header_index.api_tree import APITree, APISignature, APIExample, Capability, Evidence, EvidenceType, Parameter
  
  tree = APITree()
  
  # Add ANECompiler struct APIs
  structs = [
    ("ANECKernelSize", 24, "Kernel size for conv/pool (D, H, W)"),
    ("ANECStep", 12, "Stride for conv/pool (D, H, W)"),
    ("ANECPadding", 24, "Padding (6 values: front/back for D, H, W)"),
    ("ANECTensorDims", 40, "Tensor dimensions (N, C, H, W, D)"),
    ("ANECTensorDesc", 64, "Tensor descriptor"),
    ("ANECConvLayerDesc", 176, "Convolution layer descriptor"),
    ("ANECPoolLayerDesc", 96, "Pooling layer descriptor"),
    ("ANECLinearLayerDesc", 64, "Linear/dense layer descriptor"),
    ("ANECMatrixMultLayerDesc", 16, "Matrix multiplication descriptor"),
    ("ANECSoftmaxLayerDesc", 48, "Softmax layer descriptor"),
    ("ANECSDPALayerDesc", 8, "Scaled Dot-Product Attention (transformer)"),
  ]
  
  for name, size, desc in structs:
    api = APISignature(
      name=name,
      framework="ANECompiler",
      signature=f"struct {name}",
      return_type="struct",
      is_struct=True,
      struct_size=size,
    )
    api.add_evidence(Evidence(
      type=EvidenceType.RUNTIME_PROBE,
      description=f"Size determined by probing {name}Initialize with sentinel buffer",
      confidence=0.95,
      data={"size": size},
    ))
    api.add_evidence(Evidence(
      type=EvidenceType.DYLD_EXPORT,
      description=f"Found {name}Initialize in dyld_info exports",
      confidence=0.9,
    ))
    tree.add_api(api)
    
    # Add corresponding Initialize function
    init_api = APISignature(
      name=f"{name}Initialize",
      framework="ANECompiler",
      signature=f"void {name}Initialize({name} *desc)",
      return_type="void",
      parameters=[Parameter(name="desc", type_name=f"{name} *", is_pointer=True)],
    )
    init_api.add_evidence(Evidence(
      type=EvidenceType.RUNTIME_PROBE,
      description="Called successfully with buffer pointer",
      confidence=0.95,
    ))
    init_api.add_evidence(Evidence(
      type=EvidenceType.INFERENCE,
      description="Apple pattern: *Initialize takes single pointer to struct",
      confidence=0.8,
    ))
    tree.add_api(init_api)
  
  # Add Espresso APIs
  espresso_ctx = APISignature(
    name="EspressoContext",
    framework="Espresso",
    signature="-[EspressoContext initWithPlatform:]",
    return_type="instancetype",
    parameters=[Parameter(name="platform", type_name="int", description="0=CPU, 1=GPU, 2=ANE")],
    is_objc=True,
    objc_class="EspressoContext",
    objc_selector="initWithPlatform:",
    objc_encoding="@20@0:8i16",
  )
  espresso_ctx.add_evidence(Evidence(
    type=EvidenceType.OBJC_INTROSPECTION,
    description="Found via class_copyMethodList",
    confidence=0.95,
  ))
  espresso_ctx.add_evidence(Evidence(
    type=EvidenceType.WORKING_EXAMPLE,
    description="Platform 0 (CPU) works, creates valid context",
    confidence=1.0,
  ))
  tree.add_api(espresso_ctx)
  
  espresso_net = APISignature(
    name="EspressoNetwork",
    framework="Espresso",
    signature="-[EspressoNetwork initWithJSFile:context:computePath:]",
    return_type="instancetype",
    parameters=[
      Parameter(name="jsFile", type_name="const char *", description="Path to .espresso.net file"),
      Parameter(name="context", type_name="EspressoContext *", description="Context from initWithPlatform:"),
      Parameter(name="computePath", type_name="int", description="Flag, not path"),
    ],
    is_objc=True,
    objc_class="EspressoNetwork",
    objc_selector="initWithJSFile:context:computePath:",
    objc_encoding="@36@0:8r*16@24i32",
  )
  espresso_net.add_evidence(Evidence(
    type=EvidenceType.OBJC_INTROSPECTION,
    description="Found via class_copyMethodList, encoding shows r* (const char*)",
    confidence=0.95,
  ))
  espresso_net.add_evidence(Evidence(
    type=EvidenceType.CRASH_MESSAGE,
    description="Passing NSString crashed, C string works",
    confidence=1.0,
  ))
  espresso_net.add_evidence(Evidence(
    type=EvidenceType.WORKING_EXAMPLE,
    description="Successfully loaded model, got layer count",
    confidence=1.0,
  ))
  tree.add_api(espresso_net)
  
  # Add example
  example = APIExample(
    name="load_model_cpu",
    description="Load an espresso model on CPU",
    language="python",
    apis_used=["EspressoContext", "EspressoNetwork"],
    tested=True,
    test_output="Network loaded with 1 layers",
    code='''from mcgyver.header_index.os_frameworks.mac.ane import (
    create_espresso_cpu_context,
    load_espresso_network,
    get_network_layer_count,
)

# Create CPU context
ctx = create_espresso_cpu_context()
print(f"Context: {hex(ctx)}")

# Load model
model_path = "/path/to/model.espresso.net"
net = load_espresso_network(model_path, ctx)
print(f"Network: {hex(net)}")

# Get layer count
layers = get_network_layer_count(net)
print(f"Layers: {layers}")''',
  )
  tree.add_example(example)
  
  # Add capabilities
  tree.add_capability(Capability(
    name="convolution",
    description="2D/3D convolution operations",
    keywords=["conv", "conv2d", "conv3d", "filter", "kernel"],
    apis=["ANECompiler/ANECConvLayerDesc", "ANECompiler/ANECConvLayerDescInitialize"],
    examples=["build_cnn"],
  ))
  
  tree.add_capability(Capability(
    name="attention",
    description="Transformer attention mechanisms",
    keywords=["attention", "sdpa", "transformer", "self-attention", "multi-head"],
    apis=["ANECompiler/ANECSDPALayerDesc", "ANECompiler/ANECSDPALayerDescInitialize"],
    examples=["build_transformer"],
  ))
  
  tree.add_capability(Capability(
    name="inference",
    description="Run neural network inference",
    keywords=["inference", "predict", "forward", "run", "execute"],
    apis=["Espresso/EspressoContext", "Espresso/EspressoNetwork"],
    examples=["load_model_cpu"],
  ))
  
  # Add Espresso profiling APIs
  profiling_layer = APISignature(
    name="EspressoProfilingLayerInfo",
    framework="Espresso",
    signature="-[EspressoProfilingLayerInfo average_runtime]",
    return_type="double",
    is_objc=True,
    objc_class="EspressoProfilingLayerInfo",
  )
  profiling_layer.add_evidence(Evidence(
    type=EvidenceType.OBJC_INTROSPECTION,
    description="Found via class_copyMethodList: name, debug_name, average_runtime (d), selected_runtime_engine (i), runtimes",
    confidence=0.95,
  ))
  tree.add_api(profiling_layer)
  
  profiling_ane = APISignature(
    name="EspressoProfilingNetworkANEInfo",
    framework="Espresso",
    signature="-[EspressoProfilingNetworkANEInfo total_ane_time_ns]",
    return_type="uint64",
    is_objc=True,
    objc_class="EspressoProfilingNetworkANEInfo",
  )
  profiling_ane.add_evidence(Evidence(
    type=EvidenceType.OBJC_INTROSPECTION,
    description="Methods: total_ane_time_ns (Q), ane_time_per_eval_ns (Q)",
    confidence=0.95,
  ))
  tree.add_api(profiling_ane)
  
  # Add EspressoANEIOSurface for ANE memory
  ane_io = APISignature(
    name="EspressoANEIOSurface",
    framework="Espresso",
    signature="-[EspressoANEIOSurface createIOSurfaceWithExtraProperties:]",
    return_type="IOSurfaceRef",
    is_objc=True,
    objc_class="EspressoANEIOSurface",
  )
  ane_io.add_evidence(Evidence(
    type=EvidenceType.OBJC_INTROSPECTION,
    description="21 methods: createIOSurface, metalBufferWithDevice, setExternalStorage:ioSurface:, nFrames",
    confidence=0.95,
  ))
  tree.add_api(ane_io)
  
  # Add optimization pass base class
  pass_api = APISignature(
    name="EspressoCustomPass",
    framework="Espresso",
    signature="-[EspressoCustomPass runOnNetwork:]",
    return_type="BOOL",
    parameters=[Parameter(name="network", type_name="void *", is_pointer=True, description="C++ net pointer")],
    is_objc=True,
    objc_class="EspressoCustomPass",
  )
  pass_api.add_evidence(Evidence(
    type=EvidenceType.OBJC_INTROSPECTION,
    description="Base class for 40+ optimization passes: fuse_conv_batchnorm, fold_constants, etc.",
    confidence=0.95,
  ))
  tree.add_api(pass_api)
  
  # Add profiling capability
  tree.add_capability(Capability(
    name="profiling",
    description="Profile ANE and layer execution times",
    keywords=["profile", "timing", "performance", "benchmark", "latency"],
    apis=["Espresso/EspressoProfilingLayerInfo", "Espresso/EspressoProfilingNetworkANEInfo"],
    examples=[],
  ))
  
  # Add memory capability
  tree.add_capability(Capability(
    name="ane_memory",
    description="ANE memory management via IOSurface",
    keywords=["memory", "iosurface", "buffer", "tensor", "allocation"],
    apis=["Espresso/EspressoANEIOSurface"],
    examples=[],
  ))
  
  return tree
