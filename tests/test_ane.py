# Tests for ANE (Apple Neural Engine) compiler bindings
from __future__ import annotations
import pytest
import sys
import struct

# Skip all tests if not on macOS
pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="ANE only available on macOS")


class TestANEStructs:
  """Test ANE struct dataclasses."""
  
  def test_kernel_size_defaults(self):
    from ane import ANECKernelSize
    ks = ANECKernelSize()
    assert ks.depth == 1
    assert ks.height == 1
    assert ks.width == 1
  
  def test_kernel_size_custom(self):
    from ane import ANECKernelSize
    ks = ANECKernelSize(depth=1, height=3, width=3)
    assert ks.height == 3
    assert ks.width == 3
  
  def test_kernel_size_to_bytes(self):
    from ane import ANECKernelSize
    ks = ANECKernelSize(depth=1, height=3, width=5)
    data = ks.to_bytes()
    assert len(data) == 24  # 3 x u64
    d, h, w = struct.unpack("<QQQ", data)
    assert d == 1
    assert h == 3
    assert w == 5
  
  def test_kernel_size_from_bytes(self):
    from ane import ANECKernelSize
    data = struct.pack("<QQQ", 2, 7, 7)
    ks = ANECKernelSize.from_bytes(data)
    assert ks.depth == 2
    assert ks.height == 7
    assert ks.width == 7
  
  def test_step_to_bytes(self):
    from ane import ANECStep
    step = ANECStep(depth=1, height=2, width=2)
    data = step.to_bytes()
    assert len(data) == 12  # 3 x u32
    d, h, w = struct.unpack("<III", data)
    assert d == 1
    assert h == 2
    assert w == 2
  
  def test_padding_to_bytes(self):
    from ane import ANECPadding
    pad = ANECPadding(depth_front=0, depth_back=0, height_front=1, height_back=1, width_front=2, width_back=2)
    data = pad.to_bytes()
    assert len(data) == 24  # 6 x u32
  
  def test_tensor_dims_defaults(self):
    from ane import ANECTensorDims
    dims = ANECTensorDims()
    assert dims.n == 1
    assert dims.c == 1
    assert dims.h == 1
    assert dims.w == 1
    assert dims.d == 1
  
  def test_tensor_dims_to_bytes(self):
    from ane import ANECTensorDims
    dims = ANECTensorDims(n=1, c=64, h=224, w=224, d=1)
    data = dims.to_bytes()
    assert len(data) == 40  # 5 x u64
    n, c, h, w, d = struct.unpack("<QQQQQ", data)
    assert n == 1
    assert c == 64
    assert h == 224
    assert w == 224


class TestANECompiler:
  """Test ANECompiler wrapper class."""
  
  @pytest.fixture
  def ane(self):
    from ane import ANECompiler
    return ANECompiler()
  
  def test_load_framework(self, ane):
    """Test that ANECompiler framework loads successfully."""
    assert ane is not None
    assert ane._ane is not None
  
  def test_mps_dialect_version(self, ane):
    """Test version getter."""
    ver = ane.mps_dialect_version
    assert isinstance(ver, int)
    assert ver >= 0
  
  def test_analytics_buffer_size(self, ane):
    """Test analytics buffer size getter."""
    size = ane.analytics_buffer_size
    assert isinstance(size, int)
    assert size > 0
  
  def test_init_kernel_size(self, ane):
    """Test kernel size initialization via ANE framework."""
    ks = ane.init_kernel_size(d=1, h=3, w=3)
    assert ks.depth == 1
    assert ks.height == 3
    assert ks.width == 3
  
  def test_init_step(self, ane):
    """Test step/stride initialization."""
    step = ane.init_step(d=1, h=2, w=2)
    assert step.depth == 1
    assert step.height == 2
    assert step.width == 2
  
  def test_init_padding(self, ane):
    """Test padding initialization."""
    pad = ane.init_padding(hf=1, hb=1, wf=1, wb=1)
    assert pad.height_front == 1
    assert pad.height_back == 1
    assert pad.width_front == 1
    assert pad.width_back == 1
  
  def test_init_tensor_dims(self, ane):
    """Test tensor dims initialization."""
    dims = ane.init_tensor_dims(n=1, c=64, h=224, w=224)
    assert dims.n == 1
    assert dims.c == 64
    assert dims.h == 224
    assert dims.w == 224
  
  def test_init_conv_layer_desc(self, ane):
    """Test conv layer descriptor initialization."""
    conv = ane.init_conv_layer_desc()
    assert conv.size == 176
    assert len(conv._raw) == 176
  
  def test_init_pool_layer_desc(self, ane):
    """Test pool layer descriptor initialization."""
    pool = ane.init_pool_layer_desc()
    assert pool.size == 96
  
  def test_init_linear_layer_desc(self, ane):
    """Test linear layer descriptor initialization."""
    linear = ane.init_linear_layer_desc()
    assert linear.size == 64
  
  def test_init_matmul_layer_desc(self, ane):
    """Test matmul layer descriptor initialization."""
    matmul = ane.init_matmul_layer_desc()
    assert matmul.size == 16
  
  def test_init_softmax_layer_desc(self, ane):
    """Test softmax layer descriptor initialization."""
    softmax = ane.init_softmax_layer_desc()
    assert softmax.size == 48
  
  def test_init_sdpa_layer_desc(self, ane):
    """Test SDPA (attention) layer descriptor initialization."""
    sdpa = ane.init_sdpa_layer_desc()
    assert sdpa.size == 8


class TestANEHelpers:
  """Test helper functions."""
  
  def test_get_ane_info(self):
    """Test get_ane_info returns expected structure."""
    from ane import get_ane_info
    info = get_ane_info()
    
    assert "available" in info
    assert info["available"] == True
    assert "mps_dialect_version" in info
    assert "struct_sizes" in info
  
  def test_struct_sizes_constant(self):
    """Test STRUCT_SIZES contains expected keys."""
    from ane import STRUCT_SIZES
    
    expected = [
      "ANECKernelSize",
      "ANECStep",
      "ANECPadding",
      "ANECTensorDims",
      "ANECTensorDesc",
      "ANECConvLayerDesc",
      "ANECPoolLayerDesc",
    ]
    
    for name in expected:
      assert name in STRUCT_SIZES
      assert STRUCT_SIZES[name] > 0
  
  def test_probe_all_layer_desc_sizes(self):
    """Test probing all layer descriptor sizes."""
    from ane import probe_all_layer_desc_sizes
    sizes = probe_all_layer_desc_sizes()
    
    # Should find at least 40 layer types
    assert len(sizes) >= 40
    
    # Check some known layers
    assert "ANECConvLayerDesc" in sizes
    assert "ANECPoolLayerDesc" in sizes
    assert "ANECSDPALayerDesc" in sizes
    
    # Sizes should be reasonable
    for name, size in sizes.items():
      assert size > 0, f"{name} has zero size"
      assert size < 1024, f"{name} has unreasonably large size: {size}"


class TestANESample:
  """Test the sample graph building code."""
  
  def test_simple_cnn(self):
    """Test building a simple CNN graph."""
    from ane import build_simple_cnn
    
    graph = build_simple_cnn()
    assert len(graph.layers) == 8
    
    # Check layer types
    layer_types = [l.op_type for l in graph.layers]
    assert layer_types.count("conv2d") == 3
    assert layer_types.count("pool2d") == 2
    assert layer_types.count("linear") == 2
    assert layer_types.count("softmax") == 1
  
  def test_transformer_attention(self):
    """Test building a transformer attention block."""
    from ane import build_transformer_attention
    
    graph = build_transformer_attention()
    assert len(graph.layers) == 3
    
    # Check for SDPA layer
    layer_types = [l.op_type for l in graph.layers]
    assert "sdpa" in layer_types
    assert layer_types.count("linear") == 2
  
  def test_graph_summary(self):
    """Test graph summary output."""
    from ane import build_simple_cnn
    
    graph = build_simple_cnn()
    summary = graph.summary()
    
    assert "ANE Computation Graph" in summary
    assert "conv1" in summary
    assert "Total layers: 8" in summary
    assert "Total descriptor bytes:" in summary


class TestANELayerSizes:
  """Test specific layer descriptor sizes match our probed values."""
  
  def test_conv_layer_size(self):
    from ane import ANECompiler
    ane = ANECompiler()
    conv = ane.init_conv_layer_desc()
    assert conv.size == 176
  
  def test_pool_layer_size(self):
    from ane import ANECompiler
    ane = ANECompiler()
    pool = ane.init_pool_layer_desc()
    assert pool.size == 96
  
  def test_sdpa_layer_size(self):
    from ane import ANECompiler
    ane = ANECompiler()
    sdpa = ane.init_sdpa_layer_desc()
    # SDPA is remarkably small - 8 bytes
    assert sdpa.size == 8


class TestEspressoDiscovery:
  """Test Espresso class discovery functions."""
  
  def test_list_espresso_classes(self):
    from ane import list_espresso_classes
    classes = list_espresso_classes()
    
    # Should find many Espresso classes
    assert len(classes) >= 50
    
    # Check for known classes
    assert "EspressoContext" in classes
    assert "EspressoNetwork" in classes
    assert "EspressoANEIOSurface" in classes
    
    # Should find optimization passes
    passes = [c for c in classes if "Pass_" in c]
    assert len(passes) >= 40
  
  def test_list_class_methods(self):
    from ane import list_class_methods
    methods = list_class_methods("EspressoContext")
    
    # Should have multiple methods
    assert len(methods) >= 5
    
    # Check for known methods
    method_names = [m[0] for m in methods]
    assert "initWithPlatform:" in method_names
    assert "platform" in method_names
  
  def test_list_class_methods_with_encoding(self):
    from ane import list_class_methods
    methods = list_class_methods("EspressoNetwork")
    
    # Find initWithJSFile method
    init_method = None
    for name, encoding in methods:
      if "initWithJSFile:" in name:
        init_method = (name, encoding)
        break
    
    assert init_method is not None
    # Should have type encoding with r* (const char*)
    assert "r*" in init_method[1]


class TestEspressoFormat:
  """Test espresso.net format parsing."""
  
  def test_find_system_models(self):
    from ane import find_system_models
    models = find_system_models()
    
    # Should find many system models
    assert len(models) >= 100
  
  def test_is_json_format(self):
    from ane import find_system_models, is_json_format
    from pathlib import Path
    
    models = find_system_models()
    
    # Count formats
    json_count = sum(1 for m in models if is_json_format(m))
    binary_count = len(models) - json_count
    
    # Should have both formats
    assert json_count > 0
    assert binary_count > 0
  
  def test_parse_json_model(self):
    from ane import EspressoNet, find_system_models, is_json_format
    
    # Find a JSON model
    models = find_system_models()
    json_model = next((m for m in models if is_json_format(m)), None)
    
    if json_model is None:
      pytest.skip("No JSON espresso.net models found")
    
    net = EspressoNet.from_file(json_model)
    
    assert net.format_version in [200, 300]
    assert len(net.layers) > 0
    
    # Check layer type counts
    counts = net.layer_type_counts()
    assert len(counts) > 0
  
  def test_espresso_layer_parsing(self):
    from ane import EspressoLayer
    
    # Test parsing a layer dict
    layer_dict = {
      "name": "conv1",
      "type": "convolution",
      "bottom": "input",
      "top": "conv1_output",
      "kernel_size": 3,
      "stride": 1,
    }
    
    layer = EspressoLayer.from_dict(layer_dict)
    
    assert layer.name == "conv1"
    assert layer.type == "convolution"
    assert layer.bottom == ["input"]
    assert layer.top == "conv1_output"
    assert layer.attributes["kernel_size"] == 3


class TestPBZE:
  """Test PBZE format decoder."""
  
  def test_is_pbze(self):
    from ane import is_pbze
    
    assert is_pbze(b'pbze\x00\x00\x00\x00')
    assert not is_pbze(b'{\n  "layers"')
    assert not is_pbze(b'')
  
  def test_find_pbze_file(self):
    from ane import is_pbze_file, find_system_models
    
    models = find_system_models()
    pbze_files = [m for m in models if is_pbze_file(m)]
    
    # Should find some PBZE files
    assert len(pbze_files) > 0
  
  def test_decode_pbze(self):
    from ane import is_pbze_file, load_pbze_json, find_system_models
    
    models = find_system_models()
    pbze_file = next((m for m in models if is_pbze_file(m)), None)
    
    if pbze_file is None:
      pytest.skip("No PBZE files found")
    
    data = load_pbze_json(pbze_file)
    
    # Should be valid espresso.net JSON
    assert "format_version" in data
    assert "layers" in data
    assert len(data["layers"]) > 0
  
  def test_get_pbze_stats(self):
    from ane import is_pbze_file, get_pbze_stats, find_system_models
    
    models = find_system_models()
    pbze_file = next((m for m in models if is_pbze_file(m)), None)
    
    if pbze_file is None:
      pytest.skip("No PBZE files found")
    
    stats = get_pbze_stats(pbze_file)
    
    assert stats["file_size"] > 0
    assert stats["uncompressed_size"] > stats["compressed_size"]
    assert stats["compression_ratio"] > 1.0
  
  def test_decode_espresso_net_json(self):
    from ane import decode_espresso_net, is_pbze_file, find_system_models
    
    models = find_system_models()
    json_file = next((m for m in models if not is_pbze_file(m)), None)
    
    if json_file is None:
      pytest.skip("No JSON espresso.net files found")
    
    data = decode_espresso_net(json_file)
    assert "layers" in data
  
  def test_decode_espresso_net_pbze(self):
    from ane import decode_espresso_net, is_pbze_file, find_system_models
    
    models = find_system_models()
    pbze_file = next((m for m in models if is_pbze_file(m)), None)
    
    if pbze_file is None:
      pytest.skip("No PBZE files found")
    
    data = decode_espresso_net(pbze_file)
    assert "layers" in data


class TestANEXPC:
  """Test ANE XPC protocol discovery."""
  
  def test_get_ane_xpc_info(self):
    from ane import get_ane_xpc_info
    
    info = get_ane_xpc_info()
    
    assert info.available
    assert info.daemon_connection_class != 0
    assert info.client_class != 0
    assert info.model_class != 0
    assert info.request_class != 0
  
  def test_list_daemon_connection_methods(self):
    from ane import list_daemon_connection_methods
    
    methods = list_daemon_connection_methods()
    
    # Should have multiple methods
    assert len(methods) >= 10
    
    # Check for known methods
    method_names = [m[0] for m in methods]
    assert "compileModel:sandboxExtension:options:qos:withReply:" in method_names
    assert "loadModel:sandboxExtension:options:qos:withReply:" in method_names
    assert "echo:withReply:" in method_names
  
  def test_list_client_methods(self):
    from ane import list_client_methods
    
    methods = list_client_methods()
    
    # Should have many methods
    assert len(methods) >= 30
    
    # Check for known methods
    method_names = [m[0] for m in methods]
    assert "compileModel:options:qos:error:" in method_names
    assert "evaluateWithModel:options:request:qos:error:" in method_names
    assert "mapIOSurfacesWithModel:request:cacheInference:error:" in method_names
  
  def test_create_ane_client(self):
    from ane import create_ane_client
    
    client = create_ane_client()
    
    # Should create successfully (even without entitlements)
    assert client is not None
    assert client != 0
  
  def test_xpc_operations_structure(self):
    from ane import XPC_OPERATIONS, REQUIRED_ENTITLEMENTS
    
    # Check structure
    assert "compile" in XPC_OPERATIONS
    assert "load" in XPC_OPERATIONS
    assert "execute" in XPC_OPERATIONS
    
    # Check entitlements
    assert len(REQUIRED_ENTITLEMENTS) >= 2
    assert "com.apple.aned.internal" in REQUIRED_ENTITLEMENTS


class TestANEDevice:
  """Tests for ANE device info and security research."""
  
  def test_boot_args_constants(self):
    from ane import BOOT_ARGS
    
    # Check known boot args exist
    assert "ane_skipAdapterWeightAccessCheck" in BOOT_ARGS
    assert "ane_vm_allowPrecompiledBinary" in BOOT_ARGS
    
    # Check structure
    bypass_info = BOOT_ARGS["ane_skipAdapterWeightAccessCheck"]
    assert "purpose" in bypass_info
    assert "bypasses_entitlement" in bypass_info
    assert bypass_info["bypasses_entitlement"] == "com.apple.aned.private.adapterWeight.allow"
  
  def test_entitlements_constants(self):
    from ane import ENTITLEMENTS
    
    # Check known entitlements exist
    assert "com.apple.aned.private.allow" in ENTITLEMENTS
    assert "com.apple.aned.private.adapterWeight.allow" in ENTITLEMENTS
    
    # Check structure
    primary = ENTITLEMENTS["com.apple.aned.private.allow"]
    assert "purpose" in primary
    assert "required_for" in primary
    assert "compile" in primary["required_for"]
  
  def test_system_paths_constants(self):
    from ane import SYSTEM_PATHS
    
    # Check known paths exist
    assert "aned_daemon" in SYSTEM_PATHS
    assert SYSTEM_PATHS["aned_daemon"] == "/usr/libexec/aned"
    assert "ane_services" in SYSTEM_PATHS
    assert "internal_library" in SYSTEM_PATHS
    assert SYSTEM_PATHS["internal_library"] == "/AppleInternal/Library"
  
  def test_check_internal_build_indicators(self):
    from ane import check_internal_build_indicators
    
    indicators = check_internal_build_indicators()
    
    # Should have all expected keys
    assert "apple_internal_exists" in indicators
    assert "os_variant" in indicators
    assert "is_internal_build" in indicators
    
    # On consumer macOS, should be False
    assert indicators["is_internal_build"] == False
  
  def test_get_boot_args(self):
    from ane import get_boot_args
    
    # Should return a string (may be empty)
    boot_args = get_boot_args()
    assert isinstance(boot_args, str)
  
  def test_is_boot_arg_present(self):
    from ane import is_boot_arg_present
    
    # ANE bypass should not be present on consumer macOS
    assert is_boot_arg_present("ane_skipAdapterWeightAccessCheck") == False
  
  def test_get_entitlement_requirements(self):
    from ane import get_entitlement_requirements
    
    reqs = get_entitlement_requirements()
    
    # Should have expected operations
    assert "compile" in reqs
    assert "load" in reqs
    assert "evaluate" in reqs
    
    # Should have entitlements listed
    # Note: The actual entitlements depend on the mapping logic


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
