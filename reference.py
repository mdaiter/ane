"""
ANE Reference - Comprehensive documentation of ANE interfaces and research findings.

This module consolidates all reverse engineering findings including:
- IOKit selectors and struct layouts
- Entitlement requirements  
- Security mechanisms and bypass analysis
- API layer documentation
- File format specifications

Based on reverse engineering of:
- ANECompiler.framework, ANEServices.framework, AppleNeuralEngine.framework
- AppleH11ANEInterface.kext (kernel driver)
- aned daemon (/usr/libexec/aned)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import IntEnum
import ctypes


# =============================================================================
# IOKit Error Codes
# =============================================================================

class IOKitError(IntEnum):
    """IOKit return codes."""
    SUCCESS = 0x0
    ERROR = 0xe00002bc
    NO_MEMORY = 0xe00002bd
    NO_RESOURCES = 0xe00002be
    BAD_ARGUMENT = 0xe00002c2
    EXCLUSIVE_ACCESS = 0xe00002c1
    UNSUPPORTED = 0xe00002c7
    NOT_PRIVILEGED = 0xe00002cd
    NOT_READY = 0xe00002e9
    NOT_FOUND = 0xe00002f0
    
    @classmethod
    def to_string(cls, code: int) -> str:
        try:
            return cls(code).name
        except ValueError:
            return f"UNKNOWN_0x{code:08x}"


# =============================================================================
# ANEDeviceInfo Struct (0x68 = 104 bytes)
# =============================================================================

class ANEDeviceInfo(ctypes.Structure):
    """
    ANEDeviceInfo structure for DeviceOpen (selector 0).
    Size: 0x68 (104) bytes.
    """
    _layout_ = "ms"
    _pack_ = 1
    _fields_ = [
        ("callback", ctypes.c_uint64),       # 0x00
        ("context", ctypes.c_uint64),        # 0x08
        ("controller", ctypes.c_uint64),     # 0x10
        ("init_marker", ctypes.c_int32),     # 0x18 - Set to -1 on default init
        ("device_type", ctypes.c_uint8),     # 0x1c
        ("_pad_1d", ctypes.c_uint8 * 3),     # 0x1d-0x1f
        ("status_field1", ctypes.c_uint32),  # 0x20
        ("status_field2", ctypes.c_uint32),  # 0x24
        ("field_0x28", ctypes.c_uint64),     # 0x28
        ("field_0x30", ctypes.c_uint64),     # 0x30
        ("field_0x38", ctypes.c_uint64),     # 0x38
        ("field_0x40", ctypes.c_uint64),     # 0x40
        ("field_0x48", ctypes.c_uint32),     # 0x48
        ("version", ctypes.c_uint32),        # 0x4c - ANE version
        ("minor_version", ctypes.c_uint32),  # 0x50
        ("field_0x54", ctypes.c_uint32),     # 0x54
        ("field_0x58", ctypes.c_uint64),     # 0x58
        ("field_0x60", ctypes.c_uint64),     # 0x60
    ]
    
    def __init__(self):
        super().__init__()
        self.init_marker = -1

assert ctypes.sizeof(ANEDeviceInfo) == 0x68


# =============================================================================
# IOKit Selectors
# =============================================================================

@dataclass
class SelectorInfo:
    """IOKit selector information."""
    number: int
    name: str
    method_type: str  # "scalar" or "struct"
    input_size: Optional[int]
    output_size: Optional[int]
    description: str

# H11ANEIn (ANEHWDevice) Selectors
H11ANEIN_SELECTORS: Dict[int, SelectorInfo] = {
    0x00: SelectorInfo(0x00, "DeviceOpen", "struct", 0x68, 0x68, "Opens device, returns info"),
    0x01: SelectorInfo(0x01, "DeviceClose", "scalar", 0, 0, "Closes device connection"),
    0x02: SelectorInfo(0x02, "GetStatus", "struct", 0, 32, "Gets device status"),
    0x05: SelectorInfo(0x05, "IsPowered", "scalar", 0, 1, "Checks if ANE is powered"),
    0x06: SelectorInfo(0x06, "PowerOn", "scalar", 0, 0, "Powers on ANE"),
    0x07: SelectorInfo(0x07, "PowerOff", "scalar", 0, 0, "Powers off ANE"),
    0x19: SelectorInfo(0x19, "GetVersion", "scalar", 0, 1, "Gets ANE version"),
    0x1a: SelectorInfo(0x1a, "ReadANERegister", "scalar", 1, 1, "Reads register"),
    0x1b: SelectorInfo(0x1b, "WriteANERegister", "scalar", 2, 0, "Writes register"),
}

# H1xANELoadBalancer Selectors
H1XANELOADBALANCER_SELECTORS: Dict[int, SelectorInfo] = {
    0x00: SelectorInfo(0x00, "DeviceOpen", "struct", 0x68, 0x68, "Opens device"),
    0x01: SelectorInfo(0x01, "DeviceClose", "scalar", 0, 0, "Closes device"),
    0x03: SelectorInfo(0x03, "ProgramCreate", "struct", 0x20, 0, "Creates ANE program"),
    0x06: SelectorInfo(0x06, "ProgramDestroy", "struct", None, None, "Destroys program"),
    0x0a: SelectorInfo(0x0a, "GetVersion", "scalar", 0, 1, "Gets version"),
}


# =============================================================================
# Entitlements
# =============================================================================

@dataclass
class EntitlementInfo:
    """Entitlement information."""
    name: str
    description: str
    is_restricted: bool  # Requires Apple signature
    required_for: List[str]

ANE_ENTITLEMENTS = {
    "com.apple.ane.iokit-user-access": EntitlementInfo(
        "com.apple.ane.iokit-user-access",
        "IOKit direct access to ANE hardware",
        is_restricted=True,
        required_for=["DeviceOpen", "All IOKit operations"],
    ),
    "com.apple.aned.private.allow": EntitlementInfo(
        "com.apple.aned.private.allow",
        "XPC connection to aned for program execution",
        is_restricted=True,
        required_for=["Model compilation", "Model loading", "Evaluation"],
    ),
    "com.apple.ANECompilerService.allow": EntitlementInfo(
        "com.apple.ANECompilerService.allow",
        "Access to ANE compiler service",
        is_restricted=True,
        required_for=["Model compilation"],
    ),
}


# =============================================================================
# API Layers
# =============================================================================

class APIAccessLevel(IntEnum):
    """API access levels."""
    COREML = 0          # Public API
    ESPRESSO = 1        # Private framework
    ANE_SERVICES = 2    # Private framework
    XPC_ANED = 3        # XPC to daemon
    IOKIT_DIRECT = 4    # Direct hardware

@dataclass
class APILayerInfo:
    """API layer documentation."""
    name: str
    level: APIAccessLevel
    status: str  # "working", "limited", "blocked"
    entitlement: Optional[str]
    key_classes: List[str]
    notes: str

API_LAYERS = {
    "CoreML": APILayerInfo(
        "CoreML", APIAccessLevel.COREML, "working", None,
        ["MLModel", "MLModelConfiguration", "MLComputePlan", "MLNeuralEngineComputeDevice"],
        "Use MLComputeUnitsAll to enable ANE. System handles device selection."
    ),
    "Espresso": APILayerInfo(
        "Espresso", APIAccessLevel.ESPRESSO, "limited", None,
        ["EspressoContext", "EspressoNetwork", "EspressoPlan"],
        "Platform 2 (ANE) context crashes. CPU (platform 0) works."
    ),
    "ANEServices": APILayerInfo(
        "ANEServices", APIAccessLevel.ANE_SERVICES, "limited", None,
        ["_ANEClient", "_ANEModel", "_ANEDaemonConnection"],
        "XPC connection works. Model loading needs .espresso.net file."
    ),
    "IOKit": APILayerInfo(
        "IOKit", APIAccessLevel.IOKIT_DIRECT, "blocked",
        "com.apple.ane.iokit-user-access",
        ["H11ANEIn", "H1xANELoadBalancer"],
        "DeviceOpen returns kIOReturnNotFound without entitlement."
    ),
}


# =============================================================================
# File Formats
# =============================================================================

@dataclass
class FileFormatInfo:
    """File format documentation."""
    extension: str
    description: str
    format_type: str  # "json", "binary", "mach-o"
    notes: str

ESPRESSO_FILE_FORMATS = {
    ".espresso.net": FileFormatInfo(
        ".espresso.net",
        "Network description",
        "json",
        "JSON with layers, configurations, analyses. May be PBZE compressed."
    ),
    ".espresso.weights": FileFormatInfo(
        ".espresso.weights",
        "Model weights",
        "binary",
        "Binary blob referenced by blob_weights indices in .net file."
    ),
    ".espresso.shape": FileFormatInfo(
        ".espresso.shape",
        "Shape information",
        "json",
        "Input/output shapes for network configurations."
    ),
    ".espresso.hwx": FileFormatInfo(
        ".espresso.hwx",
        "Pre-compiled ANE binary",
        "mach-o",
        "Magic 0xBEEFFACE. Chip-specific (H13/H14/H15/H16). Needs .net file."
    ),
    ".precompilation_info": FileFormatInfo(
        ".precompilation_info",
        "Compiler metadata",
        "json",
        "ANE_COMPILER validation status, visit order, unsupported ops."
    ),
}


# =============================================================================
# Compute Unit Masks
# =============================================================================

COMPUTE_UNIT_MASKS = {
    1: "CPU only",
    2: "GPU only",
    3: "CPU + GPU",
    4: "Neural Engine only",
    5: "CPU + Neural Engine",
    6: "GPU + Neural Engine",
    7: "CPU + GPU + Neural Engine (all)",
}


# =============================================================================
# Security Summary
# =============================================================================

SECURITY_SUMMARY = """
## ANE Security Model

### What Works Without Entitlements
- Load frameworks (ANECompiler, ANEServices, Espresso)
- Call ANEC*Initialize() functions (struct probing)
- Create _ANEClient object
- XPC connection to aned (but operations fail)
- CoreML with MLComputeUnitsAll (working ANE path!)

### What Requires Entitlements
- IOKit DeviceOpen (selector 0) - returns kIOReturnNotFound
- compileModel: via XPC - returns nil
- loadModel: via XPC - returns nil
- evaluateWithModel: via XPC - returns nil

### Security Mechanisms
1. AMFI - Kills processes with restricted entitlements
2. Hardened Runtime - Blocks DYLD injection
3. SIP - Prevents kernel modifications
4. Code Signing - Restricted entitlements need Apple signature

### Bypass Status
- Self-signing: SIGKILL (AMFI enforcement)
- Boot args: Require internal build
- DYLD injection: Blocked by hardened runtime
- Kernel mods: Require disabled SIP
- CoreML APIs: WORKING (legitimate path)
"""


# =============================================================================
# Key Findings
# =============================================================================

KEY_FINDINGS = """
## Key Research Findings

### 1. Working Path: CoreML API
```objc
MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
config.computeUnits = MLComputeUnitsAll;  // Enables ANE
MLModel *model = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
```

### 2. XPC Connection Works
- _ANEClient.sharedConnection returns valid connection
- No special entitlement required for connection
- Operations (compile/load/evaluate) need entitlement

### 3. HWX Files Need Network Description
- HWX contains compiled ANE instructions (magic 0xBEEFFACE)
- Cannot load HWX alone - needs companion .espresso.net
- Different HWX versions for different chip generations

### 4. IOKit Blocked by Entitlement
- IOServiceOpen succeeds
- DeviceOpen (selector 0) fails with kIOReturnNotFound
- Only aned has com.apple.ane.iokit-user-access

### 5. Hardware Info
- MLNeuralEngineComputeDevice.physicalDevice works
- totalCoreCount: 16 (M3 Pro)
- Device available in MLComputePlan
"""


# =============================================================================
# Helper Functions
# =============================================================================

def print_selectors(service: str = "H11ANEIn"):
    """Print selector table for a service."""
    selectors = H11ANEIN_SELECTORS if service == "H11ANEIn" else H1XANELOADBALANCER_SELECTORS
    print(f"\n{service} Selectors:")
    print(f"{'Sel':>4} {'Name':<25} {'Type':<8} {'In':<6} {'Out':<6}")
    print("-" * 55)
    for sel in sorted(selectors.values(), key=lambda s: s.number):
        in_str = str(sel.input_size) if sel.input_size is not None else "?"
        out_str = str(sel.output_size) if sel.output_size is not None else "?"
        print(f"{sel.number:>#4x} {sel.name:<25} {sel.method_type:<8} {in_str:<6} {out_str:<6}")

def print_api_summary():
    """Print API layer summary."""
    print("\nAPI Layer Summary:")
    print(f"{'Layer':<15} {'Status':<10} {'Entitlement':<35}")
    print("-" * 60)
    for name, api in sorted(API_LAYERS.items(), key=lambda x: x[1].level):
        ent = api.entitlement or "None"
        print(f"{name:<15} {api.status:<10} {ent:<35}")

def print_file_formats():
    """Print file format documentation."""
    print("\nEspresso File Formats:")
    for ext, info in ESPRESSO_FILE_FORMATS.items():
        print(f"\n{ext}")
        print(f"  Type: {info.format_type}")
        print(f"  {info.notes}")

def print_full_reference():
    """Print complete reference."""
    print("=" * 70)
    print(" ANE Complete Reference")
    print("=" * 70)
    print_selectors("H11ANEIn")
    print_selectors("H1xANELoadBalancer")
    print_api_summary()
    print_file_formats()
    print(KEY_FINDINGS)
    print(SECURITY_SUMMARY)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "IOKitError",
    "ANEDeviceInfo",
    "SelectorInfo",
    "H11ANEIN_SELECTORS",
    "H1XANELOADBALANCER_SELECTORS",
    "EntitlementInfo",
    "ANE_ENTITLEMENTS",
    "APIAccessLevel",
    "APILayerInfo", 
    "API_LAYERS",
    "FileFormatInfo",
    "ESPRESSO_FILE_FORMATS",
    "COMPUTE_UNIT_MASKS",
    "SECURITY_SUMMARY",
    "KEY_FINDINGS",
    "print_selectors",
    "print_api_summary",
    "print_file_formats",
    "print_full_reference",
]


if __name__ == "__main__":
    print_full_reference()
