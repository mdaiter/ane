"""
ANE Device Information and Security Bypass Research

This module provides access to ANE device information and documents
the security mechanisms used by the ANE daemon (aned).

Key findings:
- Boot-arg "ane_skipAdapterWeightAccessCheck" bypasses adapter weight entitlement
- isInternalBuild checks /AppleInternal and os_variant_* functions
- All bypass mechanisms require Apple internal builds (not available on consumer macOS)
"""

import ctypes
from ctypes import c_bool, c_char_p, c_void_p, c_int, c_long, c_uint64
import os
from typing import Dict, List, Optional, Any

# Known boot arguments and their purposes
BOOT_ARGS = {
    "ane_skipAdapterWeightAccessCheck": {
        "purpose": "Bypass adapter weight entitlement check",
        "bypasses_entitlement": "com.apple.aned.private.adapterWeight.allow",
        "requires_internal_build": True,
    },
    "ane_vm_allowPrecompiledBinary": {
        "purpose": "Allow precompiled binaries in VM",
        "requires_internal_build": True,
    },
    "ane_vm_debugDumpBootArg": {
        "purpose": "Enable debug dumps on errors",
        "requires_internal_build": True,
    },
    "ane_vm_forceValidationOnGuest": {
        "purpose": "Force extra validation in VM guests",
        "requires_internal_build": True,
    },
}

# Known entitlements and their purposes
ENTITLEMENTS = {
    "com.apple.aned.private.allow": {
        "purpose": "Primary ANE access",
        "required_for": ["compile", "load", "evaluate"],
        "category": "primary",
    },
    "com.apple.aned.private.adapterWeight.allow": {
        "purpose": "Adapter weights access",
        "required_for": ["custom weight loading"],
        "category": "weights",
        "bypass_boot_arg": "ane_skipAdapterWeightAccessCheck",
    },
    "com.apple.aned.private.aggressivePowerSaving.allow": {
        "purpose": "Power saving modes",
        "required_for": ["low-power inference"],
        "category": "power",
    },
    "com.apple.ANECompilerService.allow": {
        "purpose": "Compiler service access",
        "required_for": ["model compilation"],
        "category": "compiler",
    },
    "com.apple.aned.private.processModelShare.allow": {
        "purpose": "Cross-process model sharing",
        "required_for": ["shared inference"],
        "category": "sharing",
    },
    "com.apple.ane.memoryUnwiringOptOutAccess.allow": {
        "purpose": "Memory unwiring control",
        "required_for": ["large model persistence"],
        "category": "memory",
    },
    "com.apple.private.modelPurgeInAllPartitions.allow": {
        "purpose": "Model cache purging",
        "required_for": ["cache management"],
        "category": "cache",
    },
    "com.apple.aned.private.secondaryANECompilerServiceAccess.allow": {
        "purpose": "Secondary compiler access",
        "required_for": ["parallel compilation"],
        "category": "compiler",
    },
    "com.apple.private.ANEStorageMaintainer.allow": {
        "purpose": "Storage maintenance",
        "required_for": ["cache cleanup"],
        "category": "cache",
    },
}

# System paths used by ANE
SYSTEM_PATHS = {
    "internal_library": "/AppleInternal/Library",
    "system_library": "/System/Library",
    "aned_daemon": "/usr/libexec/aned",
    "ane_services": "/System/Library/PrivateFrameworks/ANEServices.framework/ANEServices",
    "ane_compiler": "/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler",
    "apple_neural_engine": "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
}


def _load_ane_services():
    """Load ANEServices framework for runtime introspection."""
    try:
        ane_services = ctypes.CDLL(SYSTEM_PATHS["ane_services"], ctypes.RTLD_GLOBAL)
        return ane_services
    except OSError:
        return None


def _load_libsystem_darwin():
    """Load libsystem_darwin for os_variant functions."""
    try:
        return ctypes.CDLL("/usr/lib/system/libsystem_darwin.dylib", ctypes.RTLD_GLOBAL)
    except (OSError, AttributeError):
        return None


def check_os_variant_internal(subsystem: str = "com.apple.aned") -> Dict[str, bool]:
    """
    Check os_variant functions for internal build detection.
    
    Args:
        subsystem: The subsystem to check (default: "com.apple.aned")
        
    Returns:
        Dictionary with results from each os_variant check
    """
    results = {
        "has_internal_content": False,
        "allows_internal_security_policies": False,
        "has_internal_diagnostics": False,
        "has_internal_ui": False,
    }
    
    lib = _load_libsystem_darwin()
    if not lib:
        return results
    
    checks = [
        ("has_internal_content", "os_variant_has_internal_content"),
        ("allows_internal_security_policies", "os_variant_allows_internal_security_policies"),
        ("has_internal_diagnostics", "os_variant_has_internal_diagnostics"),
        ("has_internal_ui", "os_variant_has_internal_ui"),
    ]
    
    for key, func_name in checks:
        try:
            func = getattr(lib, func_name)
            func.argtypes = [c_char_p]
            func.restype = c_bool
            results[key] = func(subsystem.encode())
        except (AttributeError, OSError):
            pass
    
    return results


def check_internal_build_indicators() -> Dict[str, Any]:
    """
    Check all indicators used to determine if this is an internal build.
    
    Returns:
        Dictionary with all internal build indicators
    """
    indicators = {
        "apple_internal_exists": os.path.exists("/AppleInternal"),
        "apple_internal_library_exists": os.path.exists("/AppleInternal/Library"),
        "os_variant": check_os_variant_internal(),
        "is_internal_build": False,  # Will be set based on indicators
    }
    
    # isInternalBuild is True if ANY indicator is True
    indicators["is_internal_build"] = (
        indicators["apple_internal_exists"] or
        indicators["os_variant"]["has_internal_content"] or
        indicators["os_variant"]["allows_internal_security_policies"]
    )
    
    return indicators


def get_boot_args() -> str:
    """
    Get current boot arguments from sysctl.
    
    Returns:
        Boot arguments string (may be empty)
    """
    import subprocess
    try:
        result = subprocess.run(
            ["sysctl", "-n", "kern.bootargs"],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except Exception:
        return ""


def is_boot_arg_present(arg: str) -> bool:
    """
    Check if a specific boot argument is present.
    
    Args:
        arg: The boot argument name to check
        
    Returns:
        True if the argument is present
    """
    boot_args = get_boot_args()
    return arg in boot_args


def get_ane_device_info() -> Dict[str, Any]:
    """
    Get comprehensive ANE device information.
    
    This requires the ANEServices framework to be loaded with ObjC runtime.
    Falls back to basic info if ObjC is not available.
    
    Returns:
        Dictionary with ANE device information
    """
    info = {
        "available": False,
        "error": None,
    }
    
    try:
        # Try using pyobjc if available
        import objc
        from Foundation import NSBundle
        
        path = SYSTEM_PATHS["ane_services"]
        bundle = NSBundle.bundleWithPath_(path)
        if bundle and bundle.load():
            # Get _ANEDeviceInfo class
            ANEDeviceInfo = objc.lookUpClass("_ANEDeviceInfo")
            
            info["available"] = True
            info["has_ane"] = ANEDeviceInfo.hasANE()
            info["num_anes"] = ANEDeviceInfo.numANEs()
            info["num_cores"] = ANEDeviceInfo.numANECores()
            info["product_name"] = str(ANEDeviceInfo.productName())
            info["build_version"] = str(ANEDeviceInfo.buildVersion())
            info["is_virtual_machine"] = ANEDeviceInfo.isVirtualMachine()
            info["is_internal_build"] = ANEDeviceInfo.isInternalBuild()
            info["precompiled_checks_disabled"] = ANEDeviceInfo.precompiledModelChecksDisabled()
            
            # Check boot-arg status
            ANEStrings = objc.lookUpClass("_ANEStrings")
            if ANEStrings:
                bypass_arg = ANEStrings.adapterWeightsAccessEntitlementBypassBootArg()
                info["bypass_boot_arg"] = str(bypass_arg)
                info["bypass_active"] = ANEDeviceInfo.isBoolBootArgSetTrue_(bypass_arg)
                
    except ImportError:
        info["error"] = "pyobjc not available - install with: pip install pyobjc"
    except Exception as e:
        info["error"] = str(e)
    
    return info


def get_entitlement_requirements() -> Dict[str, List[str]]:
    """
    Get entitlement requirements for different ANE operations.
    
    Returns:
        Dictionary mapping operations to required entitlements
    """
    ops = {
        "compile": [],
        "load": [],
        "evaluate": [],
        "custom_weights": [],
        "power_saving": [],
        "process_sharing": [],
    }
    
    for ent, info in ENTITLEMENTS.items():
        for op in info.get("required_for", []):
            if op in ops:
                ops[op].append(ent)
            elif "compile" in op.lower():
                ops["compile"].append(ent)
            elif "load" in op.lower():
                ops["load"].append(ent)
    
    return ops


def print_ane_security_report():
    """Print a comprehensive ANE security report."""
    print("=" * 60)
    print("ANE Security Report")
    print("=" * 60)
    
    # Internal build status
    print("\n--- Internal Build Detection ---")
    indicators = check_internal_build_indicators()
    print(f"/AppleInternal exists: {indicators['apple_internal_exists']}")
    print(f"/AppleInternal/Library exists: {indicators['apple_internal_library_exists']}")
    for key, value in indicators["os_variant"].items():
        print(f"os_variant_{key}: {value}")
    print(f"\nisInternalBuild would return: {indicators['is_internal_build']}")
    
    # Boot arguments
    print("\n--- Boot Arguments ---")
    boot_args = get_boot_args()
    print(f"Current boot-args: '{boot_args or '(empty)'}'")
    
    print("\nKnown ANE boot arguments:")
    for arg, info in BOOT_ARGS.items():
        present = is_boot_arg_present(arg)
        status = "[ACTIVE]" if present else "[not set]"
        print(f"  {arg}: {status}")
        print(f"    Purpose: {info['purpose']}")
        if "bypasses_entitlement" in info:
            print(f"    Bypasses: {info['bypasses_entitlement']}")
    
    # Entitlements
    print("\n--- Required Entitlements ---")
    for ent, info in sorted(ENTITLEMENTS.items()):
        print(f"\n  {ent}")
        print(f"    Purpose: {info['purpose']}")
        print(f"    Required for: {', '.join(info['required_for'])}")
        if "bypass_boot_arg" in info:
            print(f"    Can bypass with boot-arg: {info['bypass_boot_arg']}")
    
    # Device info (if available)
    print("\n--- ANE Device Info ---")
    try:
        info = get_ane_device_info()
        if info.get("error"):
            print(f"Error: {info['error']}")
        elif info.get("available"):
            print(f"Has ANE: {info['has_ane']}")
            print(f"Number of ANEs: {info['num_anes']}")
            print(f"Number of cores: {info['num_cores']}")
            print(f"Product: {info['product_name']}")
            print(f"Build: {info['build_version']}")
            print(f"Is VM: {info['is_virtual_machine']}")
            print(f"Is internal build: {info['is_internal_build']}")
            if "bypass_boot_arg" in info:
                print(f"Bypass boot-arg: {info['bypass_boot_arg']}")
                print(f"Bypass active: {info['bypass_active']}")
        else:
            print("ANE device info not available")
    except Exception as e:
        print(f"Error getting device info: {e}")
    
    print("\n" + "=" * 60)


# Exports
__all__ = [
    # Constants
    "BOOT_ARGS",
    "ENTITLEMENTS", 
    "SYSTEM_PATHS",
    # Functions
    "check_os_variant_internal",
    "check_internal_build_indicators",
    "get_boot_args",
    "is_boot_arg_present",
    "get_ane_device_info",
    "get_entitlement_requirements",
    "print_ane_security_report",
]


if __name__ == "__main__":
    print_ane_security_report()
