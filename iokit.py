"""
ANE IOKit Interface - Direct hardware access to Apple Neural Engine

This module provides direct IOKit access to ANE hardware, bypassing the XPC daemon.
Based on reverse engineering of the ANEServices framework and AppleH11ANEInterface kext.

Key findings:
- H1xANELoadBalancer (type=1) and H11ANEIn (type=1,4) user clients are accessible
- IOServiceOpen succeeds without entitlements - creating a user client connection works!
- Selector 1 (DeviceClose) works without any input
- DeviceOpen (selector 0) returns kIOReturnNotFound with correct 0x68 byte struct
- All other selectors return kIOReturnBadArgument (require DeviceOpen first)

Current Blocker:
The kext's DeviceOpen (selector 0) returns kIOReturnNotFound even with a valid 0x68 byte
ANEDeviceInfo struct. This appears to be due to the `com.apple.ane.iokit-user-access`
entitlement check - only the aned daemon has this entitlement.

ANEDeviceInfo struct layout (0x68 = 104 bytes):
  - Offset 0x00-0x17: Unknown fields (zeroed on default init)
  - Offset 0x18: int32_t field (set to -1 on default init)
  - Offset 0x1c: uint8_t deviceType (read back after open)
  - Offset 0x4c: uint32_t version (ANE version, e.g., 0x90 for ANE15)
  - Offset 0x50: uint32_t minorVersion

Required Entitlement (from aned binary):
  com.apple.ane.iokit-user-access: true

User Client Types:
- H1xANELoadBalancer type=1: ANEServicesDevice client (program management)
- H11ANEIn type=1: ANEHWDevice client (hardware control)  
- H11ANEIn type=4: DirectPath client (direct execution?)

Selector Maps (from ANEServices disassembly):
See ANESERVICES_SELECTORS and ANEHW_SELECTORS below.
"""

import ctypes
from ctypes import c_int, c_uint32, c_uint64, c_void_p, c_size_t, byref, POINTER, Structure
from typing import Optional, Dict, Any, Tuple, List

# Load IOKit framework
try:
    _iokit = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
    _iokit_available = True
except OSError:
    _iokit = None
    _iokit_available = False

# IOKit types
io_service_t = c_uint32
io_connect_t = c_uint32
io_iterator_t = c_uint32
kern_return_t = c_int
mach_port_t = c_uint32

# IOKit constants
kIOMainPortDefault = 0
KERN_SUCCESS = 0
kIOReturnSuccess = 0
kIOReturnBadArgument = 0xe00002c2
kIOReturnUnsupported = 0xe00002c7
kIOReturnNotFound = 0xe00002f0
kIOReturnNotPrivileged = 0xe00002cd

# User client types discovered via testing
USER_CLIENT_TYPES = {
    "H1xANELoadBalancer": {
        1: "ANEServicesDevice client (program management)",
    },
    "H11ANEIn": {
        1: "ANEHWDevice client (hardware control)",
        4: "DirectPath client (direct execution?)",
    },
}

# Selector map for H1xANELoadBalancer (ANEServicesDevice)
# Format: selector -> (name, method_type, input_size, output_size)
ANESERVICES_SELECTORS = {
    0x00: ("DeviceOpen/ProgramCreate_Base", "struct", 0x68, 0x68),
    0x01: ("DeviceClose", "scalar", 0, 0),
    0x02: ("ProgramSendRequest", "async", None, None),
    0x03: ("ProgramCreate", "struct", 0x20, 0),
    0x04: ("ProgramInputsReady", "struct", 0x38, 8),
    0x05: ("ProgramPrepare", "struct", None, None),
    0x06: ("ProgramDestroy", "struct", None, None),
    0x07: ("GetStatus", "struct", 0, 8),
    0x08: ("ProgramChainingSetActiveProcedure", "struct", 0x20, 0),
    0x09: ("ProgramMemoryUnmap", "struct", 0x10, 8),
    0x0a: ("GetVersion", "scalar", 0, 1),
    0x0b: ("RegisterDebugWorkProcessor", "async", None, None),
    0x0c: ("UnregisterDebugWorkProcessor", "async", None, None),
    0x0d: ("GetDebugWorkProcessorItem", "async", None, None),
    0x0e: ("CompleteDebugWorkProcessorItem", "async", None, None),
    0x0f: ("ReleaseDebugWorkProcessorBuffers", "async", None, None),
    0x10: ("LoadFirmware", "scalar", 3, 0),
}

# Selector map for H11ANEIn (ANEHWDevice)
ANEHW_SELECTORS = {
    0x00: ("DeviceOpen", "struct", 0x68, 0x68),
    0x01: ("DeviceClose", "scalar", 0, 0),
    0x02: ("GetStatus", "struct", 0, 8),
    0x03: ("LoadFirmware", "struct", 0, 0),
    0x04: ("SendCommand", "struct", None, None),
    0x05: ("IsPowered", "scalar", 0, 1),
    0x06: ("PowerOn", "scalar", 0, 0),
    0x07: ("PowerOff/ForgetFirmware", "scalar", 0, 0),
    0x08: ("UnmapDartBuffers", "struct", 0x20, 0),
    0x09: ("SetPowerManagement", "scalar", 1, 0),
    0x0a: ("SetDynamicPowerGating", "scalar", 1, 0),
    0x0b: ("SetPowerGatingHysteresisTime", "scalar", 1, 0),
    0x0c: ("GetTime", "struct", 0, 8),
    0x0d: ("SetDriverProperty", "scalar", 2, 0),
    0x0e: ("ShowSharedMemoryAllocations", "scalar", 0, 0),
    0x0f: ("ShowModelMemoryStatus", "scalar", 0, 0),
    0x10: ("SetDARTCacheTTL", "scalar", 1, 0),
    0x11: ("SetFirmwareBootArg", "scalar", 2, 0),
    0x12: ("SetThrottlingPercentage", "scalar", 1, 0),
    0x13: ("AddPersistentClient", "scalar", 0, 0),
    0x14: ("RemovePersistentClient", "scalar", 0, 0),
    0x15: ("CreateClientLoggingSession", "scalar", 2, 1),
    0x16: ("TerminateClientLoggingSession", "scalar", 1, 0),
    0x17: ("GetDriverProperty", "scalar", 1, 1),
    0x18: ("FlushInactiveDARTMappings", "scalar", 0, 0),
    0x19: ("GetVersion", "scalar", 0, 1),
    0x1a: ("ReadANERegister", "scalar", 1, 1),
    0x1b: ("WriteANERegister", "scalar", 2, 0),
    0x1e: ("GetClientsInfo", "struct", 0, None),
    0x1f: ("MPMMemoryMapRequest", "scalar", 2, 0),
    0x20: ("MPMMemoryUnmapRequest", "scalar", 1, 0),
}


def _setup_iokit_functions():
    """Set up IOKit function signatures."""
    if not _iokit_available:
        return
    
    # IOServiceGetMatchingService
    _iokit.IOServiceGetMatchingService.argtypes = [mach_port_t, c_void_p]
    _iokit.IOServiceGetMatchingService.restype = io_service_t
    
    # IOServiceMatching
    _iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
    _iokit.IOServiceMatching.restype = c_void_p
    
    # IOServiceOpen
    _iokit.IOServiceOpen.argtypes = [io_service_t, mach_port_t, c_uint32, POINTER(io_connect_t)]
    _iokit.IOServiceOpen.restype = kern_return_t
    
    # IOServiceClose
    _iokit.IOServiceClose.argtypes = [io_connect_t]
    _iokit.IOServiceClose.restype = kern_return_t
    
    # IOObjectRelease
    _iokit.IOObjectRelease.argtypes = [c_uint32]
    _iokit.IOObjectRelease.restype = kern_return_t
    
    # IOConnectCallScalarMethod
    _iokit.IOConnectCallScalarMethod.argtypes = [
        io_connect_t, c_uint32,
        POINTER(c_uint64), c_uint32,  # input
        POINTER(c_uint64), POINTER(c_uint32)  # output
    ]
    _iokit.IOConnectCallScalarMethod.restype = kern_return_t
    
    # IOConnectCallStructMethod
    _iokit.IOConnectCallStructMethod.argtypes = [
        io_connect_t, c_uint32,
        c_void_p, c_size_t,  # input
        c_void_p, POINTER(c_size_t)  # output
    ]
    _iokit.IOConnectCallStructMethod.restype = kern_return_t


# Initialize IOKit functions
_setup_iokit_functions()


def _mach_task_self() -> int:
    """Get current task port."""
    libc = ctypes.CDLL(None)
    return libc.mach_task_self()


def iokit_error_string(kr: int) -> str:
    """Convert IOKit error code to string."""
    errors = {
        0: "KERN_SUCCESS",
        0xe00002bc: "kIOReturnError",
        0xe00002bd: "kIOReturnNoMemory", 
        0xe00002be: "kIOReturnNoResources",
        0xe00002c2: "kIOReturnBadArgument",
        0xe00002c7: "kIOReturnUnsupported",
        0xe00002cd: "kIOReturnNotPrivileged",
        0xe00002f0: "kIOReturnNotFound",
    }
    return errors.get(kr, f"0x{kr:08x}")


class ANEIOKitClient:
    """
    Direct IOKit client for ANE hardware.
    
    This provides low-level access to the ANE IOKit user clients.
    Note that most operations require valid input structs that match
    internal kernel state, which typically requires the aned daemon.
    
    Example:
        client = ANEIOKitClient()
        if client.open("H11ANEIn", 1):
            # Call DeviceClose (selector 1) - this works!
            success = client.call_scalar(1)
            client.close()
    """
    
    def __init__(self):
        self._connect: int = 0
        self._service_name: str = ""
        self._client_type: int = 0
        
        if not _iokit_available:
            raise RuntimeError("IOKit not available")
    
    def open(self, service_name: str, client_type: int) -> bool:
        """
        Open a connection to an ANE IOKit service.
        
        Args:
            service_name: IOKit service name (e.g., "H11ANEIn", "H1xANELoadBalancer")
            client_type: User client type (typically 1 or 4)
            
        Returns:
            True if connection opened successfully
        """
        if self._connect:
            self.close()
        
        # Get matching service
        matching = _iokit.IOServiceMatching(service_name.encode())
        if not matching:
            return False
        
        service = _iokit.IOServiceGetMatchingService(kIOMainPortDefault, matching)
        if not service:
            return False
        
        # Open user client
        connect = io_connect_t()
        task = _mach_task_self()
        kr = _iokit.IOServiceOpen(service, task, client_type, byref(connect))
        _iokit.IOObjectRelease(service)
        
        if kr != KERN_SUCCESS:
            return False
        
        self._connect = connect.value
        self._service_name = service_name
        self._client_type = client_type
        return True
    
    def close(self):
        """Close the IOKit connection."""
        if self._connect:
            _iokit.IOServiceClose(self._connect)
            self._connect = 0
    
    def call_scalar(self, selector: int, 
                    inputs: Optional[List[int]] = None,
                    output_count: int = 0) -> Tuple[int, List[int]]:
        """
        Call an IOKit scalar method.
        
        Args:
            selector: Method selector number
            inputs: List of input scalar values (uint64)
            output_count: Number of output scalars expected
            
        Returns:
            Tuple of (return_code, output_values)
        """
        if not self._connect:
            return (kIOReturnNotFound, [])
        
        # Set up inputs
        if inputs:
            input_array = (c_uint64 * len(inputs))(*inputs)
            input_ptr = ctypes.cast(input_array, POINTER(c_uint64))
            input_count = len(inputs)
        else:
            input_ptr = None
            input_count = 0
        
        # Set up outputs
        if output_count > 0:
            output_array = (c_uint64 * output_count)()
            output_ptr = ctypes.cast(output_array, POINTER(c_uint64))
            out_count = c_uint32(output_count)
            out_count_ptr = byref(out_count)
        else:
            output_ptr = None
            out_count_ptr = None
        
        kr = _iokit.IOConnectCallScalarMethod(
            self._connect, selector,
            input_ptr, input_count,
            output_ptr, out_count_ptr
        )
        
        outputs = []
        if kr == KERN_SUCCESS and output_count > 0:
            outputs = [output_array[i] for i in range(out_count.value)]
        
        return (kr, outputs)
    
    def call_struct(self, selector: int,
                    input_data: Optional[bytes] = None,
                    output_size: int = 0) -> Tuple[int, bytes]:
        """
        Call an IOKit struct method.
        
        Args:
            selector: Method selector number
            input_data: Input struct data as bytes
            output_size: Size of output buffer
            
        Returns:
            Tuple of (return_code, output_data)
        """
        if not self._connect:
            return (kIOReturnNotFound, b"")
        
        # Set up input
        if input_data:
            input_buf = ctypes.create_string_buffer(input_data, len(input_data))
            input_ptr = ctypes.cast(input_buf, c_void_p)
            input_size = len(input_data)
        else:
            input_ptr = None
            input_size = 0
        
        # Set up output
        if output_size > 0:
            output_buf = ctypes.create_string_buffer(output_size)
            output_ptr = ctypes.cast(output_buf, c_void_p)
            out_size = c_size_t(output_size)
        else:
            output_ptr = None
            out_size = c_size_t(0)
        
        kr = _iokit.IOConnectCallStructMethod(
            self._connect, selector,
            input_ptr, input_size,
            output_ptr, byref(out_size) if output_size else None
        )
        
        output = b""
        if kr == KERN_SUCCESS and output_size > 0:
            output = output_buf.raw[:out_size.value]
        
        return (kr, output)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    @property
    def is_open(self) -> bool:
        return self._connect != 0
    
    @property
    def connection(self) -> int:
        return self._connect


def test_ane_iokit_access() -> Dict[str, Any]:
    """
    Test which ANE IOKit services are accessible.
    
    Returns:
        Dictionary with test results
    """
    results = {
        "iokit_available": _iokit_available,
        "services": {},
    }
    
    if not _iokit_available:
        return results
    
    services_to_test = [
        ("H1xANELoadBalancer", [1]),
        ("H11ANEIn", [1, 4]),
    ]
    
    for service_name, types in services_to_test:
        service_results = {}
        
        for client_type in types:
            client = ANEIOKitClient()
            if client.open(service_name, client_type):
                # Test selector 1 (DeviceClose) which is known to work
                kr, _ = client.call_scalar(1)
                service_results[client_type] = {
                    "open": True,
                    "selector_1_works": kr == KERN_SUCCESS,
                }
                client.close()
            else:
                service_results[client_type] = {"open": False}
        
        results["services"][service_name] = service_results
    
    return results


def list_available_selectors(service_name: str) -> Dict[int, Dict]:
    """
    List known selectors for a service.
    
    Args:
        service_name: "H1xANELoadBalancer" or "H11ANEIn"
        
    Returns:
        Dictionary mapping selector numbers to info
    """
    if service_name == "H1xANELoadBalancer":
        return {k: {"name": v[0], "type": v[1], "in_size": v[2], "out_size": v[3]} 
                for k, v in ANESERVICES_SELECTORS.items()}
    elif service_name == "H11ANEIn":
        return {k: {"name": v[0], "type": v[1], "in_size": v[2], "out_size": v[3]} 
                for k, v in ANEHW_SELECTORS.items()}
    else:
        return {}


def print_ane_iokit_report():
    """Print a report of ANE IOKit access status."""
    print("=" * 60)
    print("ANE IOKit Direct Access Report")
    print("=" * 60)
    
    results = test_ane_iokit_access()
    
    print(f"\nIOKit available: {results['iokit_available']}")
    
    if not results['iokit_available']:
        print("Cannot test IOKit - not available")
        return
    
    print("\n--- User Client Access ---")
    for service_name, types in results['services'].items():
        print(f"\n{service_name}:")
        for client_type, info in types.items():
            if info['open']:
                sel1 = "YES" if info.get('selector_1_works') else "NO"
                print(f"  Type {client_type}: Open=YES, Selector1={sel1}")
            else:
                print(f"  Type {client_type}: Open=NO")
    
    print("\n--- Known Selectors ---")
    print("\nH1xANELoadBalancer (ANEServicesDevice):")
    for sel, info in sorted(ANESERVICES_SELECTORS.items()):
        print(f"  {sel:#04x}: {info[0]:40s} ({info[1]})")
    
    print("\nH11ANEIn (ANEHWDevice):")
    for sel, info in sorted(ANEHW_SELECTORS.items()):
        print(f"  {sel:#04x}: {info[0]:40s} ({info[1]})")
    
    print("\n" + "=" * 60)


# Exports
__all__ = [
    "ANEIOKitClient",
    "ANESERVICES_SELECTORS",
    "ANEHW_SELECTORS",
    "USER_CLIENT_TYPES",
    "test_ane_iokit_access",
    "list_available_selectors",
    "print_ane_iokit_report",
    "iokit_error_string",
    "KERN_SUCCESS",
    "kIOReturnBadArgument",
    "kIOReturnUnsupported",
    "kIOReturnNotFound",
    "kIOReturnNotPrivileged",
]


if __name__ == "__main__":
    print_ane_iokit_report()
