# device_manager.pseudo
# Centralized (device, dtype) control for the whole system.

import torch, contextlib
from typing import Optional, Union

# --- Global policy (override at startup) ---
_DEV  = torch.device("cpu")  # Default to CPU, will be set by initialize_device()
_DTYPE = torch.float64  # Default to float64 for maximum precision, can be overridden by set_dtype()

def initialize_device(device_arg: str = "gpu") -> torch.device:
    """
    Initialize device from command-line argument with GPU-first fallback logic.
    
    Args:
        device_arg: Device specification from command line ('cpu', 'cuda', 'cuda:0', etc.)
    
    Returns:
        torch.device: The actual device that will be used
    """
    if device_arg == "cuda":
        # If user specified 'cuda', try cuda:0 first
        return set_device("cuda:0")
    elif device_arg.startswith("cuda:"):
        # User specified specific GPU
        return set_device(device_arg) 
    elif device_arg == "cpu":
        # User explicitly requested CPU
        return set_device("cpu")
    else:
        # Fallback to CPU for unknown device specs
        print(f"Unknown device specification '{device_arg}', using CPU")
        return set_device("cpu")

def get_device() -> torch.device: return _DEV
def get_dtype()  -> torch.dtype:  return _DTYPE

def set_device(dev=None) -> torch.device:
    global _DEV
    if dev is None: dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    if isinstance(dev, str):
        if dev.startswith("cuda") and not torch.cuda.is_available(): 
            print(f"CUDA not available, using CPU")
            _DEV = torch.device("cpu")
        elif dev.startswith("cuda"):
            # Try to use CUDA device with retry logic for busy conditions
            try:
                test_device = torch.device(dev)
                # Test if device is actually usable by creating a small tensor
                test_tensor = torch.zeros(1, device=test_device)
                del test_tensor  # Clean up
                _DEV = test_device
                print(f"Successfully initialized CUDA device: {dev}")
            except torch.cuda.OutOfMemoryError as e:
                print(f"CUDA device {dev} out of memory ({e}), falling back to CPU")
                _DEV = torch.device("cpu")
            except torch.AcceleratorError as e:
                if "busy" in str(e).lower():
                    print(f"Warning: CUDA device {dev} appears busy but proceeding anyway. Error: {e}")
                    _DEV = test_device  # Proceed with CUDA despite "busy" warning
                else:
                    print(f"CUDA device {dev} error ({e}), falling back to CPU")
                    _DEV = torch.device("cpu")
            except RuntimeError as e:
                print(f"CUDA device {dev} runtime error ({e}), falling back to CPU")
                _DEV = torch.device("cpu")
        else: 
            _DEV = torch.device(dev)
    elif isinstance(dev, torch.device):
        if dev.type == "cuda" and not torch.cuda.is_available(): 
            print(f"CUDA not available, using CPU")
            _DEV = torch.device("cpu")
        elif dev.type == "cuda":
            # Similar test for torch.device objects
            try:
                test_tensor = torch.zeros(1, device=dev)
                del test_tensor
                _DEV = dev
                print(f"Successfully initialized CUDA device: {dev}")
            except torch.cuda.OutOfMemoryError as e:
                print(f"CUDA device {dev} out of memory ({e}), falling back to CPU")
                _DEV = torch.device("cpu")
            except torch.AcceleratorError as e:
                if "busy" in str(e).lower():
                    print(f"Warning: CUDA device {dev} appears busy but proceeding anyway. Error: {e}")
                    _DEV = dev  # Proceed with CUDA despite "busy" warning
                else:
                    print(f"CUDA device {dev} error ({e}), falling back to CPU")
                    _DEV = torch.device("cpu")
            except RuntimeError as e:
                print(f"CUDA device {dev} runtime error ({e}), falling back to CPU")
                _DEV = torch.device("cpu")
        else: 
            _DEV = dev
    else:
        raise TypeError("bad device spec")
    return _DEV

def set_dtype(dt: torch.dtype) -> torch.dtype:
    global _DTYPE
    _DTYPE = dt
    torch.set_default_dtype(dt)
    return _DTYPE

def as_t(x, *, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None) -> torch.Tensor:
    global _DEV  # Declare global at function start
    device = device or _DEV; dtype = dtype or _DTYPE
    
    # If device is CUDA, try to create tensor on CUDA first
    if device.type == "cuda":
        try:
            if isinstance(x, torch.Tensor):
                return x.to(device=device, dtype=dtype)
            else:
                return torch.as_tensor(x, device=device, dtype=dtype)
        except (torch.cuda.OutOfMemoryError, torch.AcceleratorError, RuntimeError) as e:
            print(f"CUDA operation failed ({e}), falling back to CPU for this operation")
            # Fall back to CPU for this specific operation
            cpu_device = torch.device("cpu")
            if isinstance(x, torch.Tensor):
                return x.to(device=cpu_device, dtype=dtype)
            else:
                return torch.as_tensor(x, device=cpu_device, dtype=dtype)
    else:
        # For CPU or other devices, proceed normally
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        else:
            return torch.as_tensor(x, device=device, dtype=dtype)

def to_device(x: torch.Tensor, *, device=None, dtype=None) -> torch.Tensor:
    return x.to(device=device or _DEV, dtype=dtype or _DTYPE)

@contextlib.contextmanager
def device_scope(device=None, dtype=None):
    prev_d, prev_t = _DEV, _DTYPE
    if device is not None: set_device(device)
    if dtype  is not None: set_dtype(dtype)
    try: yield
    finally:
        set_device(prev_d); set_dtype(prev_t)

def wrap_model_fn(module: "torch.nn.Module"):
    mod = module.to(device=_DEV, dtype=_DTYPE)
    @torch.no_grad()
    def f(x: torch.Tensor) -> torch.Tensor:
        return mod(to_device(x))
    return f

def summary() -> str:
    return f"device={_DEV}, dtype={_DTYPE}, cuda_available={torch.cuda.is_available()}"
