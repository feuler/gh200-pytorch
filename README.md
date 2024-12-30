# GH200 pytorch uvm patches

(wip - work in progress)

Patch files for to enable uvm (unified virtual memory) in pytorch.
Torch based applications can run with the PYTORCH_CUDA_ALLOC_CONF parameter below to use the GH200 unified CUDA memory.

Original patch source (working for pytorch versions <= 2.2.2): https://github.com/pytorch/pytorch/compare/main...0x804d8000:pytorch:uvm

Updated patch: v2

## ToDo
- update v1 patch (increase oversubscription to sth bigger than 1.33x)
- properly test v2 patch and update for pytorch 2.6

#### Example usage (v1 patch):
```
export PYTORCH_CUDA_ALLOC_CONF=use_uvm:True 
# python my-app.py
python3 -m vllm.entrypoints.openai.api_server
```

#### v2 patch adds oversubscription_ratio (default 5x for GH200) and preferred uvm access patterns option:

  - gpu_first,      // Prefer GPU memory, fallback to CPU
  - balanced,       // No preference, let driver decide
  - cpu_first       // Prefer CPU memory, migrate as needed

#### Example usage (v2 patch):
```
export PYTORCH_CUDA_ALLOC_CONF='use_uvm:True,uvm_oversubscription_ratio:5.0,uvm_access_pattern:gpu_first'
#python my-app.py
python3 -m vllm.entrypoints.openai.api_server
```

#### uvm patch - dry run
patch -p0 --dry-run < patch_pytorch-2.5.1_uvm-v2.patch

#### apply uvm patch
patch -p0 < patch_pytorch-2.5.1_uvm-v2.patch

# Possible alternative - PyTorch pluggable allocator

- Downside Doesn't cover every scenario (cuda graphs, ...)

Example:

```python
class CudaMallocManagedAllocator(memory.Allocator):
    def __init__(self):
        super().__init__()
        
    def malloc(self, size: int):
        ptr = ctypes.c_void_p()
        torch._C._cuda_check_error(torch._C._cuda_malloc_managed(ctypes.byref(ptr), size))
        return ptr.value
        
    def free(self, ptr: int):
        if ptr:
            torch._C._cuda_check_error(torch._C._cuda_free(ptr))

def setup_managed_memory():
    """Set up CUDA managed memory allocation"""
    custom_allocator = CudaMallocManagedAllocator()
    torch.cuda.memory.set_allocator(custom_allocator)
```


