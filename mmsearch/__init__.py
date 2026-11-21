"""Project-wide GPU budget (user rule 2026-11-21: the whole project uses at most 25 GB of the shared card).
Every GPU process of ours runs with MMS_GPU_GB set; importing mmsearch then hard-caps PyTorch's allocator, so a
job that would exceed its lane's share fails with OOM instead of eating other users' memory."""
import os

if os.environ.get("MMS_GPU_GB"):
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["MMS_GPU_GB"]) * 2**30 / total))
    except ImportError:
        pass
