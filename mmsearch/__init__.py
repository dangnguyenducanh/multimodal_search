"""Project-wide GPU budget (user rule 2026-11-21: the whole project uses at most 25 GB of the shared card).
GPU jobs run with MMS_GPU_GB set and call cap_gpu() before using CUDA, which hard-caps PyTorch's allocator, so a job
that would exceed its share fails with OOM instead of eating other users' memory. Lazy on purpose: a parent process
that only imports mmsearch (ladder, queue drivers) must not open its own CUDA context next to the capped child."""
import os


def cap_gpu():
    if not os.environ.get("MMS_GPU_GB"):
        return
    import torch
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["MMS_GPU_GB"]) * 2**30 / total))
