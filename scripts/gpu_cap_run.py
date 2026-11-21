"""Run a third-party training script under the project's GPU cap without editing it:
  MMS_GPU_GB=15 python scripts/gpu_cap_run.py run_glue_no_trainer.py --args ...
Sets PyTorch's per-process memory fraction, then executes the script as __main__ with its own argv."""
import os
import runpy
import sys

import torch

if torch.cuda.is_available() and os.environ.get("MMS_GPU_GB"):
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["MMS_GPU_GB"]) * 2**30 / total))
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
