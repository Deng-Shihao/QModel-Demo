# Offload Workflow Overview

The offload utilities in `nanomodel/utils/offload.py` orchestrate how large modules are swapped between GPU memory, CPU memory, and disk using Accelerate hooks.

## Preparing Modules for Disk Offload
`offload_to_disk` accepts either a single module or a list of dotted paths inside a loaded model. It resolves each module (handling `NamedModule` wrappers) and routes to `_offload_disk` (`nanomodel/utils/offload.py:125`). Small leaf modules (<4KB) are skipped to avoid wasteful I/O.

`_offload_disk` inspects the module’s device, switches CUDA contexts when necessary, and totals the tensor payload size. Before calling Accelerate it prepares a per-module folder (`_prepare_offload_directory`) and saves a safetensors bundle plus metadata index via `_bundle_module_state_dict` (`nanomodel/utils/offload.py:83`). This produces an offline copy of parameter values so disk-backed hooks can reload them later.

With the directory ready, it invokes `accelerate.disk_offload` to register hooks that lazily pull tensors from disk (`nanomodel/utils/offload.py:201`). Tied embedding weights are re-tied afterwards so shared pointers stay intact when offloading submodules of language models.

## Auxiliary Helpers
Support routines cover housekeeping: `_tensor_nbytes` guards runtime element size lookups for bfloat16 tensors; `is_meta_module` detects meta tensors; and `get_module_fullname`/`set_submodule` provide dotted-path introspection (`nanomodel/utils/offload.py:35`). A global lock and `_fake_clear_device_cache` stub ensure multi-threaded safety with Accelerate’s global hooks.

## Restoring Modules
`undo_offload_to_disk` removes Accelerate hooks and materializes tensors back into RAM (`nanomodel/utils/offload.py:143-233`). It iterates over submodules, collecting potential offload directories for optional cleanup. If Accelerate exposes a weights map, `_restore_leaves_from_weights_map` reads tensors directly; otherwise `_maybe_align` temporarily aligns devices so clones can be taken without re-triggering hooks. Clones are created through `_clone_into_parameter` and `_clone_into_buffer`, ensuring tensors detach from memory-mapped storage before being reattached to the module.

After all leaves are restored, the function removes hooks via `remove_hook_from_submodules` and `remove_hook_from_module`. Tied embeddings are synchronized once more, and discovered offload folders are deleted if the caller requests it. The result is an ordinary PyTorch module ready for further computation without any disk-backed hooks.
`undo_offload_to_disk` removes Accelerate hooks and materializes tensors back into RAM (`nanomodel/utils/offload.py:322`). It iterates over submodules, collecting potential offload directories for optional cleanup. If Accelerate exposes a weights map, `_restore_leaves_from_weights_map` reads tensors directly; otherwise `_maybe_align` temporarily aligns devices so clones can be taken without re-triggering hooks. Clones are created through `_clone_into_parameter` and `_clone_into_buffer`, ensuring tensors detach from memory-mapped storage before being reattached to the module.

After all leaves are restored, the function removes hooks via `remove_hook_from_submodules` and `remove_hook_from_module`. Tied embeddings are synchronized once more, and discovered offload folders are deleted if the caller requests it. The result is an ordinary PyTorch module ready for further computation without any disk-backed hooks.