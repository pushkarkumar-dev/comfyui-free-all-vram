# Free All VRAM for ComfyUI

Makes ComfyUI's **Unload Models** button free *all* of the VRAM, including the memory
it can't normally reach.

Some custom nodes run their models in a separate worker process: isolated runtimes
such as TTS Audio Suite's Qwen3-ASR/TTS, nodes built on
[comfy-env](https://github.com/PozzettiAndrea/comfy-env) such as Depth Anything V3,
multiprocessing workers, or a local server a node starts. ComfyUI's unload can't free
those processes, so they keep several GB of VRAM until you restart ComfyUI. Other nodes
keep a model in a global variable, where unloading never reaches it.

With this node pack installed, the buttons you already use free everything.

## What it does

Every unload goes through the same cleanup, always **between** jobs, never during one:
- the **Unload Models** and **Free model and node cache** buttons
- `POST /free` from the API
- an idle timer
- an optional node

The cleanup does six things:
1. **ComfyUI's own unload,** unchanged: models, text encoders, VAEs, ControlNets,
   upscalers and so on.
2. **Clears the node cache,** which is where many custom nodes keep their models.
3. **Stops GPU worker processes started by this ComfyUI.** The node starts a fresh
   worker the next time you use it, at the cost of a few seconds of model loading.
4. **Stops orphaned workers** left on the GPU by a ComfyUI that crashed or was
   restarted. It also does this once at startup.
5. **Reports models a custom node keeps on the GPU for itself,** in its module globals,
   caches, or class or object attributes. The report names the package and where the
   model is held, e.g. `CACHE['sdxl']`. You can opt in, per package, to move those models
   to RAM on unload; they move back automatically the next time they're used.
6. **Returns PyTorch's cached memory** to the driver.

Each unload logs one line:

```
[Free All VRAM] GPU memory in use 11.3 -> 3.4 GB. Stopped worker process(es): pid 19908 qwen3_asr_worker.py (tts_audio_suite), pid 6068 persistent_worker.py (comfy-env).
[Free All VRAM] still on the GPU, kept by custom nodes: Sequential 2.00 GB held by some_node_pack (_MODEL). ...
```

## Install

- **ComfyUI Manager:** choose **Install via Git URL** and paste
  `https://github.com/pushkarkumar-dev/comfyui-free-all-vram`.
- **Manually:**
  ```
  cd ComfyUI/custom_nodes
  git clone https://github.com/pushkarkumar-dev/comfyui-free-all-vram
  pip install -r comfyui-free-all-vram/requirements.txt
  ```
  For the portable build, run that last line with
  `python_embeded\python.exe -m pip ...` instead of `pip`.

Restart ComfyUI. The console should show
`[Free All VRAM] active: Unload Models also stops GPU worker processes, ...`.

## Use

- **Unload Models** or **Free model and node cache:** the normal buttons, as before.
- **From the API:** `POST /free` with `{"unload_models": true}`. This is ComfyUI's
  standard endpoint; the node pack doesn't change it.
- **Automatically:** after 10 idle minutes, if a worker process is holding the GPU.
- **Free All VRAM (incl. worker processes)** node, under *utils*: queue it between two
  heavy workflows to free memory at exactly that point. Its output shows what it's
  about to stop. There's a ready-made workflow in `example_workflows/`, which also
  appears in ComfyUI's template browser.
- **What's holding VRAM right now:** `http://127.0.0.1:8188/free_all_vram/status`
  returns this as JSON. It's safe to poll at any time, including during a job.
  - `gpu_used_gb`: memory in use on the whole card, all processes
  - `comfy_loaded_models`
  - `worker_processes`
  - `orphaned_workers`
  - `last_unload`: the result of the most recent cleanup: when it ran, memory before
    and after, stopped workers, models moved to RAM, and leftover models
  - the active settings

## Settings

The settings are in `config.json`, in this folder. It's created on first start and
re-read on every unload, so changes need no restart.

| Setting | Default | Meaning |
|---|---|---|
| `stop_workers_on_unload` | `true` | Stop GPU worker processes and orphaned workers on unload. |
| `clear_node_cache_on_unload` | `true` | Also clear the node cache. It is always cleared when a worker was stopped, because cached outputs may refer to it. |
| `scan_leftovers` | `true` | Look for models custom nodes keep on the GPU. Takes milliseconds. |
| `offload_leftovers_from` | `[]` | Package folder names (as shown in the log) whose leftover models are moved to RAM on unload, or `["*"]` for all of them. |
| `auto_free_after_idle_minutes` | `10` | Unload after this many idle minutes following a job. `0` turns it off. |
| `auto_free_only_if_workers` | `true` | Only auto-unload when a worker process is holding the GPU. |
| `never_stop` | `[]` | Processes that must never be stopped, matched as text against their command line, e.g. `["ollama"]`. |

## What gets stopped, and what never does

- **Stopped:** only processes the NVIDIA driver lists as using a GPU, and only when one
  of these is true:
  - it is a child or grandchild of this ComfyUI, or
  - it is an orphan: its parent process is gone, *and* its command line runs from a
    `custom_nodes` folder or is a comfy-env worker.
- **Never stopped:**
  - other programs, even ones that use the same Python as ComfyUI;
  - other ComfyUI servers, and their workers while they're running;
  - anything matching `never_stop`;
  - anything at all while a job is running.

Models a node keeps for itself are only *reported* unless you opt in, because moving
them can confuse a node that drives its model in an unusual way. If a node breaks
after you opt in, remove it from `offload_leftovers_from` and restart ComfyUI.

## Compatibility

- **Tested on:** Windows 11, ComfyUI 0.42, PyTorch 2.11 (CUDA 13.0), NVIDIA RTX 5090,
  with TTS Audio Suite (Qwen3-ASR) and ComfyUI-DepthAnythingV3 (comfy-env).
- **Linux:** should work, but not tested yet. Reports are welcome.
- **No NVIDIA driver** (AMD, Apple, CPU): worker processes can't be detected, so steps 3
  and 4 are skipped. Everything else still works.
- **If a ComfyUI update changes the internals this relies on,** the node pack turns
  itself off with a warning, and the normal unload keeps working. Please open an issue
  if that happens.
- **If it's installed twice,** the second copy does nothing.

## Known messages

- **"Subprocess worker died; model '...' is no longer available. Please reload the
  model node."** comfy-env prints this traceback ("Exception ignored in
  ModelPatcher.__del__") the next time you use a node whose worker was stopped. It's
  harmless: the node starts a new worker and runs normally.

## Uninstall

Delete the `comfyui-free-all-vram` folder and restart ComfyUI. Unload Models goes back
to ComfyUI's stock behaviour. To switch off individual parts without uninstalling,
change the settings above.

## Changelog

- **1.0.1**
  - **Crash fix.** 1.0.0 searched for leftover models by scanning every object in the
    process, and the status page ran that scan even during a job. This could make the
    running job fail with `SystemError: ... tupleobject.c ... bad argument to internal
    function`, or abort ComfyUI. Leftovers are now found through each custom node
    package's own globals, which is safe and takes milliseconds. The status page never
    scans; it reports the last unload instead.
  - The idle timer now also notices jobs shorter than its 30-second check interval.
- **1.0.0:** first release.

## License

GPL-3.0, the same as ComfyUI.
