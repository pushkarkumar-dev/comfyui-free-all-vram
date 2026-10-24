"""Free All VRAM - make ComfyUI's "Unload Models" free everything, including what ComfyUI can't see.

"Unload Models" / "Free model and node cache" (and POST /free, the idle timer and the
"Free All VRAM" node) do this, always between prompts:
  1. ComfyUI's own unload: diffusion models, text encoders, VAEs, ControlNets, upscalers, ...
  2. Stop child processes of this ComfyUI that hold the GPU: isolated runtimes, comfy-env
     workers, multiprocessing workers, servers a node started. Their nodes start a fresh worker
     the next time they run.
  3. Stop orphaned custom-node worker processes on the GPU whose parent process is gone.
  4. Clear the node cache (node outputs and node instances, where many custom nodes keep models).
  5. Find models custom nodes still keep on the GPU in this process (module-level caches) and log
     them with the owning package. Packages listed in "offload_leftovers_from" are moved to system
     RAM; each one moves back to the GPU by itself the next time it runs.
  6. Hand PyTorch's cached VRAM back to the driver.

Settings: config.json next to this file, re-read on every unload. Live report: GET /free_all_vram/status
"""
import asyncio
import gc
import itertools
import json
import logging
import os
import sys
import threading
import time
import types
import warnings
import weakref

import psutil
import torch
from aiohttp import web

import comfy.model_management as mm
import execution
import folder_paths
import server

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
DEFAULTS = {
    "stop_workers_on_unload": True,
    "clear_node_cache_on_unload": True,
    "scan_leftovers": True,
    "offload_leftovers_from": [],
    "auto_free_after_idle_minutes": 10,
    "auto_free_only_if_workers": True,
    "never_stop": [],
}
LEFTOVER_MIN_BYTES = 64 * 1024 ** 2
TAG = "[Free All VRAM]"
GB = 1024 ** 3


def valid(default, value):
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, (int, float)):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, list) and all(isinstance(s, str) for s in value)


def load_config():
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG_PATH, encoding="utf-8-sig") as f:
            data = json.load(f)
    except FileNotFoundError:
        return cfg
    except Exception as e:
        logging.warning(f"{TAG} could not read {CONFIG_PATH} ({e}), using defaults")
        return cfg
    for k, v in data.items():
        if k not in DEFAULTS:
            continue
        if valid(DEFAULTS[k], v):
            cfg[k] = v
        else:
            logging.warning(f"{TAG} config.json: invalid value for \"{k}\": {v!r}, using {DEFAULTS[k]!r}")
    return cfg


def write_default_config():
    if os.path.exists(CONFIG_PATH):
        return
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"_help": "Free All VRAM settings, re-read on every unload (no restart needed). See README.md.", **DEFAULTS}, f, indent=2)
    except OSError as e:
        logging.warning(f"{TAG} could not write {CONFIG_PATH}: {e}")


try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pynvml
    pynvml.nvmlInit()
except Exception as e:
    pynvml = None
    logging.info(f"{TAG} NVIDIA NVML not available ({e}): worker processes can't be detected, everything else still works")


# ---------------------------------------------------------------- processes

def gpu_pids():
    if pynvml is None:
        return set()
    pids = set()
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        for get in (pynvml.nvmlDeviceGetComputeRunningProcesses, pynvml.nvmlDeviceGetGraphicsRunningProcesses):
            try:
                pids.update(p.pid for p in get(h))
            except pynvml.NVMLError:
                pass
    return pids


def args_of(p):
    try:
        return p.cmdline()
    except psutil.Error:
        return []


def stoppable(args, cfg):
    cmd = " ".join(args)
    return bool(cmd) and not any(s and s in cmd for s in cfg["never_stop"])


def gpu_workers(cfg):
    """Child processes (any depth) of this ComfyUI that hold the GPU."""
    children = psutil.Process().children(recursive=True)
    if not children:
        return []
    on_gpu = gpu_pids()
    return [p for p in children if p.pid in on_gpu and stoppable(args_of(p), cfg)]


def custom_node_dirs():
    return [os.path.normcase(os.path.abspath(d)) + os.sep for d in folder_paths.get_folder_paths("custom_nodes")]


def is_custom_node_worker(args, dirs):
    """Runs from a custom_nodes folder (script or bundled runtime) or is a comfy-env worker.

    Deliberately narrow: other programs that merely share ComfyUI's Python must never match."""
    for a in args:
        low = os.path.normcase(a)
        if "comfyui_pvenv_" in low or any(low.startswith(d) for d in dirs):
            return True
    return False


def is_comfyui_server(args, dirs):
    return any(os.path.basename(a).lower() == "main.py" and not is_custom_node_worker([a], dirs) for a in args)


def is_orphan(p):
    try:
        parent = psutil.Process(p.ppid())
        return parent.create_time() > p.create_time()  # pid reused by a newer process
    except psutil.NoSuchProcess:
        return True
    except psutil.Error:
        return False


def orphan_workers(cfg):
    """Custom-node worker processes on the GPU whose parent process has exited."""
    dirs = custom_node_dirs()
    found = []
    for pid in gpu_pids() - {os.getpid()}:
        try:
            p = psutil.Process(pid)
        except psutil.Error:
            continue
        args = args_of(p)
        if not is_custom_node_worker(args, dirs) or is_comfyui_server(args, dirs) or not stoppable(args, cfg):
            continue
        if is_orphan(p):
            found.append(p)
    return found


def describe(p):
    args = args_of(p)
    script = next((os.path.basename(a) for a in reversed(args) if a.endswith(".py")), None)
    owner = next((pkg[0] for a in args if ("/" in a or os.sep in a) and (pkg := package_of(a))), None)
    if owner is None and any("comfyui_pvenv_" in a for a in args):
        owner = "comfy-env"
    try:
        name = p.name()
    except psutil.Error:
        name = "?"
    return f"pid {p.pid} {script or name}" + (f" ({owner})" if owner else "")


def stop_processes(procs):
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=5)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=5)


_gpu_handle = None


def gpu_used_gb():
    """VRAM in use on this ComfyUI's GPU by all processes, or None."""
    global _gpu_handle
    if pynvml is None:
        return None
    try:
        if _gpu_handle is None:
            try:
                props = torch.cuda.get_device_properties(mm.get_torch_device())
                bus_id = f"{props.pci_domain_id:08X}:{props.pci_bus_id:02X}:{props.pci_device_id:02X}.0"
                _gpu_handle = pynvml.nvmlDeviceGetHandleByPciBusId(bus_id.encode())
            except Exception:
                if pynvml.nvmlDeviceGetCount() != 1:
                    return None
                _gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(_gpu_handle).used / GB
    except Exception:
        return None


# ---------------------------------------------------------------- leftover models in this process
#
# Found by walking what each custom-node package reaches from its own module globals (variables,
# containers, its classes' attributes, objects' attributes), never with gc.get_objects() or
# gc.get_referrers(): those take a reference to every object in the process, including a tuple
# another thread is still building, which then fails with "SystemError: ... tupleobject.c ... bad
# argument to internal function" - and a job running at the time can abort the whole process.

LEFTOVER_MAX_DEPTH = 5
LEFTOVER_MAX_OBJECTS = 5000  # per package
_SKIP_TYPES = (str, bytes, bytearray, int, float, complex, bool, type(None), types.ModuleType, types.FunctionType,
               types.BuiltinFunctionType, types.MethodType, weakref.ReferenceType, weakref.ProxyType,
               weakref.CallableProxyType, torch.Tensor)


def package_of(path):
    """(custom node package, file inside it) for a path, or None."""
    if not path:
        return None
    path = os.path.abspath(path)
    low = os.path.normcase(path)
    for d in custom_node_dirs():
        if low.startswith(d):
            pkg, _, rel = path[len(d):].partition(os.sep)
            return pkg, rel or pkg


def on_accelerator(t):
    return t.device.type not in ("cpu", "meta")


def accelerator_bytes(module):
    return sum(t.numel() * t.element_size() for t in itertools.chain(module.parameters(), module.buffers()) if on_accelerator(t))


def children_of(o, pkg_modules):
    """(name, child) pairs of a container or plain object, read without running any of its code."""
    t = type(o)
    if issubclass(t, _SKIP_TYPES):
        return []
    if t in (list, tuple, set, frozenset):
        return [(f"[{i}]", c) for i, c in enumerate(list(o))]
    if issubclass(t, dict):
        return [(f"[{k!r:.40}]", v) for k, v in list(dict.items(o))]
    if issubclass(t, type) and o.__module__ not in pkg_modules:
        return []  # only the package's own classes, not every library class it imports
    try:
        d = object.__getattribute__(o, "__dict__")
    except Exception:
        return []
    if not isinstance(d, (dict, types.MappingProxyType)):
        return []
    return [(f".{k}", v) for k, v in list(d.items())]


def leftover_models():
    """Top-level torch modules on an accelerator that custom nodes keep and ComfyUI doesn't manage.

    Returns [(module, bytes, (package, where))], where is e.g. "_MODEL" or "Loader.cache['sdxl']"."""
    managed = set()
    for lm in mm.current_loaded_models:
        m = getattr(lm.model, "model", None) if lm.model is not None else None
        if isinstance(m, torch.nn.Module):
            managed.update(id(x) for x in m.modules())
    packages = {}
    for name, mod in list(sys.modules.items()):
        owner = package_of(getattr(mod, "__file__", None))
        if owner:
            packages.setdefault(owner[0], []).append((name, mod))
    found = {}
    for pkg, mods in packages.items():
        pkg_modules = {name for name, _ in mods}
        seen = set()
        frontier = [(k, v) for _, mod in mods for k, v in list(vars(mod).items()) if not k.startswith("__")]
        for _ in range(LEFTOVER_MAX_DEPTH):
            nxt = []
            for where, o in frontier:
                if id(o) in seen or len(seen) > LEFTOVER_MAX_OBJECTS:
                    continue
                seen.add(id(o))
                if issubclass(type(o), torch.nn.Module):
                    if id(o) not in managed and id(o) not in found:
                        try:
                            size = accelerator_bytes(o)
                        except Exception:
                            continue
                        if size >= LEFTOVER_MIN_BYTES:
                            found[id(o)] = (o, size, (pkg, where))
                    continue
                try:
                    nxt.extend((where + name, c) for name, c in children_of(o, pkg_modules))
                except Exception:
                    pass
            frontier = nxt
    return list(found.values())


def offload(module):
    """Move a module to system RAM; it moves itself back to its device the next time it runs."""
    device = next(t.device for t in itertools.chain(module.parameters(), module.buffers()) if on_accelerator(t))
    module.to("cpu")

    def move_back(mod, args):
        handle.remove()
        mod.to(device)

    handle = module.register_forward_pre_hook(move_back)


def leftover_label(m, size, owner):
    return f"{type(m).__name__} {size / GB:.2f} GB held by {owner[0]} ({owner[1]})"


# ---------------------------------------------------------------- the unload hook

# The prompt executor owns the node cache; captured so the cache can be cleared right away.
# Setting the "free_memory" flag instead would only take effect after the next prompt.
_executors = weakref.WeakSet()
# Result of the most recent cleanup, for the status endpoint (which must never scan the heap itself).
_last_unload = {}


def prompt_running():
    q = server.PromptServer.instance.prompt_queue
    with q.mutex:
        return len(q.currently_running) > 0


def free_everything(before):
    cfg = load_config()
    procs = gpu_workers(cfg) + orphan_workers(cfg) if cfg["stop_workers_on_unload"] else []
    # Cached node outputs can reference models living in a worker, so the cache must go when its
    # workers do, and before them, so those objects are released while the worker still answers.
    if procs or cfg["clear_node_cache_on_unload"]:
        for ex in list(_executors):
            ex.reset()
    gc.collect()
    stopped = [describe(p) for p in procs]
    stop_processes(procs)

    offloaded, kept = [], []
    if cfg["scan_leftovers"]:
        allow = cfg["offload_leftovers_from"]
        try:
            leftovers = leftover_models()
        except Exception as e:
            leftovers = []
            logging.warning(f"{TAG} leftover model scan failed: {e}")
        for m, size, owner in leftovers:
            label = leftover_label(m, size, owner)
            if "*" in allow or owner[0] in allow:
                try:
                    offload(m)
                    offloaded.append(label)
                except Exception as e:
                    kept.append(f"{label} [offload failed: {e}]")
            else:
                kept.append(label)
        leftovers = m = None
        gc.collect()
    mm.soft_empty_cache()

    after = gpu_used_gb()
    _last_unload.update(time=time.strftime("%Y-%m-%d %H:%M:%S"), gpu_used_gb_before=before, gpu_used_gb_after=after,
                        stopped_workers=stopped, moved_to_ram=offloaded,
                        leftover_models=kept if cfg["scan_leftovers"] else None)
    msg = f"{TAG} GPU memory in use {before:.1f} -> {after:.1f} GB." if before is not None and after is not None else f"{TAG} done."
    if stopped:
        msg += f" Stopped worker process(es): {', '.join(stopped)}."
    if offloaded:
        msg += f" Moved to RAM: {'; '.join(offloaded)}."
    logging.info(msg)
    if kept:
        logging.info(f"{TAG} still on the GPU, kept by custom nodes: {'; '.join(kept)}. "
                     f"Add the package name to \"offload_leftovers_from\" in {CONFIG_PATH} to free it on unload.")


# ---------------------------------------------------------------- idle timer

def idle_watch():
    q = server.PromptServer.instance.prompt_queue
    try:
        orphans = orphan_workers(load_config())
        if orphans:
            logging.info(f"{TAG} stopping orphaned worker process(es) left by an earlier ComfyUI: {', '.join(describe(p) for p in orphans)}")
            stop_processes(orphans)
    except Exception as e:
        logging.warning(f"{TAG} orphan check failed: {e}")
    last_busy = time.monotonic()
    armed = False
    last_prompt = getattr(server.PromptServer.instance, "last_prompt_id", None)
    while True:
        time.sleep(30)
        try:
            # last_prompt_id also catches jobs that started and finished between two checks
            prompt = getattr(server.PromptServer.instance, "last_prompt_id", None)
            if q.get_tasks_remaining() > 0 or prompt != last_prompt:
                last_prompt = prompt
                last_busy = time.monotonic()
                armed = True
                continue
            cfg = load_config()
            minutes = cfg["auto_free_after_idle_minutes"]
            if not armed or minutes <= 0 or time.monotonic() - last_busy < minutes * 60:
                continue
            armed = False
            if cfg["auto_free_only_if_workers"] and not (gpu_workers(cfg) or orphan_workers(cfg)):
                continue
            logging.info(f"{TAG} queue idle for {minutes:g} min, unloading everything")
            q.set_flag("unload_models", True)
        except Exception as e:
            logging.warning(f"{TAG} idle check failed: {e}")


# ---------------------------------------------------------------- status endpoint and node

def status():
    """Runs on a server thread, possibly during a prompt: processes and plain attributes only."""
    cfg = load_config()
    # Names only: model_loaded_memory() calls into comfy-aimdo for dynamic-VRAM models, which a
    # running prompt is using at the same time.
    loaded = []
    for lm in list(mm.current_loaded_models):
        try:
            loaded.append(type(lm.model.model).__name__)
        except Exception:
            pass
    return {
        "gpu_used_gb": gpu_used_gb(),
        "prompt_running": prompt_running(),
        "comfy_loaded_models": loaded,
        "worker_processes": [describe(p) for p in gpu_workers(cfg)],
        "orphaned_workers": [describe(p) for p in orphan_workers(cfg)],
        "last_unload": dict(_last_unload) or None,
        "config": cfg,
    }


async def status_route(request):
    return web.json_response(await asyncio.get_running_loop().run_in_executor(None, status))


class FreeAllVRAM:
    """Run alone (optionally into a Preview Any): everything is freed right after the workflow ends."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "utils"

    @classmethod
    def IS_CHANGED(cls):
        return float("nan")

    def run(self):
        server.PromptServer.instance.prompt_queue.set_flag("unload_models", True)
        cfg = load_config()
        workers = gpu_workers(cfg) + orphan_workers(cfg)
        used = gpu_used_gb()
        lines = ["Unloading everything as soon as this workflow finishes (result in the console)."]
        lines.append("Worker processes to stop: " + (", ".join(describe(p) for p in workers) if workers else "none"))
        if used is not None:
            lines.append(f"GPU memory in use now: {used:.1f} GB")
        return ("\n".join(lines),)


# ---------------------------------------------------------------- install

def install():
    """Hook into ComfyUI. Returns False (and changes nothing) if already installed or unsupported."""
    if getattr(mm.unload_all_models, "_free_all_vram", False):
        logging.warning(f"{TAG} already loaded from another folder, ignoring the copy in {HERE}")
        return False
    q = getattr(getattr(server.PromptServer, "instance", None), "prompt_queue", None)
    needed = ((execution.PromptExecutor, "reset"), (q, "currently_running"), (q, "mutex"),
              (q, "set_flag"), (q, "get_tasks_remaining"), (mm, "current_loaded_models"))
    missing = [name for obj, name in needed if not hasattr(obj, name)]
    if missing:
        logging.warning(f"{TAG} not active: this ComfyUI version has no {', '.join(missing)}")
        return False

    orig_executor_init = execution.PromptExecutor.__init__

    def executor_init(self, *args, **kwargs):
        orig_executor_init(self, *args, **kwargs)
        _executors.add(self)

    execution.PromptExecutor.__init__ = executor_init

    orig_unload_all_models = mm.unload_all_models

    def unload_all_models(*args, **kwargs):
        # Only between prompts: a node that unloads mid-workflow must not lose workers or
        # models that later nodes in the same workflow still need.
        try:
            idle = not prompt_running()
        except Exception:
            logging.exception(f"{TAG} cleanup skipped")
            idle = False
        before = gpu_used_gb() if idle else None
        orig_unload_all_models(*args, **kwargs)
        if idle:
            try:
                free_everything(before)
            except Exception:
                logging.exception(f"{TAG} cleanup failed")

    unload_all_models._free_all_vram = True
    mm.unload_all_models = unload_all_models

    server.PromptServer.instance.routes.get("/free_all_vram/status")(status_route)
    write_default_config()
    gpu_used_gb()  # resolve the NVML handle here, not later on a server thread
    threading.Thread(target=idle_watch, name="free-all-vram-idle", daemon=True).start()
    logging.info(f"{TAG} active: Unload Models also stops GPU worker processes, clears the node cache and reports leftover models")
    return True
