from . import free_all_vram

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if free_all_vram.install():
    NODE_CLASS_MAPPINGS["FreeAllVRAM"] = free_all_vram.FreeAllVRAM
    NODE_DISPLAY_NAME_MAPPINGS["FreeAllVRAM"] = "Free All VRAM (incl. worker processes)"
