COMFYUI_PORT = 8188
FLUXGYM_PORT = 7860
PORTS = [COMFYUI_PORT, FLUXGYM_PORT]

def compose():
    from .compose import compose
    return compose
