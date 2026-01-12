"""
ComfyUI-UniRig Blender Installer Package

Handles Blender installation for mesh preprocessing (FBX export, etc.)

CUDA dependencies (torch-scatter, torch-cluster, spconv) are now handled by
comfyui-envmanager via comfyui_env.toml. Run `comfy-env install` to install them.
"""

from .blender import install_blender, find_blender_executable
from .utils import InstallLogger, get_platform_info

__all__ = [
    "install_blender",
    "find_blender_executable",
    "InstallLogger",
    "get_platform_info",
]
