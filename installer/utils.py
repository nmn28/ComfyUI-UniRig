"""
Utility functions for ComfyUI-UniRig Blender installer.
"""

import sys
import platform


class InstallLogger:
    """Structured logging with consistent prefixes."""
    PREFIX = "[UniRig Install]"

    @classmethod
    def info(cls, msg: str):
        print(f"{cls.PREFIX} {msg}")

    @classmethod
    def success(cls, msg: str):
        print(f"{cls.PREFIX} [OK] {msg}")

    @classmethod
    def warning(cls, msg: str):
        print(f"{cls.PREFIX} [WARN] {msg}")

    @classmethod
    def error(cls, msg: str):
        print(f"{cls.PREFIX} [ERROR] {msg}")

    @classmethod
    def header(cls, title: str):
        print("=" * 60)
        print(f"{cls.PREFIX} {title}")
        print("=" * 60)


def get_platform_info() -> dict:
    """Get platform information for Blender download."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        plat = "macos"
        arch = "arm64" if machine == "arm64" else "x64"
    elif system == "linux":
        plat = "linux"
        arch = "x64"
    elif system == "windows":
        plat = "windows"
        arch = "x64"
    else:
        plat = None
        arch = None

    return {
        "system": system,
        "machine": machine,
        "platform": plat,
        "arch": arch,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
