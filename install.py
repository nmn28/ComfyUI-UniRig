#!/usr/bin/env python3
"""
Installation script for ComfyUI-UniRig.

Uses comfy-env for:
- Isolated Python environment with CUDA dependencies
- Blender installation (via [tools] section)
- MSVC runtime on Windows
"""

import sys
from pathlib import Path


def main():
    """Main installation function."""
    print("\n" + "=" * 60)
    print("ComfyUI-UniRig Installation")
    print("=" * 60)

    from comfy_env import install, IsolatedEnvManager, discover_config
    from comfy_env.tools import find_blender

    node_root = Path(__file__).parent.absolute()

    # Run comfy-env install (handles tools + packages)
    try:
        install(config=node_root / "comfy-env.toml", mode="isolated", node_dir=node_root)
    except Exception as e:
        print(f"\n[UniRig] Installation FAILED: {e}")
        print("[UniRig] Report issues at: https://github.com/PozzettiAndrea/ComfyUI-UniRig/issues")
        return 1

    # Verify Blender is available
    blender_exe = find_blender()
    if blender_exe:
        print(f"[UniRig] Blender: {blender_exe}")
    else:
        print("[UniRig] WARNING: Blender not found. FBX export may not work.")

    print("\n" + "=" * 60)
    print("[UniRig] Installation completed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
