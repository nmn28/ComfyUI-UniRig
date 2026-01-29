"""UniRig Nodes - Auto-discovers and wraps isolated environments."""

from pathlib import Path
import importlib

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

nodes_dir = Path(__file__).parent

# Main nodes (no comfy-env.toml, not isolated)
from .mesh_io import UniRigLoadMesh, UniRigSaveMesh
NODE_CLASS_MAPPINGS.update({
    "UniRigLoadMesh": UniRigLoadMesh,
    "UniRigSaveMesh": UniRigSaveMesh,
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "UniRigLoadMesh": "UniRig: Load Mesh",
    "UniRigSaveMesh": "UniRig: Save Mesh",
})
print(f"[UniRig] Main nodes loaded (2 nodes)")

# Auto-discover and wrap isolated nodes
# Any subdirectory with comfy-env.toml gets wrapped
try:
    from comfy_env import wrap_isolated_nodes

    for subdir in nodes_dir.iterdir():
        if not subdir.is_dir():
            continue
        config_file = subdir / "comfy-env.toml"
        if not config_file.exists():
            continue

        # Found isolated environment - try to import and wrap
        module_name = subdir.name
        try:
            # Import the submodule
            module = importlib.import_module(f".{module_name}", package=__name__)

            sub_mappings = getattr(module, "NODE_CLASS_MAPPINGS", {})
            sub_display = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})

            if sub_mappings:
                wrapped = wrap_isolated_nodes(sub_mappings, subdir)
                NODE_CLASS_MAPPINGS.update(wrapped)
                NODE_DISPLAY_NAME_MAPPINGS.update(sub_display)
                print(f"[UniRig] {module_name} loaded ({len(sub_mappings)} nodes, isolated)")
        except ImportError as e:
            print(f"[UniRig] {module_name} not available: {e}")

except ImportError:
    print("[UniRig] comfy-env not installed, isolated nodes disabled")

print(f"[UniRig] Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
