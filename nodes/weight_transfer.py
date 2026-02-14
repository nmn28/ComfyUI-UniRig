"""
Weight Transfer Node - Robust Weight Transfer (SIGGRAPH Asia 2023)

Wraps the Robust Weight Transfer Blender addon for transferring skin weights
from a rigged body to unrigged clothing.
https://github.com/sentfromspacevr/robust-weight-transfer

Takes:
- Rigged avatar mesh (GLB/FBX with skeleton and weights)
- Fitted garment mesh (OBJ from ClothFitGarment)

Outputs:
- Rigged garment mesh (GLB with transferred skin weights)
"""

import os
import subprocess
import tempfile
import time
import trimesh
from pathlib import Path
from typing import Tuple, Optional

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
    COMFYUI_TEMP_FOLDER = folder_paths.get_temp_directory()
except:
    COMFYUI_OUTPUT_FOLDER = "/tmp/outputs"
    COMFYUI_TEMP_FOLDER = "/tmp"

# Blender binary location
BLENDER_BINARY = "/usr/local/bin/blender"

# Blender script for weight transfer
WEIGHT_TRANSFER_SCRIPT = """
import bpy
import sys
import os

# Get arguments after --
argv = sys.argv
args_start = argv.index("--") + 1 if "--" in argv else len(argv)
args = argv[args_start:]

if len(args) < 3:
    print("Usage: blender -b --python script.py -- <rigged_body> <garment> <output>")
    sys.exit(1)

rigged_body_path = args[0]
garment_path = args[1]
output_path = args[2]

print(f"[WeightTransfer] Rigged body: {rigged_body_path}")
print(f"[WeightTransfer] Garment: {garment_path}")
print(f"[WeightTransfer] Output: {output_path}")

# Clear default scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import rigged body (GLB/FBX)
ext = os.path.splitext(rigged_body_path)[1].lower()
if ext == '.glb' or ext == '.gltf':
    bpy.ops.import_scene.gltf(filepath=rigged_body_path)
elif ext == '.fbx':
    bpy.ops.import_scene.fbx(filepath=rigged_body_path)
else:
    print(f"[WeightTransfer] Unsupported format for rigged body: {ext}")
    sys.exit(1)

# Find armature and body mesh
armature = None
body_mesh = None
for obj in bpy.context.scene.objects:
    if obj.type == 'ARMATURE':
        armature = obj
    elif obj.type == 'MESH' and obj.parent and obj.parent.type == 'ARMATURE':
        body_mesh = obj

if not armature:
    print("[WeightTransfer] Error: No armature found in rigged body")
    sys.exit(1)

if not body_mesh:
    # Try to find any mesh with vertex groups
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and len(obj.vertex_groups) > 0:
            body_mesh = obj
            break

if not body_mesh:
    print("[WeightTransfer] Error: No skinned mesh found in rigged body")
    sys.exit(1)

print(f"[WeightTransfer] Found armature: {armature.name}")
print(f"[WeightTransfer] Found body mesh: {body_mesh.name} ({len(body_mesh.vertex_groups)} vertex groups)")

# Import garment mesh (OBJ)
bpy.ops.wm.obj_import(filepath=garment_path)

# Find imported garment
garment_mesh = None
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH' and obj != body_mesh:
        garment_mesh = obj
        break

if not garment_mesh:
    print("[WeightTransfer] Error: Failed to import garment mesh")
    sys.exit(1)

print(f"[WeightTransfer] Imported garment: {garment_mesh.name}")

# Enable Robust Weight Transfer addon
try:
    bpy.ops.preferences.addon_enable(module='robust_weight_transfer')
    print("[WeightTransfer] Robust Weight Transfer addon enabled")
except Exception as e:
    print(f"[WeightTransfer] Warning: Could not enable addon: {e}")
    print("[WeightTransfer] Falling back to standard Data Transfer")

# Select garment as active, body as source
bpy.ops.object.select_all(action='DESELECT')
body_mesh.select_set(True)
garment_mesh.select_set(True)
bpy.context.view_layer.objects.active = garment_mesh

# Try Robust Weight Transfer operator first
try:
    # The addon provides: bpy.ops.object.robust_weight_transfer()
    bpy.ops.object.robust_weight_transfer()
    print("[WeightTransfer] Robust Weight Transfer completed")
except AttributeError:
    # Fallback to standard weight transfer
    print("[WeightTransfer] Using standard Data Transfer modifier")

    # Add Data Transfer modifier
    mod = garment_mesh.modifiers.new(name="WeightTransfer", type='DATA_TRANSFER')
    mod.object = body_mesh
    mod.use_vert_data = True
    mod.data_types_verts = {'VGROUP_WEIGHTS'}
    mod.vert_mapping = 'NEAREST'

    # Apply modifier
    bpy.ops.object.modifier_apply(modifier=mod.name)
    print("[WeightTransfer] Data Transfer applied")

# Parent garment to armature with armature modifier
garment_mesh.parent = armature
mod = garment_mesh.modifiers.new(name="Armature", type='ARMATURE')
mod.object = armature

print(f"[WeightTransfer] Garment now has {len(garment_mesh.vertex_groups)} vertex groups")

# Export rigged garment as GLB
bpy.ops.object.select_all(action='DESELECT')
armature.select_set(True)
garment_mesh.select_set(True)
bpy.context.view_layer.objects.active = armature

bpy.ops.export_scene.gltf(
    filepath=output_path,
    use_selection=True,
    export_format='GLB',
    export_animations=False,
    export_skins=True,
)

print(f"[WeightTransfer] Exported rigged garment to: {output_path}")
print("[WeightTransfer] Done!")
"""


class TransferSkinWeights:
    """
    Transfer skin weights from rigged avatar to fitted garment.

    Uses Robust Weight Transfer (SIGGRAPH Asia 2023) Blender addon
    for high-quality weight transfer that handles loose garments.

    Input:
    - rigged_avatar_path: Path to rigged avatar (GLB/FBX with skeleton)
    - fitted_garment: Fitted garment mesh (from ClothFitGarment)

    Output:
    - rigged_garment_path: Path to rigged garment GLB
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rigged_avatar_path": ("STRING", {
                    "tooltip": "Path to rigged avatar GLB/FBX"
                }),
                "fitted_garment": ("TRIMESH", {
                    "tooltip": "Fitted garment mesh (from ClothFitGarment)"
                }),
            },
            "optional": {
                "output_name": ("STRING", {
                    "default": "",
                    "tooltip": "Custom name for output file"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("rigged_garment_path",)
    FUNCTION = "transfer"
    CATEGORY = "UniRig/Clothing"

    def transfer(self, rigged_avatar_path: str, fitted_garment, output_name: str = ""):
        """
        Transfer weights from rigged avatar to fitted garment.

        Uses Blender headless with Robust Weight Transfer addon.
        """
        start_time = time.time()
        print(f"[TransferSkinWeights] Starting weight transfer...")

        # Check if Blender binary exists
        if not os.path.exists(BLENDER_BINARY):
            raise RuntimeError(
                f"Blender binary not found at {BLENDER_BINARY}. "
                "Ensure the Docker image was built with Blender installed."
            )

        # Check if rigged avatar exists
        if not os.path.exists(rigged_avatar_path):
            raise RuntimeError(f"Rigged avatar not found: {rigged_avatar_path}")

        # Create temp directory for intermediate files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export fitted garment to OBJ
            garment_path = os.path.join(tmpdir, "fitted_garment.obj")
            fitted_garment.export(garment_path, file_type='obj')
            print(f"[TransferSkinWeights] Exported garment: {garment_path}")

            # Write Blender script to temp file
            script_path = os.path.join(tmpdir, "weight_transfer.py")
            with open(script_path, 'w') as f:
                f.write(WEIGHT_TRANSFER_SCRIPT)

            # Output path
            output_filename = output_name if output_name else f"rigged_garment_{int(time.time())}"
            output_path = os.path.join(COMFYUI_OUTPUT_FOLDER, f"{output_filename}.glb")

            # Run Blender headless
            cmd = [
                BLENDER_BINARY,
                "-b",  # Background mode (no GUI)
                "--python", script_path,
                "--",  # Arguments after this go to script
                rigged_avatar_path,
                garment_path,
                output_path
            ]

            print(f"[TransferSkinWeights] Running Blender: {' '.join(cmd[:5])}...")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=180  # 3 minute timeout
                )

                # Print Blender output
                for line in result.stdout.split('\n'):
                    if '[WeightTransfer]' in line:
                        print(line)

                if result.returncode != 0:
                    print(f"[TransferSkinWeights] Blender stderr: {result.stderr}")
                    raise RuntimeError(f"Blender weight transfer failed")

            except subprocess.TimeoutExpired:
                raise RuntimeError("Blender timed out after 180 seconds")

            # Verify output exists
            if not os.path.exists(output_path):
                raise RuntimeError(f"Blender did not produce output: {output_path}")

        elapsed = time.time() - start_time
        print(f"[TransferSkinWeights] Weight transfer complete in {elapsed:.2f}s")
        print(f"[TransferSkinWeights] Output: {output_path}")

        return (output_path,)


class CombineAvatarClothing:
    """
    Combine rigged avatar and rigged clothing into a single GLB.

    Both meshes share the same Mixamo skeleton, so they can be
    combined and animated together.

    Input:
    - avatar_path: Path to rigged avatar GLB
    - clothing_paths: Comma-separated list of rigged clothing GLB paths

    Output:
    - combined_path: Path to combined GLB with avatar + all clothing
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "avatar_path": ("STRING", {
                    "tooltip": "Path to rigged avatar GLB"
                }),
                "clothing_path": ("STRING", {
                    "tooltip": "Path to rigged clothing GLB"
                }),
            },
            "optional": {
                "output_name": ("STRING", {
                    "default": "",
                    "tooltip": "Custom name for output file"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_path",)
    FUNCTION = "combine"
    CATEGORY = "UniRig/Clothing"

    def combine(self, avatar_path: str, clothing_path: str, output_name: str = ""):
        """
        Combine avatar and clothing GLBs.

        Uses Blender to merge meshes under same armature.
        """
        start_time = time.time()
        print(f"[CombineAvatarClothing] Combining avatar + clothing...")

        combine_script = '''
import bpy
import sys
import os

argv = sys.argv
args_start = argv.index("--") + 1 if "--" in argv else len(argv)
args = argv[args_start:]

avatar_path = args[0]
clothing_path = args[1]
output_path = args[2]

# Clear scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import avatar
bpy.ops.import_scene.gltf(filepath=avatar_path)
print(f"[Combine] Imported avatar")

# Find armature
armature = None
for obj in bpy.context.scene.objects:
    if obj.type == 'ARMATURE':
        armature = obj
        break

# Import clothing
bpy.ops.import_scene.gltf(filepath=clothing_path)
print(f"[Combine] Imported clothing")

# Find clothing mesh (most recently added mesh)
clothing_mesh = None
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH' and obj.name.startswith('rigged_garment'):
        clothing_mesh = obj
        break

# If no rigged_garment found, find any mesh without parent that's not original
if not clothing_mesh:
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.parent is None:
            clothing_mesh = obj
            break

if clothing_mesh and armature:
    # Re-parent clothing to main armature
    # First, find and remove any duplicate armatures
    for obj in list(bpy.context.scene.objects):
        if obj.type == 'ARMATURE' and obj != armature:
            # Transfer children to main armature
            for child in obj.children:
                child.parent = armature
            bpy.data.objects.remove(obj)

    print(f"[Combine] Combined under armature: {armature.name}")

# Export combined
bpy.ops.object.select_all(action='SELECT')
bpy.ops.export_scene.gltf(
    filepath=output_path,
    use_selection=True,
    export_format='GLB',
    export_animations=False,
    export_skins=True,
)

print(f"[Combine] Exported to: {output_path}")
'''

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "combine.py")
            with open(script_path, 'w') as f:
                f.write(combine_script)

            output_filename = output_name if output_name else f"avatar_with_clothing_{int(time.time())}"
            output_path = os.path.join(COMFYUI_OUTPUT_FOLDER, f"{output_filename}.glb")

            cmd = [
                BLENDER_BINARY,
                "-b",
                "--python", script_path,
                "--",
                avatar_path,
                clothing_path,
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                print(f"[CombineAvatarClothing] Error: {result.stderr}")
                raise RuntimeError("Failed to combine avatar and clothing")

        elapsed = time.time() - start_time
        print(f"[CombineAvatarClothing] Combined in {elapsed:.2f}s -> {output_path}")

        return (output_path,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "TransferSkinWeights": TransferSkinWeights,
    "CombineAvatarClothing": CombineAvatarClothing,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TransferSkinWeights": "Transfer Skin Weights",
    "CombineAvatarClothing": "Combine Avatar + Clothing",
}
