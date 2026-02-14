"""
ClothFit Node - Intersection-free Garment Retargeting (SIGGRAPH 2025)

Wraps the cloth-fit C++ tool for deforming garments to fit target avatars.
https://github.com/Huangzizhou/cloth-fit

Takes:
- Source garment mesh (OBJ)
- Source skeleton (OBJ edge mesh)
- Target avatar mesh (OBJ)
- Target skeleton (OBJ edge mesh)

Outputs:
- Fitted garment mesh (OBJ) that conforms to target body with zero mesh intersections
"""

import os
import subprocess
import tempfile
import time
import numpy as np
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

# cloth-fit binary location
CLOTH_FIT_BINARY = "/usr/local/bin/cloth-fit"


def export_skeleton_as_edge_obj(skeleton: dict, output_path: str) -> bool:
    """
    Export skeleton as OBJ edge mesh for cloth-fit.

    Args:
        skeleton: Skeleton dict with 'joints' and 'parents' keys
        output_path: Path to write OBJ file

    Returns:
        True if successful, False otherwise
    """
    try:
        joints = skeleton.get("joints")
        parents = skeleton.get("parents")

        if joints is None or parents is None:
            print(f"[ClothFit] Error: skeleton missing joints or parents")
            return False

        with open(output_path, 'w') as f:
            f.write("# Skeleton edge mesh for cloth-fit\n")
            f.write(f"# {len(joints)} joints\n\n")

            # Vertices (joint positions)
            for j in joints:
                f.write(f"v {j[0]:.6f} {j[1]:.6f} {j[2]:.6f}\n")

            f.write("\n")

            # Edges (bone connections) - OBJ uses 1-indexed
            edge_count = 0
            for i, parent in enumerate(parents):
                if parent is not None and parent >= 0:
                    f.write(f"l {int(parent)+1} {i+1}\n")
                    edge_count += 1

            f.write(f"\n# {edge_count} edges\n")

        print(f"[ClothFit] Exported skeleton: {len(joints)} joints, {edge_count} edges -> {output_path}")
        return True

    except Exception as e:
        print(f"[ClothFit] Error exporting skeleton: {e}")
        return False


def export_mesh_as_obj(mesh: trimesh.Trimesh, output_path: str) -> bool:
    """
    Export trimesh to OBJ format.

    Args:
        mesh: Trimesh object
        output_path: Path to write OBJ file

    Returns:
        True if successful, False otherwise
    """
    try:
        mesh.export(output_path, file_type='obj')
        print(f"[ClothFit] Exported mesh: {len(mesh.vertices)} vertices -> {output_path}")
        return True
    except Exception as e:
        print(f"[ClothFit] Error exporting mesh: {e}")
        return False


class LoadClothingMesh:
    """
    Load a clothing mesh from URL or local path.

    Supports OBJ, GLB, FBX formats.
    Typically used to load garments from S3 clothing library.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clothing_url": ("STRING", {
                    "default": "",
                    "tooltip": "URL or path to clothing mesh (OBJ, GLB, FBX)"
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("clothing_mesh",)
    FUNCTION = "load"
    CATEGORY = "UniRig/Clothing"

    def load(self, clothing_url: str):
        """Load clothing mesh from URL or path."""
        from .mesh_io import download_mesh_from_url, is_url

        print(f"[LoadClothingMesh] Loading: {clothing_url}")

        # Download if URL
        if is_url(clothing_url):
            local_path, error = download_mesh_from_url(clothing_url)
            if error:
                raise RuntimeError(f"Failed to download clothing: {error}")
        else:
            local_path = clothing_url

        if not os.path.exists(local_path):
            raise RuntimeError(f"Clothing file not found: {local_path}")

        # Load mesh
        mesh = trimesh.load(local_path, force='mesh')
        print(f"[LoadClothingMesh] Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        return (mesh,)


class ClothFitGarment:
    """
    Fit a garment to a target avatar using cloth-fit.

    Uses the SIGGRAPH 2025 intersection-free garment retargeting algorithm.
    Both source and target must share the same skeleton topology (e.g., Mixamo).

    Input:
    - source_garment: Clothing mesh fitted to reference body
    - source_skeleton: Reference body skeleton (from skeleton dict)
    - target_mesh: Target avatar mesh (rigged)
    - target_skeleton: Target avatar skeleton (from UniRigAutoRig output)

    Output:
    - fitted_garment: Deformed clothing that fits target body with no intersections
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_garment": ("TRIMESH", {
                    "tooltip": "Source garment mesh (fitted to reference body)"
                }),
                "source_skeleton": ("SKELETON", {
                    "tooltip": "Source/reference skeleton (from reference body)"
                }),
                "target_mesh": ("TRIMESH", {
                    "tooltip": "Target avatar mesh"
                }),
                "target_skeleton": ("SKELETON", {
                    "tooltip": "Target avatar skeleton (from UniRigAutoRig)"
                }),
            },
            "optional": {
                "output_name": ("STRING", {
                    "default": "",
                    "tooltip": "Custom name for output file"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("fitted_garment", "output_path")
    FUNCTION = "fit"
    CATEGORY = "UniRig/Clothing"

    def fit(self, source_garment, source_skeleton, target_mesh, target_skeleton, output_name=""):
        """
        Fit garment from source body to target body.

        Uses cloth-fit C++ binary for intersection-free deformation.
        """
        start_time = time.time()
        print(f"[ClothFitGarment] Starting garment fitting...")

        # Check if cloth-fit binary exists
        if not os.path.exists(CLOTH_FIT_BINARY):
            raise RuntimeError(
                f"cloth-fit binary not found at {CLOTH_FIT_BINARY}. "
                "Ensure the Docker image was built with cloth-fit installed."
            )

        # Create temp directory for intermediate files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export source garment
            source_garment_path = os.path.join(tmpdir, "source_garment.obj")
            if not export_mesh_as_obj(source_garment, source_garment_path):
                raise RuntimeError("Failed to export source garment")

            # Export source skeleton
            source_skeleton_path = os.path.join(tmpdir, "source_skeleton.obj")
            if not export_skeleton_as_edge_obj(source_skeleton, source_skeleton_path):
                raise RuntimeError("Failed to export source skeleton")

            # Export target mesh
            target_mesh_path = os.path.join(tmpdir, "target_mesh.obj")
            if not export_mesh_as_obj(target_mesh, target_mesh_path):
                raise RuntimeError("Failed to export target mesh")

            # Export target skeleton
            target_skeleton_path = os.path.join(tmpdir, "target_skeleton.obj")
            if not export_skeleton_as_edge_obj(target_skeleton, target_skeleton_path):
                raise RuntimeError("Failed to export target skeleton")

            # Output path
            output_filename = output_name if output_name else f"fitted_garment_{int(time.time())}"
            output_path = os.path.join(tmpdir, f"{output_filename}.obj")

            # Run cloth-fit
            # cloth-fit CLI: cloth_fit <source_mesh> <source_skeleton> <target_mesh> <target_skeleton> <output>
            cmd = [
                CLOTH_FIT_BINARY,
                source_garment_path,
                source_skeleton_path,
                target_mesh_path,
                target_skeleton_path,
                output_path
            ]

            print(f"[ClothFitGarment] Running: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minute timeout
                )

                if result.returncode != 0:
                    print(f"[ClothFitGarment] cloth-fit stderr: {result.stderr}")
                    raise RuntimeError(f"cloth-fit failed: {result.stderr}")

                print(f"[ClothFitGarment] cloth-fit stdout: {result.stdout}")

            except subprocess.TimeoutExpired:
                raise RuntimeError("cloth-fit timed out after 120 seconds")
            except FileNotFoundError:
                raise RuntimeError(f"cloth-fit binary not found: {CLOTH_FIT_BINARY}")

            # Load fitted garment
            if not os.path.exists(output_path):
                raise RuntimeError(f"cloth-fit did not produce output: {output_path}")

            fitted_mesh = trimesh.load(output_path, force='mesh')
            print(f"[ClothFitGarment] Loaded fitted garment: {len(fitted_mesh.vertices)} vertices")

            # Copy to persistent output location
            persistent_path = os.path.join(COMFYUI_OUTPUT_FOLDER, f"{output_filename}.obj")
            fitted_mesh.export(persistent_path, file_type='obj')
            print(f"[ClothFitGarment] Saved to: {persistent_path}")

        elapsed = time.time() - start_time
        print(f"[ClothFitGarment] Garment fitting complete in {elapsed:.2f}s")

        return (fitted_mesh, persistent_path)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LoadClothingMesh": LoadClothingMesh,
    "ClothFitGarment": ClothFitGarment,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadClothingMesh": "Load Clothing Mesh",
    "ClothFitGarment": "Cloth-Fit Garment",
}
