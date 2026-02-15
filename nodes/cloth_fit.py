"""
ClothFit Node - Intersection-free Garment Retargeting (SIGGRAPH 2025)

Wraps the cloth-fit (PolyFEM-based) tool for deforming garments to fit target avatars.
https://github.com/Huangzizhou/cloth-fit

cloth-fit uses a JSON configuration approach with these required inputs:
- avatar_mesh_path: Target avatar mesh (OBJ)
- garment_mesh_path: Source garment mesh (OBJ)
- source_skeleton_path: Source skeleton (OBJ edge mesh)
- target_skeleton_path: Target skeleton (OBJ edge mesh)
- no_fit_spec_path: Collision specification mesh (OBJ)

The tool optimizes the garment mesh to:
1. Follow skeleton bone correspondences (source → target)
2. Preserve garment shape (similarity penalty)
3. Avoid mesh intersections (contact handling)
"""

import os
import subprocess
import tempfile
import time
import json
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

# cloth-fit binary location (PolyFEM_bin symlinked to cloth-fit)
CLOTH_FIT_BINARY = "/usr/local/bin/cloth-fit"

# Default fitting parameters (from cloth-fit examples)
DEFAULT_FIT_PARAMS = {
    "fit_weight": 2.0,
    "similarity_penalty_weight": 1.0,
    "curvature_penalty_weight": 0.01,
    "curve_center_target_weight": 0.0,
    "voxel_size": 0.01,
    "is_skirt": False,
    "contact": {
        "enabled": True,
        "dhat": 0.002
    },
    "solver": {
        "max_threads": 8,
        "contact": {
            "barrier_stiffness": 1e8,
            "CCD": {
                "tolerance": 1e-3,
                "max_iterations": 5000
            }
        },
        "nonlinear": {
            "solver": "Newton",
            "line_search": {
                "method": "backtracking"
            }
        },
        "augmented_lagrangian": {
            "initial_weight": 1e3,
            "max_weight": 1e8
        }
    }
}


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


def create_no_fit_spec(garment_mesh: trimesh.Trimesh, output_path: str) -> bool:
    """
    Create a no-fit specification mesh (defines collision boundaries).

    For most garments, this can be a simplified version of the garment boundary
    or an empty mesh if we want the garment to fit freely.

    Args:
        garment_mesh: The garment mesh
        output_path: Path to write the no-fit OBJ

    Returns:
        True if successful
    """
    try:
        # For now, create an empty mesh (no collision constraints)
        # This allows the garment to fit freely
        with open(output_path, 'w') as f:
            f.write("# No-fit specification (empty = no constraints)\n")
            f.write("v 0 0 0\n")  # Single dummy vertex

        print(f"[ClothFit] Created no-fit spec: {output_path}")
        return True
    except Exception as e:
        print(f"[ClothFit] Error creating no-fit spec: {e}")
        return False


def generate_cloth_fit_config(
    avatar_mesh_path: str,
    garment_mesh_path: str,
    source_skeleton_path: str,
    target_skeleton_path: str,
    no_fit_spec_path: str,
    output_dir: str,
    is_skirt: bool = False,
    **kwargs
) -> str:
    """
    Generate a JSON configuration file for cloth-fit.

    Returns:
        Path to the generated JSON config file
    """
    config = {
        "avatar_mesh_path": avatar_mesh_path,
        "garment_mesh_path": garment_mesh_path,
        "source_skeleton_path": source_skeleton_path,
        "target_skeleton_path": target_skeleton_path,
        "no_fit_spec_path": no_fit_spec_path,
        "is_skirt": is_skirt,
        "output": {
            "directory": output_dir,
            "log": {"level": "info"}
        },
        **DEFAULT_FIT_PARAMS
    }

    # Override with any custom parameters
    for key, value in kwargs.items():
        if key in config:
            config[key] = value

    config_path = os.path.join(output_dir, "setup.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"[ClothFit] Generated config: {config_path}")
    return config_path


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

    Uses the SIGGRAPH 2025 intersection-free garment retargeting algorithm
    (PolyFEM-based optimization).

    Both source and target must share the same skeleton topology (e.g., Mixamo).

    Input:
    - source_garment: Clothing mesh fitted to reference body
    - source_skeleton: Reference body skeleton (from skeleton dict)
    - target_mesh: Target avatar mesh (the body to fit to)
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
                    "tooltip": "Target avatar mesh (the body to fit to)"
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
                "is_skirt": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable skirt-specific handling (for skirts/dresses)"
                }),
                "fit_weight": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 10.0,
                    "tooltip": "How tightly garment follows body (higher = tighter fit)"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("fitted_garment", "output_path")
    FUNCTION = "fit"
    CATEGORY = "UniRig/Clothing"

    def fit(self, source_garment, source_skeleton, target_mesh, target_skeleton,
            output_name="", is_skirt=False, fit_weight=2.0):
        """
        Fit garment from source body to target body.

        Uses cloth-fit (PolyFEM_bin) with JSON configuration.
        """
        start_time = time.time()
        print(f"[ClothFitGarment] Starting garment fitting...")

        # Check if cloth-fit binary exists
        if not os.path.exists(CLOTH_FIT_BINARY):
            raise RuntimeError(
                f"cloth-fit binary not found at {CLOTH_FIT_BINARY}. "
                "Ensure the Docker image was built with cloth-fit installed. "
                "The binary should be at /opt/cloth-fit/build/PolyFEM_bin"
            )

        # Create temp directory for intermediate files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export source garment
            garment_path = os.path.join(tmpdir, "garment.obj")
            if not export_mesh_as_obj(source_garment, garment_path):
                raise RuntimeError("Failed to export source garment")

            # Export source skeleton
            source_skeleton_path = os.path.join(tmpdir, "source_skeleton.obj")
            if not export_skeleton_as_edge_obj(source_skeleton, source_skeleton_path):
                raise RuntimeError("Failed to export source skeleton")

            # Export target mesh (avatar body)
            avatar_path = os.path.join(tmpdir, "avatar.obj")
            if not export_mesh_as_obj(target_mesh, avatar_path):
                raise RuntimeError("Failed to export target mesh")

            # Export target skeleton
            target_skeleton_path = os.path.join(tmpdir, "target_skeleton.obj")
            if not export_skeleton_as_edge_obj(target_skeleton, target_skeleton_path):
                raise RuntimeError("Failed to export target skeleton")

            # Create no-fit spec (collision boundaries)
            no_fit_path = os.path.join(tmpdir, "no_fit.obj")
            if not create_no_fit_spec(source_garment, no_fit_path):
                raise RuntimeError("Failed to create no-fit spec")

            # Output directory
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # Generate JSON config
            config_path = generate_cloth_fit_config(
                avatar_mesh_path=avatar_path,
                garment_mesh_path=garment_path,
                source_skeleton_path=source_skeleton_path,
                target_skeleton_path=target_skeleton_path,
                no_fit_spec_path=no_fit_path,
                output_dir=output_dir,
                is_skirt=is_skirt,
                fit_weight=fit_weight
            )

            # Run cloth-fit (PolyFEM_bin)
            cmd = [
                CLOTH_FIT_BINARY,
                "-j", config_path,
                "--max_threads", "8"
            ]

            print(f"[ClothFitGarment] Running: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout (fitting can be slow)
                    cwd=tmpdir
                )

                # Log output
                if result.stdout:
                    for line in result.stdout.split('\n')[-20:]:  # Last 20 lines
                        print(f"[ClothFit] {line}")

                if result.returncode != 0:
                    print(f"[ClothFitGarment] cloth-fit stderr: {result.stderr}")
                    raise RuntimeError(f"cloth-fit failed with code {result.returncode}")

            except subprocess.TimeoutExpired:
                raise RuntimeError("cloth-fit timed out after 300 seconds")
            except FileNotFoundError:
                raise RuntimeError(f"cloth-fit binary not found: {CLOTH_FIT_BINARY}")

            # Find output mesh (cloth-fit outputs to the output directory)
            output_mesh_path = None
            for f in os.listdir(output_dir):
                if f.endswith('.obj') and 'garment' in f.lower():
                    output_mesh_path = os.path.join(output_dir, f)
                    break

            # Fallback: look for any OBJ in output
            if not output_mesh_path:
                for f in os.listdir(output_dir):
                    if f.endswith('.obj'):
                        output_mesh_path = os.path.join(output_dir, f)
                        break

            if not output_mesh_path or not os.path.exists(output_mesh_path):
                # List what's in output dir for debugging
                print(f"[ClothFitGarment] Output dir contents: {os.listdir(output_dir)}")
                raise RuntimeError(f"cloth-fit did not produce output mesh")

            # Load fitted garment
            fitted_mesh = trimesh.load(output_mesh_path, force='mesh')
            print(f"[ClothFitGarment] Loaded fitted garment: {len(fitted_mesh.vertices)} vertices")

            # Copy to persistent output location
            output_filename = output_name if output_name else f"fitted_garment_{int(time.time())}"
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
