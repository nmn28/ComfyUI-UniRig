"""
GarmentPlacement Node - Position AI-generated clothing on reference mannequin

When users generate clothing via AI (Tripo/Meshy), the mesh comes back
floating in arbitrary space with no body context. This node:

1. Takes raw AI mesh + clothing category (shoes/top/bottom/hat/etc)
2. Loads category-specific placement transform from S3
3. Positions the mesh correctly on a reference mannequin
4. Outputs positioned mesh + reference skeleton (ready for cloth-fit)

The reference mannequin is a standard T-pose humanoid with Mixamo skeleton.
All template garments in the library are positioned relative to this body.
"""

import os
import json
import tempfile
import time
import numpy as np
import trimesh
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
    COMFYUI_TEMP_FOLDER = folder_paths.get_temp_directory()
except:
    COMFYUI_OUTPUT_FOLDER = "/tmp/outputs"
    COMFYUI_TEMP_FOLDER = "/tmp"

# S3 paths for placement templates
S3_TEMPLATES_BUCKET = "endure-media"
S3_TEMPLATES_PREFIX = "clothing/templates/"

# Clothing categories with descriptions
CLOTHING_CATEGORIES = [
    "shoes",      # Footwear - positioned at feet
    "top",        # Shirts, jackets - positioned at torso
    "bottom",     # Pants, skirts - positioned at waist/legs
    "hat",        # Headwear - positioned at head
    "gloves",     # Handwear - positioned at hands
    "fullbody",   # Full outfits - scaled to full body
    "accessory",  # Bags, jewelry - positioned at center
]

# Default placement transforms (used if S3 fetch fails)
# These position the bounding box center of the garment at the right body region
# Coordinates assume Mixamo T-pose: Y-up, facing -Z, arms along X
DEFAULT_PLACEMENTS = {
    "shoes": {
        "translate": [0, 0.05, 0],      # Feet level (just above ground)
        "scale": [1.0, 1.0, 1.0],
        "rotation": [0, 0, 0],
        "body_region": "feet",
        "reference_height": 0.1,         # Expected height of shoe
    },
    "top": {
        "translate": [0, 1.2, 0],        # Chest level
        "scale": [1.0, 1.0, 1.0],
        "rotation": [0, 0, 0],
        "body_region": "torso",
        "reference_height": 0.5,
    },
    "bottom": {
        "translate": [0, 0.8, 0],        # Waist/hip level
        "scale": [1.0, 1.0, 1.0],
        "rotation": [0, 0, 0],
        "body_region": "legs",
        "reference_height": 0.7,
    },
    "hat": {
        "translate": [0, 1.7, 0],        # Head level
        "scale": [1.0, 1.0, 1.0],
        "rotation": [0, 0, 0],
        "body_region": "head",
        "reference_height": 0.25,
    },
    "gloves": {
        "translate": [0.6, 1.0, 0],      # Hand level (will mirror for left)
        "scale": [1.0, 1.0, 1.0],
        "rotation": [0, 0, 0],
        "body_region": "hands",
        "reference_height": 0.15,
    },
    "fullbody": {
        "translate": [0, 0.9, 0],        # Center of body
        "scale": [1.0, 1.0, 1.0],
        "rotation": [0, 0, 0],
        "body_region": "full",
        "reference_height": 1.7,
    },
    "accessory": {
        "translate": [0, 1.0, 0],        # Center
        "scale": [1.0, 1.0, 1.0],
        "rotation": [0, 0, 0],
        "body_region": "center",
        "reference_height": 0.3,
    },
}

# Reference mannequin skeleton (Mixamo T-pose joint positions)
# This is used when we can't load from S3
REFERENCE_SKELETON = {
    "joints": np.array([
        [0.0, 1.04, 0.0],       # 0: Hips
        [0.0, 1.2, 0.0],        # 1: Spine
        [0.0, 1.35, 0.0],       # 2: Spine1
        [0.0, 1.5, 0.0],        # 3: Spine2
        [0.0, 1.6, 0.0],        # 4: Neck
        [0.0, 1.75, 0.0],       # 5: Head
        [0.15, 1.5, 0.0],       # 6: LeftShoulder
        [0.35, 1.5, 0.0],       # 7: LeftArm
        [0.6, 1.5, 0.0],        # 8: LeftForeArm
        [0.85, 1.5, 0.0],       # 9: LeftHand
        [-0.15, 1.5, 0.0],      # 10: RightShoulder
        [-0.35, 1.5, 0.0],      # 11: RightArm
        [-0.6, 1.5, 0.0],       # 12: RightForeArm
        [-0.85, 1.5, 0.0],      # 13: RightHand
        [0.1, 1.0, 0.0],        # 14: LeftUpLeg
        [0.1, 0.55, 0.0],       # 15: LeftLeg
        [0.1, 0.08, 0.0],       # 16: LeftFoot
        [0.1, 0.0, 0.1],        # 17: LeftToeBase
        [-0.1, 1.0, 0.0],       # 18: RightUpLeg
        [-0.1, 0.55, 0.0],      # 19: RightLeg
        [-0.1, 0.08, 0.0],      # 20: RightFoot
        [-0.1, 0.0, 0.1],       # 21: RightToeBase
    ], dtype=np.float32),
    "parents": [
        None,  # 0: Hips (root)
        0,     # 1: Spine -> Hips
        1,     # 2: Spine1 -> Spine
        2,     # 3: Spine2 -> Spine1
        3,     # 4: Neck -> Spine2
        4,     # 5: Head -> Neck
        3,     # 6: LeftShoulder -> Spine2
        6,     # 7: LeftArm -> LeftShoulder
        7,     # 8: LeftForeArm -> LeftArm
        8,     # 9: LeftHand -> LeftForeArm
        3,     # 10: RightShoulder -> Spine2
        10,    # 11: RightArm -> RightShoulder
        11,    # 12: RightForeArm -> RightArm
        12,    # 13: RightHand -> RightForeArm
        0,     # 14: LeftUpLeg -> Hips
        14,    # 15: LeftLeg -> LeftUpLeg
        15,    # 16: LeftFoot -> LeftLeg
        16,    # 17: LeftToeBase -> LeftFoot
        0,     # 18: RightUpLeg -> Hips
        18,    # 19: RightLeg -> RightUpLeg
        19,    # 20: RightFoot -> RightLeg
        20,    # 21: RightToeBase -> RightFoot
    ],
    "names": [
        "mixamorig:Hips",
        "mixamorig:Spine",
        "mixamorig:Spine1",
        "mixamorig:Spine2",
        "mixamorig:Neck",
        "mixamorig:Head",
        "mixamorig:LeftShoulder",
        "mixamorig:LeftArm",
        "mixamorig:LeftForeArm",
        "mixamorig:LeftHand",
        "mixamorig:RightShoulder",
        "mixamorig:RightArm",
        "mixamorig:RightForeArm",
        "mixamorig:RightHand",
        "mixamorig:LeftUpLeg",
        "mixamorig:LeftLeg",
        "mixamorig:LeftFoot",
        "mixamorig:LeftToeBase",
        "mixamorig:RightUpLeg",
        "mixamorig:RightLeg",
        "mixamorig:RightFoot",
        "mixamorig:RightToeBase",
    ]
}


def load_placement_transform(category: str) -> Dict[str, Any]:
    """
    Load placement transform for a clothing category.

    Tries S3 first, falls back to defaults.

    Args:
        category: Clothing category (shoes, top, bottom, etc.)

    Returns:
        Dict with translate, scale, rotation, reference_height
    """
    # For now, use defaults (S3 loading can be added later)
    if category in DEFAULT_PLACEMENTS:
        return DEFAULT_PLACEMENTS[category]
    else:
        print(f"[GarmentPlacement] Unknown category '{category}', using accessory defaults")
        return DEFAULT_PLACEMENTS["accessory"]


def normalize_mesh_to_unit(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, float, np.ndarray]:
    """
    Normalize mesh to fit in unit cube centered at origin.

    Returns:
        (normalized_mesh, original_scale, original_center)
    """
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    extents = bounds[1] - bounds[0]
    scale = np.max(extents)

    # Center and scale
    normalized = mesh.copy()
    normalized.vertices -= center
    if scale > 0:
        normalized.vertices /= scale

    return normalized, scale, center


def apply_placement_transform(
    mesh: trimesh.Trimesh,
    transform: Dict[str, Any],
    auto_scale: bool = True
) -> trimesh.Trimesh:
    """
    Apply placement transform to position mesh on reference body.

    Args:
        mesh: Input mesh (raw AI-generated)
        transform: Placement transform dict
        auto_scale: If True, scale mesh to match reference_height

    Returns:
        Transformed mesh positioned on reference body
    """
    # First normalize to unit cube
    normalized, original_scale, original_center = normalize_mesh_to_unit(mesh)

    # Get transform parameters
    translate = np.array(transform.get("translate", [0, 0, 0]))
    scale = np.array(transform.get("scale", [1, 1, 1]))
    rotation = np.array(transform.get("rotation", [0, 0, 0]))  # Euler angles in degrees
    reference_height = transform.get("reference_height", 1.0)

    # Auto-scale to match reference height
    if auto_scale and reference_height > 0:
        current_height = normalized.bounds[1][1] - normalized.bounds[0][1]
        if current_height > 0:
            height_scale = reference_height / current_height
            scale = scale * height_scale

    # Apply scale
    result = normalized.copy()
    result.vertices *= scale

    # Apply rotation (if any)
    if np.any(rotation != 0):
        rotation_rad = np.radians(rotation)
        # Create rotation matrix (XYZ order)
        Rx = trimesh.transformations.rotation_matrix(rotation_rad[0], [1, 0, 0])
        Ry = trimesh.transformations.rotation_matrix(rotation_rad[1], [0, 1, 0])
        Rz = trimesh.transformations.rotation_matrix(rotation_rad[2], [0, 0, 1])
        R = Rz @ Ry @ Rx
        result.apply_transform(R)

    # Apply translation (position on body)
    result.vertices += translate

    return result


class GarmentPlacement:
    """
    Position AI-generated clothing on reference mannequin.

    Takes raw mesh from AI generation (Tripo/Meshy) + clothing category,
    positions it correctly on a reference body for cloth-fit.

    Categories:
    - shoes: Footwear, positioned at feet
    - top: Shirts/jackets, positioned at torso
    - bottom: Pants/skirts, positioned at waist
    - hat: Headwear, positioned at head
    - gloves: Handwear, positioned at hands
    - fullbody: Full outfits, scaled to body
    - accessory: Bags/jewelry, centered
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_mesh": ("TRIMESH", {
                    "tooltip": "Raw AI-generated clothing mesh"
                }),
                "category": (CLOTHING_CATEGORIES, {
                    "default": "top",
                    "tooltip": "Clothing category (determines placement on body)"
                }),
            },
            "optional": {
                "auto_scale": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically scale to appropriate size for category"
                }),
                "custom_translate_y": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Manual Y offset adjustment"
                }),
                "custom_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Manual scale multiplier"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "SKELETON")
    RETURN_NAMES = ("positioned_mesh", "reference_skeleton")
    FUNCTION = "place"
    CATEGORY = "UniRig/Clothing"

    def place(self, raw_mesh, category: str, auto_scale: bool = True,
              custom_translate_y: float = 0.0, custom_scale: float = 1.0):
        """
        Position garment on reference mannequin.
        """
        start_time = time.time()
        print(f"[GarmentPlacement] Positioning {category} garment...")
        print(f"[GarmentPlacement] Input mesh: {len(raw_mesh.vertices)} vertices")

        # Load placement transform for category
        transform = load_placement_transform(category)

        # Apply custom adjustments
        if custom_translate_y != 0.0:
            transform = transform.copy()
            translate = list(transform.get("translate", [0, 0, 0]))
            translate[1] += custom_translate_y
            transform["translate"] = translate

        if custom_scale != 1.0:
            transform = transform.copy()
            scale = list(transform.get("scale", [1, 1, 1]))
            scale = [s * custom_scale for s in scale]
            transform["scale"] = scale

        # Apply placement transform
        positioned_mesh = apply_placement_transform(raw_mesh, transform, auto_scale)

        print(f"[GarmentPlacement] Positioned at: {transform.get('translate')}")
        print(f"[GarmentPlacement] Output bounds: {positioned_mesh.bounds}")

        # Return reference skeleton
        reference_skeleton = {
            "joints": REFERENCE_SKELETON["joints"].copy(),
            "parents": REFERENCE_SKELETON["parents"].copy(),
            "names": REFERENCE_SKELETON["names"].copy(),
        }

        elapsed = time.time() - start_time
        print(f"[GarmentPlacement] Placement complete in {elapsed:.3f}s")

        return (positioned_mesh, reference_skeleton)


class LoadReferenceMannequin:
    """
    Load the reference mannequin mesh and skeleton.

    The reference mannequin is a standard T-pose humanoid that all
    template garments in the library are positioned relative to.

    Used as source_mesh and source_skeleton for cloth-fit when
    fitting AI-generated clothing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "mannequin_url": ("STRING", {
                    "default": "",
                    "tooltip": "URL to custom reference mannequin (leave empty for default)"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "SKELETON")
    RETURN_NAMES = ("mannequin_mesh", "mannequin_skeleton")
    FUNCTION = "load"
    CATEGORY = "UniRig/Clothing"

    def load(self, mannequin_url: str = ""):
        """
        Load reference mannequin mesh and skeleton.
        """
        print(f"[LoadReferenceMannequin] Loading reference mannequin...")

        if mannequin_url:
            # Load from URL
            from .mesh_io import download_mesh_from_url
            local_path, error = download_mesh_from_url(mannequin_url)
            if error:
                raise RuntimeError(f"Failed to download mannequin: {error}")
            mannequin_mesh = trimesh.load(local_path, force='mesh')
        else:
            # Create a simple capsule mannequin as placeholder
            # In production, load from S3: clothing/templates/reference_mannequin.obj
            print(f"[LoadReferenceMannequin] Using placeholder mannequin (load from S3 in production)")

            # Create a simple humanoid shape from primitives
            # This is just for testing - real mannequin should be loaded from S3
            body = trimesh.creation.capsule(height=1.4, radius=0.15)
            body.vertices[:, 1] += 0.9  # Move up

            head = trimesh.creation.icosphere(radius=0.12)
            head.vertices[:, 1] += 1.7

            mannequin_mesh = trimesh.util.concatenate([body, head])

        print(f"[LoadReferenceMannequin] Mannequin: {len(mannequin_mesh.vertices)} vertices")

        # Return reference skeleton
        reference_skeleton = {
            "joints": REFERENCE_SKELETON["joints"].copy(),
            "parents": REFERENCE_SKELETON["parents"].copy(),
            "names": REFERENCE_SKELETON["names"].copy(),
        }

        return (mannequin_mesh, reference_skeleton)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GarmentPlacement": GarmentPlacement,
    "LoadReferenceMannequin": LoadReferenceMannequin,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GarmentPlacement": "Clothing: Place on Body",
    "LoadReferenceMannequin": "Clothing: Load Reference Mannequin",
}
