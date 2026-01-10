"""
UniRigAutoRig - Single node for complete rigging pipeline.
Takes mesh, outputs animation-ready FBX.
"""

import os
import sys
import time

# Support both relative imports (ComfyUI) and absolute imports (testing)
try:
    from .skeleton_extraction import UniRigExtractSkeletonNew
    from .skinning import UniRigApplySkinningMLNew
except ImportError:
    from skeleton_extraction import UniRigExtractSkeletonNew
    from skinning import UniRigApplySkinningMLNew


class UniRigAutoRig:
    """
    Single node for complete rigging pipeline.

    Combines skeleton extraction + skinning + normalization into one step.
    Takes mesh, outputs animation-ready FBX.

    For Mixamo template: outputs FBX normalized to Mixamo rest pose
    (T-pose, human scale, Hips at 1.04m) ready for Mixamo animations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "skeleton_model": ("UNIRIG_SKELETON_MODEL", {
                    "tooltip": "Pre-loaded skeleton model (from UniRigLoadSkeletonModel)"
                }),
                "skinning_model": ("UNIRIG_SKINNING_MODEL", {
                    "tooltip": "Pre-loaded skinning model (from UniRigLoadSkinningModel)"
                }),
            },
            "optional": {
                "skeleton_template": (["mixamo", "smpl", "vroid", "articulationxl"], {
                    "default": "mixamo",
                    "tooltip": "Skeleton template. 'mixamo' outputs Mixamo-compatible FBX ready for Mixamo animations. 'smpl' outputs SMPL-compatible skeleton."
                }),
                "fbx_name": ("STRING", {
                    "default": "",
                    "tooltip": "Custom filename for saved FBX (without extension). If empty, uses auto-generated name."
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 4294967295,
                    "tooltip": "Random seed for skeleton generation"
                }),
                "target_face_count": ("INT", {
                    "default": 50000,
                    "min": 10000,
                    "max": 500000,
                    "step": 10000,
                    "tooltip": "Target face count for mesh decimation. Higher = more detail, slower."
                }),
                # Skinning parameters
                "voxel_grid_size": ("INT", {
                    "default": 196,
                    "min": 64,
                    "max": 512,
                    "step": 64,
                    "tooltip": "Voxel grid resolution for spatial weight distribution. Higher = better quality, more VRAM. Default: 196 (model trained with this)"
                }),
                "num_samples": ("INT", {
                    "default": 32768,
                    "min": 8192,
                    "max": 131072,
                    "step": 8192,
                    "tooltip": "Number of surface samples for weight calculation. Higher = more accurate, slower. Default: 32768"
                }),
                "vertex_samples": ("INT", {
                    "default": 8192,
                    "min": 2048,
                    "max": 32768,
                    "step": 2048,
                    "tooltip": "Number of vertex samples. Higher = more accurate vertex processing. Default: 8192"
                }),
                "voxel_mask_power": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Power for voxel mask weight sharpness (alpha). Lower = smoother transitions. Default: 0.5 (model trained with this)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_output_path",)
    FUNCTION = "auto_rig"
    CATEGORY = "UniRig"

    def auto_rig(self, trimesh, skeleton_model, skinning_model,
                 skeleton_template="mixamo", fbx_name="", seed=42, target_face_count=50000,
                 voxel_grid_size=196, num_samples=32768, vertex_samples=8192, voxel_mask_power=0.5):
        """
        Complete rigging pipeline in one step.

        1. Extract skeleton from mesh
        2. Compute skin weights
        3. Normalize for target template (happens in blender_export_fbx.py)
        4. Export FBX
        """
        total_start = time.time()
        print(f"[UniRigAutoRig] Starting complete rigging pipeline...")
        print(f"[UniRigAutoRig] Skeleton template: {skeleton_template}")

        # Step 1: Extract skeleton
        print(f"[UniRigAutoRig] Step 1/2: Extracting skeleton...")
        step_start = time.time()

        skeleton_extractor = UniRigExtractSkeletonNew()
        normalized_mesh, skeleton, texture_preview = skeleton_extractor.extract(
            trimesh=trimesh,
            skeleton_model=skeleton_model,
            seed=seed,
            skeleton_template=skeleton_template,
            target_face_count=target_face_count
        )

        skeleton_time = time.time() - step_start
        print(f"[UniRigAutoRig] Skeleton extraction complete in {skeleton_time:.2f}s")
        print(f"[UniRigAutoRig] Extracted {len(skeleton.get('names', []))} bones")

        # Step 2: Apply skinning
        print(f"[UniRigAutoRig] Step 2/2: Applying skinning...")
        step_start = time.time()

        skinning_applier = UniRigApplySkinningMLNew()
        fbx_output_path, _ = skinning_applier.apply_skinning(
            normalized_mesh=normalized_mesh,
            skeleton=skeleton,
            skinning_model=skinning_model,
            fbx_name=fbx_name,
            voxel_grid_size=voxel_grid_size,
            num_samples=num_samples,
            vertex_samples=vertex_samples,
            voxel_mask_power=voxel_mask_power
        )

        skinning_time = time.time() - step_start
        print(f"[UniRigAutoRig] Skinning complete in {skinning_time:.2f}s")

        total_time = time.time() - total_start
        print(f"[UniRigAutoRig] ========================================")
        print(f"[UniRigAutoRig] Complete rigging pipeline finished!")
        print(f"[UniRigAutoRig] Total time: {total_time:.2f}s")
        print(f"[UniRigAutoRig] Output: {fbx_output_path}")
        print(f"[UniRigAutoRig] ========================================")

        return (fbx_output_path,)
