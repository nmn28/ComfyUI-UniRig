"""
Skinning nodes for UniRig - Apply skinning weights using ML models.

Uses comfy-env isolated environment for GPU dependencies.
Supports both subprocess inference (legacy) and direct Python inference.
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
import time
import shutil
import glob
import json
import folder_paths

from comfy_env import isolated

# Direct inference flag - when True, uses in-process inference without subprocess
# Enable direct inference for skinning (no subprocess)
USE_DIRECT_INFERENCE = True

# Support both relative imports (ComfyUI) and absolute imports (testing)
try:
    from .constants import BLENDER_TIMEOUT, INFERENCE_TIMEOUT
except ImportError:
    from constants import BLENDER_TIMEOUT, INFERENCE_TIMEOUT

try:
    from .base import (
        UNIRIG_PATH,
        BLENDER_EXE,
        UNIRIG_MODELS_DIR,
        LIB_DIR,
        setup_subprocess_env,
        decode_texture_to_comfy_image,
        create_placeholder_texture,
    )
except ImportError:
    from base import (
        UNIRIG_PATH,
        BLENDER_EXE,
        UNIRIG_MODELS_DIR,
        LIB_DIR,
        setup_subprocess_env,
        decode_texture_to_comfy_image,
        create_placeholder_texture,
    )

# In-process model cache module
_MODEL_CACHE_MODULE = None

# Direct inference module
_DIRECT_INFERENCE_MODULE = None

# Direct FBX export module (bpy as Python module)
_DIRECT_EXPORT_MODULE = None


def _get_direct_export():
    """Get the direct FBX export module for in-process export using bpy."""
    global _DIRECT_EXPORT_MODULE
    if _DIRECT_EXPORT_MODULE is None:
        export_path = os.path.join(LIB_DIR, "unirig", "src", "inference", "direct_export_fbx.py")
        if os.path.exists(export_path):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("unirig_direct_export", export_path)
                _DIRECT_EXPORT_MODULE = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_DIRECT_EXPORT_MODULE)
                print(f"[UniRig] Loaded direct FBX export module from {export_path}")
            except ImportError as e:
                print(f"[UniRig] Direct FBX export not available (bpy not installed): {e}")
                _DIRECT_EXPORT_MODULE = False
            except Exception as e:
                print(f"[UniRig] Warning: Could not load direct FBX export module: {e}")
                _DIRECT_EXPORT_MODULE = False
        else:
            print(f"[UniRig] Warning: Direct FBX export module not found at {export_path}")
            _DIRECT_EXPORT_MODULE = False
    return _DIRECT_EXPORT_MODULE if _DIRECT_EXPORT_MODULE else None


def _get_direct_inference():
    """Get the direct inference module for in-process model inference."""
    global _DIRECT_INFERENCE_MODULE
    if _DIRECT_INFERENCE_MODULE is None:
        direct_path = os.path.join(LIB_DIR, "unirig", "src", "inference", "direct.py")
        if os.path.exists(direct_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("unirig_direct", direct_path)
            _DIRECT_INFERENCE_MODULE = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_DIRECT_INFERENCE_MODULE)
            print(f"[UniRig] Loaded direct inference module from {direct_path}")
        else:
            print(f"[UniRig] Warning: Direct inference module not found at {direct_path}")
            _DIRECT_INFERENCE_MODULE = False
    return _DIRECT_INFERENCE_MODULE if _DIRECT_INFERENCE_MODULE else None


def _get_model_cache():
    """Get the in-process model cache module."""
    global _MODEL_CACHE_MODULE
    if _MODEL_CACHE_MODULE is None:
        # Use sys.modules to ensure same instance across all imports
        if "unirig_model_cache" in sys.modules:
            _MODEL_CACHE_MODULE = sys.modules["unirig_model_cache"]
        else:
            cache_path = os.path.join(LIB_DIR, "model_cache.py")
            if os.path.exists(cache_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("unirig_model_cache", cache_path)
                _MODEL_CACHE_MODULE = importlib.util.module_from_spec(spec)
                sys.modules["unirig_model_cache"] = _MODEL_CACHE_MODULE
                spec.loader.exec_module(_MODEL_CACHE_MODULE)
            else:
                print(f"[UniRig] Warning: Model cache module not found at {cache_path}")
                _MODEL_CACHE_MODULE = False
    return _MODEL_CACHE_MODULE if _MODEL_CACHE_MODULE else None


@isolated(env="unirig", import_paths=[".", ".."])
class UniRigApplySkinningMLNew:
    """
    Apply skinning weights using ML.

    Takes skeleton dict and mesh, prepares data and runs ML inference.

    Runs in isolated environment with GPU dependencies.
    Requires pre-loaded model from UniRigLoadSkinningModel.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normalized_mesh": ("TRIMESH",),
                "skeleton": ("SKELETON",),
                "skinning_model": ("UNIRIG_SKINNING_MODEL", {
                    "tooltip": "Pre-loaded skinning model (from UniRigLoadSkinningModel) - REQUIRED"
                }),
            },
            "optional": {
                "fbx_name": ("STRING", {
                    "default": "",
                    "tooltip": "Custom filename for saved FBX (without extension). If empty, uses rigged_<timestamp>.fbx"
                }),
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

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("fbx_output_path", "texture_preview")
    FUNCTION = "apply_skinning"
    CATEGORY = "UniRig"

    def apply_skinning(self, normalized_mesh, skeleton, skinning_model,
                       fbx_name=None, voxel_grid_size=None, num_samples=None, vertex_samples=None,
                       voxel_mask_power=None):
        print(f"[UniRigApplySkinningMLNew] Starting ML skinning (cached model only)...")

        # Validate model is provided
        if skinning_model is None:
            raise RuntimeError(
                "skinning_model is required for UniRigApplySkinningMLNew. "
                "Please connect a UniRigLoadSkinningModel node."
            )

        # Validate model has cache key
        if not skinning_model.get("model_cache_key"):
            # Check if there was an import error that caused cache loading to fail
            try:
                from .model_loaders import get_model_cache_error
            except ImportError:
                from model_loaders import get_model_cache_error

            import_error = get_model_cache_error()

            if import_error:
                error_str = str(import_error).lower()
                if "spconv" in error_str:
                    raise RuntimeError(
                        "UniRig model loading failed: spconv is not installed.\n\n"
                        "spconv is required for GPU-accelerated point cloud processing.\n\n"
                        "Install with:\n"
                        "  pip install spconv-cu118  # For CUDA 11.8\n"
                        "  pip install spconv-cu120  # For CUDA 12.0\n"
                        "  pip install spconv-cu121  # For CUDA 12.1\n\n"
                        "Choose the version matching your CUDA installation.\n"
                        "Check CUDA version with: nvidia-smi"
                    )
                elif "torch_scatter" in error_str:
                    raise RuntimeError(
                        "UniRig model loading failed: torch-scatter is not installed.\n\n"
                        "torch-scatter is required for efficient scatter operations.\n\n"
                        "Install with:\n"
                        "  pip install torch-scatter\n\n"
                        "Note: If no wheel is available for your PyTorch version,\n"
                        "you may need to build from source:\n"
                        "  pip install git+https://github.com/rusty1s/pytorch_scatter.git"
                    )
                elif "torch_cluster" in error_str:
                    raise RuntimeError(
                        "UniRig model loading failed: torch-cluster is not installed.\n\n"
                        "Install with:\n"
                        "  pip install torch-cluster\n\n"
                        "Note: If no wheel is available for your PyTorch version,\n"
                        "you may need to build from source:\n"
                        "  pip install git+https://github.com/rusty1s/pytorch_cluster.git"
                    )
                else:
                    raise RuntimeError(
                        f"UniRig model loading failed due to import error:\n{import_error}\n\n"
                        "Check that all required dependencies are installed.\n"
                        "Run: pip install -r requirements.txt"
                    )
            else:
                print(f"[UniRigApplySkinningMLNew] VRAM saving mode: No cached model available.")
                print(f"[UniRigApplySkinningMLNew] Falling back to subprocess inference...")
                # We will handle fallback later in the inference step
                pass

        print(f"[UniRigApplySkinningMLNew] Using pre-loaded cached model")
        task_config_path = skinning_model.get("task_config_path")

        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="unirig_skinning_new_")
        predict_skeleton_dir = os.path.join(temp_dir, "input")
        os.makedirs(predict_skeleton_dir, exist_ok=True)

        # Prepare skeleton NPZ from dict
        predict_skeleton_path = os.path.join(predict_skeleton_dir, "predict_skeleton.npz")
        save_data = {
            'joints': skeleton['joints'],
            'names': skeleton['names'],
            'parents': skeleton['parents'],
            'tails': skeleton['tails'],
        }

        # Add mesh data
        mesh_data_mapping = {
            'mesh_vertices': 'vertices',
            'mesh_faces': 'faces',
            'mesh_vertex_normals': 'vertex_normals',
            'mesh_face_normals': 'face_normals',
        }
        for skel_key, npz_key in mesh_data_mapping.items():
            if skel_key in skeleton:
                save_data[npz_key] = skeleton[skel_key]

        # Add optional RawData fields
        save_data['skin'] = None
        save_data['no_skin'] = None
        save_data['matrix_local'] = skeleton.get('matrix_local')
        save_data['path'] = None
        save_data['cls'] = skeleton.get('cls')

        # Add UV data if available
        if skeleton.get('uv_coords') is not None:
            save_data['uv_coords'] = skeleton['uv_coords']
            save_data['uv_faces'] = skeleton.get('uv_faces')
            print(f"[UniRigApplySkinningMLNew] UV data included: {len(skeleton['uv_coords'])} UVs")
        else:
            save_data['uv_coords'] = np.array([], dtype=np.float32)
            save_data['uv_faces'] = np.array([], dtype=np.int32)

        # Add texture data if available
        if skeleton.get('texture_data_base64') is not None:
            save_data['texture_data_base64'] = skeleton['texture_data_base64']
            save_data['texture_format'] = skeleton.get('texture_format', 'PNG')
            save_data['texture_width'] = skeleton.get('texture_width', 0)
            save_data['texture_height'] = skeleton.get('texture_height', 0)
            save_data['material_name'] = skeleton.get('material_name', '')
            print(f"[UniRigApplySkinningMLNew] Texture data included: {skeleton['texture_width']}x{skeleton['texture_height']} {skeleton['texture_format']}")
        else:
            save_data['texture_data_base64'] = ""
            save_data['texture_format'] = ""
            save_data['texture_width'] = 0
            save_data['texture_height'] = 0
            save_data['material_name'] = skeleton.get('material_name', '')

        np.savez(predict_skeleton_path, **save_data)
        print(f"[UniRigApplySkinningMLNew] Prepared skeleton NPZ: {predict_skeleton_path}")

        # Export mesh to GLB
        input_glb = os.path.join(temp_dir, "input.glb")

        normalized_mesh.export(input_glb)
        print(f"[UniRigApplySkinningMLNew] Exported mesh: {normalized_mesh.vertices.shape[0]} vertices, {normalized_mesh.faces.shape[0]} faces")

        # Run skinning inference
        step_start = time.time()
        output_fbx = os.path.join(temp_dir, "rigged.fbx")

        # Build config overrides from optional parameters
        config_overrides = {}
        if voxel_grid_size is not None:
            config_overrides['voxel_grid_size'] = voxel_grid_size
        if num_samples is not None:
            config_overrides['num_samples'] = num_samples
        if vertex_samples is not None:
            config_overrides['vertex_samples'] = vertex_samples
        if voxel_mask_power is not None:
            config_overrides['voxel_mask_power'] = voxel_mask_power

        if config_overrides:
            print(f"[UniRigApplySkinningMLNew] Config overrides: {config_overrides}")

        # Variables for inference results
        direct_skin_weights = None
        use_direct = USE_DIRECT_INFERENCE and _get_direct_inference() is not None

        if use_direct:
            # DIRECT INFERENCE - no subprocess, no Lightning Trainer
            print(f"[UniRigApplySkinningMLNew] Running skinning inference with DIRECT inference (no subprocess)...")
            direct_module = _get_direct_inference()

            # Get checkpoint path from skinning_model
            checkpoint_path = skinning_model.get("checkpoint_path")
            if not checkpoint_path:
                # Fallback: use default path
                checkpoint_path = os.path.join(UNIRIG_MODELS_DIR, "skin.safetensors")

            if not os.path.exists(checkpoint_path):
                raise RuntimeError(f"Skinning checkpoint not found: {checkpoint_path}")

            print(f"[UniRigApplySkinningMLNew] Using checkpoint: {checkpoint_path}")

            try:
                # Get mesh data from skeleton dict or normalized_mesh
                mesh_vertices = skeleton.get('mesh_vertices')
                if mesh_vertices is None:
                    mesh_vertices = np.array(normalized_mesh.vertices, dtype=np.float32)

                mesh_normals = skeleton.get('mesh_vertex_normals')
                if mesh_normals is None:
                    mesh_normals = np.array(normalized_mesh.vertex_normals, dtype=np.float32)

                joints = np.array(skeleton['joints'], dtype=np.float32)
                # Convert parent indices: None -> -1 for the model (handle before numpy conversion)
                parents_list = skeleton['parents']
                parents = np.array([-1 if p is None else int(p) for p in parents_list], dtype=np.int64)

                # Get mesh faces
                mesh_faces = skeleton.get('mesh_faces')
                if mesh_faces is None:
                    mesh_faces = np.array(normalized_mesh.faces, dtype=np.int32)

                # Get bone tails (if available)
                tails = skeleton.get('tails')
                if tails is not None:
                    tails = np.array(tails, dtype=np.float32)

                # Get voxel grid size from config overrides
                voxel_grid_size = config_overrides.get('voxel_grid_size', 196)

                print(f"[UniRigApplySkinningMLNew] Mesh: {len(mesh_vertices)} vertices, {len(mesh_faces)} faces")
                print(f"[UniRigApplySkinningMLNew] Skeleton: {len(joints)} joints")

                # Run direct skinning prediction
                direct_skin_weights = direct_module.predict_skinning(
                    vertices=mesh_vertices,
                    normals=mesh_normals,
                    joints=joints,
                    parents=parents,
                    checkpoint_path=checkpoint_path,
                    faces=mesh_faces,
                    tails=tails,
                    voxel_grid_size=voxel_grid_size,
                )

                inference_time = time.time() - step_start
                print(f"[UniRigApplySkinningMLNew] ✓ Direct inference completed in {inference_time:.2f}s")
                print(f"[UniRigApplySkinningMLNew] Skin weights shape: {direct_skin_weights.shape}")

            except Exception as e:
                import traceback
                print(f"[UniRigApplySkinningMLNew] Direct inference failed: {e}")
                traceback.print_exc()
                # Fall back to subprocess
                print(f"[UniRigApplySkinningMLNew] Falling back to subprocess inference...")
                use_direct = False
                direct_skin_weights = None

        if not use_direct:
            # SUBPROCESS INFERENCE (legacy fallback)
            cache_key = skinning_model.get("model_cache_key")

            if cache_key:
                print(f"[UniRigApplySkinningMLNew] Running skinning inference with cached model...")
                model_cache = _get_model_cache()
                if not model_cache:
                    raise RuntimeError(
                        "Model cache module not available. "
                        "Cannot run cached inference."
                    )

                print(f"[UniRigApplySkinningMLNew] Using cached model: {cache_key}")

                request_data = {
                    "seed": 123,
                    "input": input_glb,
                    "output": output_fbx,
                    "npz_dir": temp_dir,
                    "cls": skeleton.get('cls'),
                    "data_name": "predict_skeleton.npz",
                    "config_overrides": config_overrides,
                }

                try:
                    result = model_cache.run_inference(cache_key, request_data)
                    if "error" in result:
                        error_msg = result['error']
                        traceback_msg = result.get('traceback', 'No traceback available')
                        raise RuntimeError(
                            f"Cached model inference failed: {error_msg}\n"
                            f"Traceback:\n{traceback_msg}\n\n"
                            f"This node requires a working cached model. "
                            f"If you need fallback support, use UniRigApplySkinningML instead."
                        )

                    inference_time = time.time() - step_start
                    print(f"[UniRigApplySkinningMLNew] ✓ Cached inference completed in {inference_time:.2f}s")

                except Exception as e:
                    raise RuntimeError(
                        f"Cached model inference exception: {str(e)}\n\n"
                        f"This node requires a working cached model. "
                        f"If you need fallback support, use UniRigApplySkinningML instead."
                    )
            else:
                # FALLBACK TO SUBPROCESS (VRAM SAVING MODE)
                print(f"[UniRigApplySkinningMLNew] Running skinning inference with subprocess (VRAM saving mode)...")

                # Use lib/unirig/run.py directly
                unirig_run_py = os.path.join(UNIRIG_PATH, "run.py")
                if not os.path.exists(unirig_run_py):
                     raise RuntimeError(f"UniRig run script not found at {unirig_run_py}")

                # Build command
                run_cmd = [
                    sys.executable, unirig_run_py,
                    "--task", task_config_path,
                    "--seed", "123",
                    "--input", input_glb,
                    "--output", output_fbx,
                    "--npz_dir", temp_dir,
                    "--data_name", "predict_skeleton.npz",
                ]

                # Add config overrides as environment variable
                # (unirig/run.py supports UNIRIG_CONFIG_OVERRIDES)
                import json
                env = setup_subprocess_env()
                if config_overrides:
                    env['UNIRIG_CONFIG_OVERRIDES'] = json.dumps(config_overrides)

                # Add cls if available
                cls_value = skeleton.get('cls')
                if cls_value:
                    run_cmd.extend(["--cls", cls_value])

                print(f"[UniRigApplySkinningMLNew] Running command: {' '.join(run_cmd)}")

                try:
                    result = subprocess.run(
                        run_cmd,
                        capture_output=True,
                        text=True,
                        env=env,
                        cwd=UNIRIG_PATH,
                        timeout=INFERENCE_TIMEOUT
                    )

                    if result.stdout:
                         print(f"[UniRigApplySkinningMLNew] Run output:\n{result.stdout}")

                    if result.returncode != 0:
                        raise RuntimeError(
                            f"UniRig subprocess inference failed with exit code {result.returncode}\n"
                            f"Error output:\n{result.stderr}"
                        )

                    inference_time = time.time() - step_start
                    print(f"[UniRigApplySkinningMLNew] ✓ Subprocess inference completed in {inference_time:.2f}s")
                except subprocess.TimeoutExpired:
                    raise RuntimeError(f"UniRig inference timed out (>{INFERENCE_TIMEOUT}s)")
                except Exception as e:
                    print(f"[UniRigApplySkinningMLNew] Inference error: {e}")
                    raise

        # Generate FBX output
        if direct_skin_weights is not None:
            # DIRECT INFERENCE - generate FBX using direct bpy import (no subprocess)
            print(f"[UniRigApplySkinningMLNew] Generating FBX from direct inference results...")

            # Prepare data for FBX export
            mesh_vertices = skeleton.get('mesh_vertices')
            if mesh_vertices is None:
                mesh_vertices = np.array(normalized_mesh.vertices, dtype=np.float32)
            mesh_faces = skeleton.get('mesh_faces')
            if mesh_faces is None:
                mesh_faces = np.array(normalized_mesh.faces, dtype=np.int32)

            # Try direct FBX export first (bpy as Python module - no subprocess)
            direct_export = _get_direct_export()
            use_direct_export = direct_export is not None

            if use_direct_export:
                print(f"[UniRigApplySkinningMLNew] Using DIRECT bpy FBX export (no subprocess)...")
                try:
                    direct_export.export_rigged_fbx(
                        joints=skeleton['joints'],
                        parents=[int(p) if p is not None else -1 for p in skeleton['parents']],
                        names=list(skeleton['names']),
                        output_fbx=output_fbx,
                        vertices=mesh_vertices,
                        faces=mesh_faces,
                        skin=direct_skin_weights,
                        tails=skeleton.get('tails'),
                        uv_coords=skeleton.get('uv_coords'),
                        uv_faces=skeleton.get('uv_faces'),
                        texture_data_base64=skeleton.get('texture_data_base64') or '',
                        texture_format=skeleton.get('texture_format') or 'PNG',
                        material_name=skeleton.get('material_name') or 'Material',
                    )
                    print(f"[UniRigApplySkinningMLNew] ✓ FBX generated (direct bpy): {output_fbx}")

                except Exception as e:
                    print(f"[UniRigApplySkinningMLNew] Direct FBX export failed: {e}")
                    print(f"[UniRigApplySkinningMLNew] Falling back to Blender subprocess...")
                    use_direct_export = False

            if not use_direct_export:
                # Fallback: use Blender subprocess
                import pickle

                # Convert numpy arrays to plain Python lists for pickle
                fbx_data = {
                    'joints': skeleton['joints'].tolist() if isinstance(skeleton['joints'], np.ndarray) else skeleton['joints'],
                    'parents': [int(p) if p is not None else -1 for p in skeleton['parents']],
                    'names': list(skeleton['names']),
                    'vertices': mesh_vertices.tolist() if isinstance(mesh_vertices, np.ndarray) else mesh_vertices,
                    'faces': mesh_faces.tolist() if isinstance(mesh_faces, np.ndarray) else mesh_faces,
                    'skin': direct_skin_weights.tolist() if isinstance(direct_skin_weights, np.ndarray) else direct_skin_weights,
                    'tails': skeleton['tails'].tolist() if isinstance(skeleton.get('tails'), np.ndarray) else skeleton.get('tails'),
                    'uv_coords': skeleton.get('uv_coords', []),
                    'uv_faces': skeleton.get('uv_faces', []),
                    'texture_data_base64': skeleton.get('texture_data_base64', ''),
                    'texture_format': skeleton.get('texture_format', 'PNG'),
                    'material_name': skeleton.get('material_name', 'Material'),
                }

                if isinstance(fbx_data['uv_coords'], np.ndarray):
                    fbx_data['uv_coords'] = fbx_data['uv_coords'].tolist()
                if isinstance(fbx_data['uv_faces'], np.ndarray):
                    fbx_data['uv_faces'] = fbx_data['uv_faces'].tolist()

                pkl_path = os.path.join(temp_dir, "skinned_data.pkl")
                with open(pkl_path, 'wb') as f:
                    pickle.dump(fbx_data, f)

                print(f"[UniRigApplySkinningMLNew] Using Blender subprocess for FBX export...")

                blender_export_script = os.path.join(LIB_DIR, "blender_export_fbx.py")
                if not os.path.exists(blender_export_script):
                    raise RuntimeError(f"Blender FBX export script not found: {blender_export_script}")

                blender_cmd = [
                    BLENDER_EXE,
                    "--background",
                    "--python", blender_export_script,
                    "--",
                    pkl_path,
                    output_fbx,
                ]

                try:
                    result = subprocess.run(
                        blender_cmd,
                        capture_output=True,
                        text=True,
                        timeout=BLENDER_TIMEOUT
                    )
                    if result.stdout:
                        for line in result.stdout.split('\n'):
                            if '[Blender FBX Export]' in line:
                                print(f"[UniRigApplySkinningMLNew] {line}")

                    if result.returncode != 0:
                        print(f"[UniRigApplySkinningMLNew] Blender stderr: {result.stderr}")
                        raise RuntimeError(f"Blender FBX export failed with code {result.returncode}")

                    print(f"[UniRigApplySkinningMLNew] ✓ FBX generated (subprocess): {output_fbx}")

                except subprocess.TimeoutExpired:
                    raise RuntimeError(f"Blender FBX export timed out (>{BLENDER_TIMEOUT}s)")
                except Exception as e:
                    print(f"[UniRigApplySkinningMLNew] FBX export error: {e}")
                    raise

        print(f"[UniRigApplySkinningMLNew] Skinning completed")

        # Find output FBX
        possible_paths = [
            output_fbx,
            os.path.join(temp_dir, "rigged.fbx"),
            os.path.join(temp_dir, "output", "rigged.fbx"),
        ]

        fbx_path = None
        for path in possible_paths:
            if os.path.exists(path):
                fbx_path = path
                break

        if not fbx_path:
            # Search for any FBX files
            search_paths = [temp_dir, os.path.join(temp_dir, "output")]
            for search_dir in search_paths:
                if os.path.exists(search_dir):
                    fbx_files = glob.glob(os.path.join(search_dir, "*.fbx"))
                    if fbx_files:
                        fbx_path = fbx_files[0]
                        break

        if not fbx_path or not os.path.exists(fbx_path):
            raise RuntimeError(f"Skinning output FBX not found. Searched: {possible_paths}")

        print(f"[UniRigApplySkinningMLNew] Found output FBX: {fbx_path}")
        print(f"[UniRigApplySkinningMLNew] FBX file size: {os.path.getsize(fbx_path)} bytes")

        # Auto-save FBX to output directory
        output_dir = folder_paths.get_output_directory()

        # Determine output filename with skeleton template suffix
        template_suffix = skeleton.get('output_format', 'unknown')

        if fbx_name and fbx_name.strip():
            # Use custom name from user
            base_name = fbx_name.strip()
            # Remove .fbx extension if present
            if base_name.lower().endswith('.fbx'):
                base_name = base_name[:-4]
            output_filename = f"{base_name}_{template_suffix}.fbx"
        else:
            # Use auto-generated name with timestamp
            output_filename = f"rigged_{int(time.time())}_{template_suffix}.fbx"

        output_path = os.path.join(output_dir, output_filename)
        shutil.copy(fbx_path, output_path)

        print(f"[UniRigApplySkinningMLNew] Auto-saved FBX to output: {output_filename}")
        print(f"[UniRigApplySkinningMLNew] Full path: {output_path}")

        # Create texture preview output
        texture_preview = None
        if skeleton.get('texture_data_base64'):
            texture_preview, tex_w, tex_h = decode_texture_to_comfy_image(skeleton['texture_data_base64'])
            if texture_preview is not None:
                print(f"[UniRigApplySkinningMLNew] Texture preview created: {tex_w}x{tex_h}")
            else:
                print(f"[UniRigApplySkinningMLNew] Warning: Could not decode texture for preview")
                texture_preview = create_placeholder_texture()
        else:
            print(f"[UniRigApplySkinningMLNew] No texture available for preview")
            texture_preview = create_placeholder_texture()

        print(f"[UniRigApplySkinningMLNew] Skinning application complete!")

        # Clean up temporary directory
        # Windows-specific: Force garbage collection and retry logic to release file handles
        # This helps prevent "file in use by another process" errors during cleanup
        if sys.platform == 'win32':
            import gc
            gc.collect()
            # Give Windows a moment to release file handles
            time.sleep(0.1)

        try:
            # First attempt with ignore_errors
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"[UniRigApplySkinningMLNew] Cleaned up temp directory")
        except Exception as e:
            # Don't fail the whole operation if cleanup fails
            print(f"[UniRigApplySkinningMLNew] Warning: Could not clean up temp directory: {e}")

        # Windows: If directory still exists, schedule for deletion on restart
        if sys.platform == 'win32' and os.path.exists(temp_dir):
            print(f"[UniRigApplySkinningMLNew] Note: Temp directory will be cleaned on next restart: {temp_dir}")

        return (output_filename, texture_preview)
