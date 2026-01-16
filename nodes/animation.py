"""
UniRig Animation Node - Apply animations to rigged meshes
"""

import os
import subprocess
import time
from pathlib import Path

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_INPUT_FOLDER = folder_paths.get_input_directory()
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except:
    COMFYUI_INPUT_FOLDER = None
    COMFYUI_OUTPUT_FOLDER = None

# Import from base module
try:
    from .base import BLENDER_EXE, LIB_DIR
except ImportError:
    from base import BLENDER_EXE, LIB_DIR

# Blender animation script
BLENDER_APPLY_ANIMATION = str(LIB_DIR / "blender_apply_animation.py")

# Animation templates folder (in input directory, copied by prestartup_script.py)
ANIMATION_TEMPLATES_DIR = Path(COMFYUI_INPUT_FOLDER) / "animation_templates" if COMFYUI_INPUT_FOLDER else None

# Timeout for Blender operations
BLENDER_TIMEOUT = 300  # 5 minutes for animation baking


class UniRigApplyAnimation:
    """
    Apply an animation to a rigged FBX model.
    Supports Mixamo FBX animations and SMPL NPZ animations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get animation files for default (mixamo)
        mixamo_files = cls.get_animation_files("mixamo")

        return {
            "required": {
                "model_fbx_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to the rigged model FBX file (from UniRig: Apply Skinning output)"
                }),
                "animation_type": (["mixamo", "smpl"], {
                    "default": "mixamo",
                    "tooltip": "Animation format: mixamo (FBX) or smpl (NPZ). Refresh node after changing."
                }),
                "animation_file": (mixamo_files, {
                    "default": mixamo_files[0] if mixamo_files else "No animations found",
                    "tooltip": "Animation file to apply. Add more animations to input/animation_templates/<type>/"
                }),
            },
            "optional": {
                "output_name": ("STRING", {
                    "default": "",
                    "tooltip": "Custom output filename (without extension). If empty, auto-generates name."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("animated_fbx_path",)
    FUNCTION = "apply_animation"
    CATEGORY = "UniRig/Animation"
    OUTPUT_NODE = True

    @classmethod
    def get_animation_files(cls, animation_type):
        """Get list of animation files for the specified type."""
        animation_files = []

        if ANIMATION_TEMPLATES_DIR is None:
            return ["No animations found"]

        if animation_type == "mixamo":
            folder = ANIMATION_TEMPLATES_DIR / "mixamo"
            extensions = ['.fbx']
        else:  # smpl
            folder = ANIMATION_TEMPLATES_DIR / "smpl"
            extensions = ['.npz']

        if folder.exists():
            for file in folder.iterdir():
                if file.is_file() and file.suffix.lower() in extensions:
                    animation_files.append(file.name)

        return sorted(animation_files) if animation_files else ["No animations found"]

    @classmethod
    def IS_CHANGED(cls, model_fbx_path, animation_type, animation_file, output_name=""):
        """Force re-execution when inputs change."""
        if ANIMATION_TEMPLATES_DIR is None:
            return f"{model_fbx_path}:{animation_type}:{animation_file}"

        anim_path = ANIMATION_TEMPLATES_DIR / animation_type / animation_file
        if anim_path.exists():
            return f"{model_fbx_path}:{os.path.getmtime(anim_path)}"
        return f"{model_fbx_path}:{animation_type}:{animation_file}"

    def apply_animation(self, model_fbx_path, animation_type, animation_file, output_name=""):
        """Apply animation to rigged model using Blender."""

        # Validate inputs
        if not model_fbx_path or not model_fbx_path.strip():
            raise ValueError("Model FBX path cannot be empty")

        if animation_file == "No animations found":
            raise ValueError(f"No animation files found for type: {animation_type}")

        if not os.path.exists(model_fbx_path):
            raise ValueError(f"Model FBX not found: {model_fbx_path}")

        if ANIMATION_TEMPLATES_DIR is None:
            raise ValueError("Animation templates directory not found")

        # Get animation file path
        animation_path = ANIMATION_TEMPLATES_DIR / animation_type / animation_file

        if not animation_path.exists():
            raise ValueError(f"Animation file not found: {animation_path}")

        # Check Blender availability
        if not BLENDER_EXE or not os.path.exists(BLENDER_EXE):
            raise RuntimeError(
                "Blender is required for animation application.\n"
                "Run: comfy-env install"
            )

        if not os.path.exists(BLENDER_APPLY_ANIMATION):
            raise RuntimeError(f"Blender animation script not found: {BLENDER_APPLY_ANIMATION}")

        print(f"[UniRigApplyAnimation] Model: {model_fbx_path}")
        print(f"[UniRigApplyAnimation] Animation type: {animation_type}")
        print(f"[UniRigApplyAnimation] Animation: {animation_file}")

        # Early validation: Check if model filename suggests skeleton template compatibility
        model_basename = os.path.basename(model_fbx_path).lower()
        if animation_type == "mixamo":
            # Mixamo animations require mixamo skeleton template
            if "_smpl.fbx" in model_basename or "_smpl_" in model_basename:
                raise ValueError(
                    "Skeleton template mismatch!\n"
                    f"Model appears to use SMPL skeleton (based on filename: {os.path.basename(model_fbx_path)})\n"
                    "but Mixamo animations require the 'mixamo' skeleton template.\n\n"
                    "To fix: Re-run skeleton extraction with skeleton_template='mixamo'"
                )
            if "_vroid.fbx" in model_basename or "_vroid_" in model_basename:
                raise ValueError(
                    "Skeleton template mismatch!\n"
                    f"Model appears to use VRoid skeleton (based on filename: {os.path.basename(model_fbx_path)})\n"
                    "but Mixamo animations require the 'mixamo' skeleton template.\n\n"
                    "To fix: Re-run skeleton extraction with skeleton_template='mixamo'"
                )
            # Positive check: should have mixamo in name
            if "_mixamo" not in model_basename:
                print(f"[UniRigApplyAnimation] Warning: Model filename does not contain '_mixamo'. "
                      f"Make sure you used skeleton_template='mixamo' when extracting the skeleton.")

        # Handle different animation types
        if animation_type == "smpl":
            # TODO: SMPL animation support requires different processing
            raise NotImplementedError(
                "SMPL animation support is not yet implemented. "
                "Please use Mixamo animations for now."
            )

        # Determine output path
        output_dir = COMFYUI_OUTPUT_FOLDER if COMFYUI_OUTPUT_FOLDER else os.path.dirname(model_fbx_path)

        if output_name and output_name.strip():
            output_filename = output_name.strip()
            if not output_filename.lower().endswith('.fbx'):
                output_filename += '.fbx'
        else:
            # Auto-generate name from model + animation
            model_name = os.path.splitext(os.path.basename(model_fbx_path))[0]
            anim_name = os.path.splitext(animation_file)[0]
            output_filename = f"{model_name}_{anim_name}.fbx"

        output_path = os.path.join(output_dir, output_filename)
        print(f"[UniRigApplyAnimation] Output: {output_path}")

        # Run Blender script
        cmd = [
            BLENDER_EXE,
            "--background",
            "--python", BLENDER_APPLY_ANIMATION,
            "--",
            model_fbx_path,
            str(animation_path),
            output_path
        ]

        print(f"[UniRigApplyAnimation] Running Blender...")
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=BLENDER_TIMEOUT
            )

            # Log Blender output and check for specific errors
            skeleton_mismatch = False
            error_details = []
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('[Blender'):
                        print(f"[UniRigApplyAnimation] {line}")
                        # Check for skeleton mismatch errors
                        if "ERROR:" in line:
                            error_details.append(line)
                            if "mixamorig:" in line or "bone names" in line.lower():
                                skeleton_mismatch = True

            if result.returncode != 0:
                if skeleton_mismatch:
                    error_msg = (
                        "Skeleton mismatch error!\n"
                        "The model's skeleton does not match the animation skeleton.\n\n"
                        "For Mixamo animations, the model must use skeleton_template='mixamo'.\n"
                        "Please re-run UniRig: Extract Skeleton with the correct template.\n\n"
                        "Details:\n" + "\n".join(error_details)
                    )
                else:
                    error_msg = f"Blender animation failed (code {result.returncode})"
                    if error_details:
                        error_msg += "\n" + "\n".join(error_details)
                    elif result.stderr:
                        error_msg += f"\n{result.stderr[:1000]}"
                raise RuntimeError(error_msg)

            if not os.path.exists(output_path):
                raise RuntimeError("Blender did not produce output FBX file")

            elapsed = time.time() - start_time
            print(f"[UniRigApplyAnimation] Animation applied in {elapsed:.2f}s")
            print(f"[UniRigApplyAnimation] Output: {output_path}")

            return (output_path,)

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Blender animation timed out after {BLENDER_TIMEOUT}s")
        except Exception as e:
            raise RuntimeError(f"Animation application failed: {str(e)}")
