# ComfyUI-UniRig

ComfyUI nodes for automatic 3D character rigging using NVIDIA's UniRig model.

## Architecture

```
nodes/           - ComfyUI node definitions
lib/             - Core logic and Blender scripts
  unirig/        - UniRig model code (submodule)
  blender/       - Bundled Blender 4.2.3 for FBX operations
assets/          - Skeleton templates and animation characters
```

## Key Files

- `nodes/auto_rig.py` - Combined rigging pipeline (skeleton + skinning + normalization)
- `lib/blender_export_fbx.py` - FBX export with Mixamo normalization (T-pose, scale, positioning)
- `lib/blender_apply_animation.py` - Animation retargeting via constraints

## Skeleton Templates

- **mixamo** - Mixamo-compatible naming, normalized to T-pose at human scale (1.7m), Hips at 1.04m
- **smpl** - SMPL body model skeleton
- **vroid** - VRoid/VRM skeleton
- **articulationxl** - Generic articulated skeleton

## Mixamo Normalization

UniRig generates skeletons adapted to input mesh geometry. For Mixamo animation compatibility, `blender_export_fbx.py` normalizes:

1. **Orientation** - Rotate model so face points -Y, arms along X axis
2. **T-pose** - Rotate arms horizontal using skin weights
3. **Scale** - Scale to 1.7m height
4. **Position** - Move Hips to Z=1.04m

## Animation Retargeting

`blender_apply_animation.py` uses world-space constraints to copy animation from Mixamo FBX to rigged model. Known issue: `nla.bake()` may produce identity keyframes instead of actual animation data.

## Dependencies

- PyTorch with CUDA
- Blender 4.2+ (bundled)
- trimesh, numpy

## Testing

```bash
# Verify FBX output
blender --background --python /tmp/check_script.py

# Check bone positions
python -c "import numpy as np; d=np.load('skeleton.npz'); print(d['bone_names'])"
```
