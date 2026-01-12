# ComfyUI-UniRig

ComfyUI nodes for automatic 3D character rigging using UniRig and Make-It-Animatable.

## Architecture

```
nodes/           - ComfyUI node definitions
lib/             - Core logic and Blender scripts
  unirig/        - UniRig model code (submodule)
  blender/       - Bundled Blender 4.2.3 for FBX operations
  mia_inference.py - Make-It-Animatable inference wrapper
assets/          - Skeleton templates and animation characters
```

## Rigging Backends

### UniRig (SIGGRAPH 2025)
- **Use case**: Diverse 3D assets (humanoids, animals, objects)
- **Skeleton templates**: `mixamo` (humanoid), `articulationxl` (generic)
- **Speed**: Several seconds
- **Model**: Trained on Articulation-XL2.0 dataset

### Make-It-Animatable (CVPR 2025)
- **Use case**: Humanoid characters only
- **Skeleton**: Mixamo (52 bones with fingers)
- **Speed**: <1 second
- **Options**: `no_fingers`, `use_normal`, `reset_to_rest`

## Node List (11 nodes)

### UniRig Nodes
| Node | Purpose |
|------|---------|
| `UniRigLoadModel` | Load UniRig skeleton + skinning models |
| `UniRigAutoRig` | Full rigging pipeline |

### MIA Nodes
| Node | Purpose |
|------|---------|
| `MIALoadModel` | Load Make-It-Animatable models (5 PCAE models) |
| `MIAAutoRig` | Fast humanoid rigging |

### Shared Utility Nodes
| Node | Purpose |
|------|---------|
| `UniRigLoadMesh` | Load mesh files (OBJ, GLB, FBX, etc.) |
| `UniRigSaveMesh` | Save mesh files |
| `UniRigLoadRiggedMesh` | Load existing rigged FBX |
| `UniRigPreviewRiggedMesh` | Interactive FBX preview |
| `UniRigSaveSkeleton` | Save skeleton to file |
| `UniRigExportPosedFBX` | Export with custom pose |
| `UniRigApplyAnimation` | Apply Mixamo animations |

## Workflows

### UniRig Workflow (general 3D assets)
```
UniRigLoadMesh → UniRigLoadModel → UniRigAutoRig → FBX
                                   ↓
                                   skeleton_template: mixamo | articulationxl
```

### MIA Workflow (fast humanoid rigging)
```
UniRigLoadMesh → MIALoadModel → MIAAutoRig → FBX
                                ↓
                                Options: no_fingers, use_normal, reset_to_rest
```

## Skeleton Templates

- **mixamo** - Mixamo-compatible naming, normalized to T-pose at human scale (1.7m), Hips at 1.04m
- **articulationxl** - Generic articulated skeleton (native UniRig output)

Note: UniRig's VRoid-trained model is not yet released. Current model is trained on Articulation-XL2.0.

## Dependencies

- PyTorch with CUDA
- Blender 4.2+ (bundled)
- trimesh, numpy
- pytorch3d (for MIA)
- huggingface_hub (for model downloads)

## Model Downloads

Models are auto-downloaded from HuggingFace on first use:
- **UniRig**: `VAST-AI/UniRig` (~2GB)
- **MIA**: `jasongzy/Make-It-Animatable` (~500MB)
