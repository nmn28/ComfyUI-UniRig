"""UniRig Nodes."""

from .mesh_io import UniRigLoadMesh, UniRigSaveMesh
from .model_loaders import UniRigLoadModel
from .auto_rig import UniRigAutoRig
from .skeleton_extraction import UniRigExtractSkeletonNew
from .skinning import UniRigApplySkinningMLNew
from .mia_model_loader import MIALoadModel
from .mia_auto_rig import MIAAutoRig
from .animation import UniRigApplyAnimation
from .skeleton_io import (
    UniRigLoadRiggedMesh,
    UniRigPreviewRiggedMesh,
    UniRigExportPosedFBX,
    UniRigViewRigging,
    UniRigDebugSkeleton,
    UniRigCompareSkeletons,
)
from .rest_pose_node import UniRigExtractRestPose
from .orientation_check import UniRigOrientationCheck

# Clothing pipeline nodes
from .cloth_fit import LoadClothingMesh, ClothFitGarment
from .weight_transfer import TransferSkinWeights, CombineAvatarClothing

NODE_CLASS_MAPPINGS = {
    "UniRigLoadMesh": UniRigLoadMesh,
    "UniRigSaveMesh": UniRigSaveMesh,
    "UniRigLoadModel": UniRigLoadModel,
    "UniRigAutoRig": UniRigAutoRig,
    "UniRigExtractSkeletonNew": UniRigExtractSkeletonNew,
    "UniRigApplySkinningMLNew": UniRigApplySkinningMLNew,
    "MIALoadModel": MIALoadModel,
    "MIAAutoRig": MIAAutoRig,
    "UniRigApplyAnimation": UniRigApplyAnimation,
    "UniRigLoadRiggedMesh": UniRigLoadRiggedMesh,
    "UniRigPreviewRiggedMesh": UniRigPreviewRiggedMesh,
    "UniRigExportPosedFBX": UniRigExportPosedFBX,
    "UniRigViewRigging": UniRigViewRigging,
    "UniRigDebugSkeleton": UniRigDebugSkeleton,
    "UniRigCompareSkeletons": UniRigCompareSkeletons,
    "UniRigExtractRestPose": UniRigExtractRestPose,
    "UniRigOrientationCheck": UniRigOrientationCheck,
    # Clothing pipeline
    "LoadClothingMesh": LoadClothingMesh,
    "ClothFitGarment": ClothFitGarment,
    "TransferSkinWeights": TransferSkinWeights,
    "CombineAvatarClothing": CombineAvatarClothing,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniRigLoadMesh": "UniRig: Load Mesh",
    "UniRigSaveMesh": "UniRig: Save Mesh",
    "UniRigLoadModel": "UniRig: Load Model",
    "UniRigAutoRig": "UniRig: Auto Rig",
    "UniRigExtractSkeletonNew": "UniRig: Extract Skeleton",
    "UniRigApplySkinningMLNew": "UniRig: Apply Skinning ML",
    "MIALoadModel": "MIA: Load Model",
    "MIAAutoRig": "MIA: Auto Rig",
    "UniRigApplyAnimation": "UniRig: Apply Animation",
    "UniRigLoadRiggedMesh": "UniRig: Load Rigged Mesh",
    "UniRigPreviewRiggedMesh": "UniRig: Preview Rigged Mesh",
    "UniRigExportPosedFBX": "UniRig: Export Posed FBX",
    "UniRigViewRigging": "UniRig: View Rigging",
    "UniRigDebugSkeleton": "UniRig: Debug Skeleton",
    "UniRigCompareSkeletons": "UniRig: Compare Skeletons",
    "UniRigExtractRestPose": "UniRig: Extract Rest Pose",
    "UniRigOrientationCheck": "UniRig: Orientation Check",
    # Clothing pipeline
    "LoadClothingMesh": "Clothing: Load Mesh",
    "ClothFitGarment": "Clothing: Cloth-Fit Garment",
    "TransferSkinWeights": "Clothing: Transfer Skin Weights",
    "CombineAvatarClothing": "Clothing: Combine Avatar + Clothing",
}
