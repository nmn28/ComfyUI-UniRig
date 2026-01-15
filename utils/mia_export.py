"""
Blender script for MIA (Make-It-Animatable) FBX export.
Applies MIA-predicted joints and weights to a mesh and exports to FBX.

Usage: blender --background --python mia_export.py -- --input_path <json> --output_path <fbx> --template_path <fbx> [options]
"""

import bpy
import sys
import os
import json
import numpy as np
from mathutils import Vector, Matrix


def parse_args():
    """Parse command line arguments after '--'."""
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    args = {
        "input_path": None,
        "output_path": None,
        "template_path": None,
        "remove_fingers": False,
        "reset_to_rest": False,
    }

    i = 0
    while i < len(argv):
        if argv[i] == "--input_path" and i + 1 < len(argv):
            args["input_path"] = argv[i + 1]
            i += 2
        elif argv[i] == "--output_path" and i + 1 < len(argv):
            args["output_path"] = argv[i + 1]
            i += 2
        elif argv[i] == "--template_path" and i + 1 < len(argv):
            args["template_path"] = argv[i + 1]
            i += 2
        elif argv[i] == "--remove_fingers":
            args["remove_fingers"] = True
            i += 1
        elif argv[i] == "--reset_to_rest":
            args["reset_to_rest"] = True
            i += 1
        else:
            i += 1

    return args


def reset_scene():
    """Clear all objects from the scene."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def load_fbx(filepath):
    """Load FBX file and return imported objects."""
    old_objs = set(bpy.context.scene.objects)
    bpy.ops.import_scene.fbx(filepath=filepath)
    new_objs = set(bpy.context.scene.objects) - old_objs
    return list(new_objs)


def get_armature(objects):
    """Find armature object in a list of objects."""
    for obj in objects:
        if obj.type == "ARMATURE":
            return obj
    return None


def get_meshes(objects):
    """Find all mesh objects in a list of objects."""
    return [obj for obj in objects if obj.type == "MESH"]


def get_template_bone_data(armature_obj):
    """
    Capture bone orientations from template armature before modification.
    Returns dict mapping bone name to roll value.
    """
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    bone_data = {}
    for bone in armature_obj.data.edit_bones:
        bone_data[bone.name] = {
            'roll': bone.roll,
            'head': tuple(bone.head),
            'tail': tuple(bone.tail),
            'matrix': [list(row) for row in bone.matrix],
        }

    bpy.ops.object.mode_set(mode='OBJECT')
    return bone_data


def apply_template_orientations(armature_obj, template_bone_data, bones_idx_dict):
    """
    Apply template bone rolls to MIA skeleton.
    This ensures bone orientations match Mixamo template for animation compatibility.
    """
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    applied_count = 0
    for bone in armature_obj.data.edit_bones:
        if bone.name in template_bone_data and bone.name in bones_idx_dict:
            template_data = template_bone_data[bone.name]
            # Set the bone roll to match template
            bone.roll = template_data['roll']
            applied_count += 1

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[MIA Export] Applied template orientations to {applied_count} bones")


def set_rest_bones(armature_obj, head, tail, bones_idx_dict, template_bone_data=None, reset_as_rest=False):
    """Update bone positions in the armature."""
    import bmesh

    if reset_as_rest:
        # Apply current pose as rest pose
        mesh_list = []
        for obj in armature_obj.children:
            if obj.type != "MESH":
                continue
            mesh_list.append(obj)
            bpy.context.view_layer.objects.active = obj
            for mod in obj.modifiers:
                if mod.type == 'ARMATURE':
                    bpy.ops.object.modifier_apply(modifier=mod.name)
                    break
            obj.select_set(True)
            bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
            obj.select_set(False)

        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.armature_apply(selected=False)
        bpy.ops.object.mode_set(mode='OBJECT')

        # Re-parent meshes
        for mesh_obj in mesh_list:
            mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.parent_set(type='ARMATURE')
        bpy.ops.object.select_all(action='DESELECT')

    # Update bone positions in edit mode
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    # First pass: update bones that are in the prediction dict
    # Use template roll if provided, otherwise preserve original roll direction
    for bone in armature_obj.data.edit_bones:
        bone.use_connect = False
        if bone.name in bones_idx_dict:
            idx = bones_idx_dict[bone.name]

            # Get template roll if available
            template_roll = None
            if template_bone_data and bone.name in template_bone_data:
                template_roll = template_bone_data[bone.name]['roll']

            # Update positions
            bone.head = Vector(head[idx])
            if tail is not None:
                bone.tail = Vector(tail[idx])

            # Apply template roll to ensure Mixamo animation compatibility
            if template_roll is not None:
                bone.roll = template_roll

    # Second pass: remove end/leaf bones not in MIA's prediction dict
    # These are bones like HeadTop_End, *Thumb4, *Index4, etc.
    # They have no skinning weights, so removing them doesn't affect deformation
    # Animation retargeting should still work with the 52 functional bones
    bones_to_remove = []
    for bone in armature_obj.data.edit_bones:
        if bone.name not in bones_idx_dict:
            bones_to_remove.append(bone.name)

    for bone_name in bones_to_remove:
        bone = armature_obj.data.edit_bones.get(bone_name)
        if bone:
            armature_obj.data.edit_bones.remove(bone)

    if bones_to_remove:
        print(f"[MIA Export] Removed {len(bones_to_remove)} end bones: {bones_to_remove[:5]}...")

    bpy.ops.object.mode_set(mode='OBJECT')

    if reset_as_rest:
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.transforms_clear()
        bpy.ops.object.mode_set(mode='OBJECT')

    return armature_obj


def set_weights(mesh_obj_list, weights, bones_idx_dict):
    """Apply bone weights to mesh objects."""
    if not mesh_obj_list:
        return

    # Calculate vertex counts per mesh
    vertices_num = [len(mesh_obj.data.vertices) for mesh_obj in mesh_obj_list]
    total_verts = sum(vertices_num)

    if total_verts != weights.shape[0]:
        print(f"[MIA Export] Warning: vertex count mismatch: {total_verts} vs {weights.shape[0]}")
        return

    # Split weights per mesh
    weights_list = np.split(weights, np.cumsum(vertices_num)[:-1])

    for mesh_obj, bw in zip(mesh_obj_list, weights_list):
        mesh_data = mesh_obj.data
        mesh_obj.vertex_groups.clear()

        for bone_name, bone_index in bones_idx_dict.items():
            group = mesh_obj.vertex_groups.new(name=bone_name)
            for v in mesh_data.vertices:
                v_w = bw[v.index, bone_index]
                if v_w > 1e-3:
                    group.add([v.index], float(v_w), "REPLACE")

        mesh_data.update()

    return mesh_obj_list


def remove_finger_bones(armature_obj, bones_idx_dict):
    """Remove finger bones from armature and update bones_idx_dict."""
    finger_prefixes = [
        "LeftHandThumb", "LeftHandIndex", "LeftHandMiddle", "LeftHandRing", "LeftHandPinky",
        "RightHandThumb", "RightHandIndex", "RightHandMiddle", "RightHandRing", "RightHandPinky",
        "mixamorig:LeftHandThumb", "mixamorig:LeftHandIndex", "mixamorig:LeftHandMiddle",
        "mixamorig:LeftHandRing", "mixamorig:LeftHandPinky",
        "mixamorig:RightHandThumb", "mixamorig:RightHandIndex", "mixamorig:RightHandMiddle",
        "mixamorig:RightHandRing", "mixamorig:RightHandPinky",
    ]

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    bones_to_remove = []
    for bone in armature_obj.data.edit_bones:
        for prefix in finger_prefixes:
            if bone.name.startswith(prefix):
                bones_to_remove.append(bone.name)
                break

    for bone_name in bones_to_remove:
        bone = armature_obj.data.edit_bones.get(bone_name)
        if bone:
            armature_obj.data.edit_bones.remove(bone)
        if bone_name in bones_idx_dict:
            del bones_idx_dict[bone_name]

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[MIA Export] Removed {len(bones_to_remove)} finger bones")
    return armature_obj


def export_fbx(output_path):
    """Export scene to FBX."""
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=False,
        object_types={'ARMATURE', 'MESH'},
        add_leaf_bones=False,
        bake_anim=False,
        path_mode='COPY',
        embed_textures=True,
    )
    print(f"[MIA Export] Exported to: {output_path}")


def main():
    args = parse_args()

    if not args["input_path"] or not args["output_path"] or not args["template_path"]:
        print("Usage: blender --background --python mia_export.py -- --input_path <json> --output_path <fbx> --template_path <fbx>")
        sys.exit(1)

    print(f"[MIA Export] Input: {args['input_path']}")
    print(f"[MIA Export] Output: {args['output_path']}")
    print(f"[MIA Export] Template: {args['template_path']}")
    print(f"[MIA Export] Remove fingers: {args['remove_fingers']}")
    print(f"[MIA Export] Reset to rest: {args['reset_to_rest']}")

    # Load input data
    with open(args["input_path"], 'r') as f:
        data = json.load(f)

    # Load binary arrays
    bw_shape = data["bw_shape"]
    joints_shape = data["joints_shape"]

    bw = np.fromfile(data["bw_path"], dtype=np.float32).reshape(bw_shape)
    joints = np.fromfile(data["joints_path"], dtype=np.float32).reshape(joints_shape)

    joints_tail = None
    if "joints_tail_path" in data:
        joints_tail_shape = data["joints_tail_shape"]
        joints_tail = np.fromfile(data["joints_tail_path"], dtype=np.float32).reshape(joints_tail_shape)

    bones_idx_dict = data["bones_idx_dict"]
    mesh_path = data["mesh_path"]

    print(f"[MIA Export] Loaded weights: {bw.shape}")
    print(f"[MIA Export] Loaded joints: {joints.shape}")
    print(f"[MIA Export] Bones: {len(bones_idx_dict)}")

    # Reset scene and load template
    reset_scene()
    template_objs = load_fbx(args["template_path"])
    armature = get_armature(template_objs)

    if armature is None:
        print("[MIA Export] ERROR: No armature found in template!")
        sys.exit(1)

    print(f"[MIA Export] Loaded template armature: {armature.name}")

    # Capture template bone orientations BEFORE any modifications
    # This data will be used to restore bone rolls after setting MIA positions
    template_bone_data = get_template_bone_data(armature)
    print(f"[MIA Export] Captured orientations for {len(template_bone_data)} template bones")

    # Reset armature to identity transform
    # Both mesh and joints from MIA are in normalized space, so we work in that space
    armature.matrix_world.identity()

    # Clear any pose transforms
    armature.animation_data_clear()
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.pose.transforms_clear()
    bpy.ops.object.mode_set(mode='OBJECT')

    # Load input mesh
    old_objs = set(bpy.context.scene.objects)

    if mesh_path.endswith(".glb") or mesh_path.endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif mesh_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=mesh_path)
    elif mesh_path.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=mesh_path)
    else:
        print(f"[MIA Export] ERROR: Unsupported mesh format: {mesh_path}")
        sys.exit(1)

    new_objs = set(bpy.context.scene.objects) - old_objs
    input_meshes = get_meshes(list(new_objs))

    if not input_meshes:
        print("[MIA Export] ERROR: No mesh found in input!")
        sys.exit(1)

    print(f"[MIA Export] Loaded {len(input_meshes)} mesh(es) from input")

    # Note: Both mesh and joints from MIA are in normalized space (approx [-1, 1])
    # We don't apply any transforms - they should be consistent with each other

    # Remove template meshes (we use input mesh instead)
    template_meshes = get_meshes(template_objs)
    for mesh in template_meshes:
        bpy.data.objects.remove(mesh, do_unlink=True)

    # Remove finger bones if requested
    if args["remove_fingers"]:
        remove_finger_bones(armature, bones_idx_dict)

    # Transform joints from Y-up (MIA inference space) to Z-up (Blender space)
    # The mesh GLB import already converts Y-up to Z-up, so we need to match
    # Conversion: (x, y, z) -> (x, -z, y) (Y-up to Z-up, preserving handedness)
    joints_blender = joints.copy()
    joints_blender[:, 1], joints_blender[:, 2] = -joints[:, 2].copy(), joints[:, 1].copy()

    joints_tail_blender = None
    if joints_tail is not None:
        joints_tail_blender = joints_tail.copy()
        joints_tail_blender[:, 1], joints_tail_blender[:, 2] = -joints_tail[:, 2].copy(), joints_tail[:, 1].copy()

    # Update bone positions with template orientations for animation compatibility
    set_rest_bones(armature, joints_blender, joints_tail_blender, bones_idx_dict,
                   template_bone_data=template_bone_data, reset_as_rest=args["reset_to_rest"])

    # Parent meshes to armature
    for mesh_obj in input_meshes:
        mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type='ARMATURE')
    bpy.ops.object.select_all(action='DESELECT')

    # Apply weights
    set_weights(input_meshes, bw, bones_idx_dict)

    # Update scene before export
    bpy.context.view_layer.update()

    # Export
    export_fbx(args["output_path"])
    print("[MIA Export] Done!")


if __name__ == "__main__":
    main()
