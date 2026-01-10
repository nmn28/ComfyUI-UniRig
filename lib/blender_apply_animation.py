"""
Blender script to apply a Mixamo animation to a rigged FBX model.
Uses constraint-based retargeting to properly transfer animation between armatures.
Usage: blender --background --python blender_apply_animation.py -- <model_fbx> <animation_fbx> <output_fbx>
"""

import bpy
import sys
import os
import math

# Get arguments after '--'
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

if len(argv) < 3:
    print("Usage: blender --background --python blender_apply_animation.py -- <model_fbx> <animation_fbx> <output_fbx>")
    sys.exit(1)

model_fbx = argv[0]
animation_fbx = argv[1]
output_fbx = argv[2]

print(f"[Blender Apply Animation] Model FBX: {model_fbx}")
print(f"[Blender Apply Animation] Animation FBX: {animation_fbx}")
print(f"[Blender Apply Animation] Output FBX: {output_fbx}")


def clean_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for c in bpy.data.armatures:
        bpy.data.armatures.remove(c)
    for c in bpy.data.meshes:
        bpy.data.meshes.remove(c)
    for c in bpy.data.materials:
        bpy.data.materials.remove(c)
    for c in bpy.data.images:
        bpy.data.images.remove(c)
    for c in bpy.data.actions:
        bpy.data.actions.remove(c)


def check_mixamo_prefix(bone_names):
    """Check if bones have mixamorig: prefix."""
    mixamo_count = sum(1 for name in bone_names if name.startswith("mixamorig:"))
    return mixamo_count, len(bone_names)


def get_bone_basename(bone_name):
    """Get bone name without mixamorig: prefix."""
    if bone_name.startswith("mixamorig:"):
        return bone_name[10:]  # Remove "mixamorig:" prefix
    return bone_name


clean_scene()

# Step 1: Import the rigged model FIRST (so we know what we're targeting)
print(f"[Blender Apply Animation] Importing model...")
try:
    bpy.ops.import_scene.fbx(filepath=model_fbx)
    print(f"[Blender Apply Animation] Model imported successfully")
except Exception as e:
    print(f"[Blender Apply Animation] Failed to import model: {e}")
    sys.exit(1)

# Find the model's armature and meshes
model_armature = None
model_meshes = []
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE':
        model_armature = obj
    elif obj.type == 'MESH':
        model_meshes.append(obj)

if model_armature is None:
    print(f"[Blender Apply Animation] Error: No armature found in model FBX")
    sys.exit(1)

model_bone_names = set(bone.name for bone in model_armature.pose.bones)
print(f"[Blender Apply Animation] Found model armature: {model_armature.name} with {len(model_bone_names)} bones")
print(f"[Blender Apply Animation] Found {len(model_meshes)} mesh object(s)")

# Check model bones for mixamo prefix
model_mixamo_count, model_total = check_mixamo_prefix(model_bone_names)
print(f"[Blender Apply Animation] Model has {model_mixamo_count}/{model_total} bones with mixamorig: prefix")

if model_mixamo_count == 0:
    print(f"[Blender Apply Animation] ERROR: Model does not have mixamorig: bone names!")
    print(f"[Blender Apply Animation] The model must use 'mixamo' skeleton template for Mixamo animations.")
    print(f"[Blender Apply Animation] Model bone names: {sorted(model_bone_names)[:10]}...")
    sys.exit(1)

# Store model objects to track
existing_objects = set(bpy.data.objects[:])

# Step 2: Import the animation FBX
print(f"[Blender Apply Animation] Importing animation...")
try:
    bpy.ops.import_scene.fbx(filepath=animation_fbx)
    print(f"[Blender Apply Animation] Animation imported successfully")
except Exception as e:
    print(f"[Blender Apply Animation] Failed to import animation: {e}")
    sys.exit(1)

# Find the animation armature
anim_armature = None
anim_meshes = []
for obj in bpy.data.objects:
    if obj not in existing_objects:
        if obj.type == 'ARMATURE':
            anim_armature = obj
        elif obj.type == 'MESH':
            anim_meshes.append(obj)

if anim_armature is None:
    print(f"[Blender Apply Animation] Error: No armature found in animation FBX")
    sys.exit(1)

anim_bone_names = set(bone.name for bone in anim_armature.pose.bones)
print(f"[Blender Apply Animation] Found animation armature: {anim_armature.name} with {len(anim_bone_names)} bones")

# Check animation bones for mixamo prefix
anim_mixamo_count, anim_total = check_mixamo_prefix(anim_bone_names)
print(f"[Blender Apply Animation] Animation has {anim_mixamo_count}/{anim_total} bones with mixamorig: prefix")

if anim_mixamo_count == 0:
    print(f"[Blender Apply Animation] ERROR: Animation does not have mixamorig: bone names!")
    print(f"[Blender Apply Animation] This animation file is not compatible with Mixamo-rigged models.")
    print(f"[Blender Apply Animation] Animation bone names: {sorted(anim_bone_names)[:10]}...")
    sys.exit(1)

# Get animation action and frame range
anim_action = None
if anim_armature.animation_data and anim_armature.animation_data.action:
    anim_action = anim_armature.animation_data.action
    print(f"[Blender Apply Animation] Found animation action: {anim_action.name}")
    print(f"[Blender Apply Animation] Frame range: {anim_action.frame_range[0]} - {anim_action.frame_range[1]}")
else:
    for action in bpy.data.actions:
        if action.users > 0 or anim_action is None:
            anim_action = action
    if anim_action:
        print(f"[Blender Apply Animation] Using action from data: {anim_action.name}")
    else:
        print(f"[Blender Apply Animation] Error: No animation action found")
        sys.exit(1)

frame_start = int(anim_action.frame_range[0])
frame_end = int(anim_action.frame_range[1])

# Delete any meshes from animation (we don't need them)
for mesh in anim_meshes:
    bpy.data.objects.remove(mesh, do_unlink=True)

# Find matching bones
matching_bones = model_bone_names.intersection(anim_bone_names)
print(f"[Blender Apply Animation] Matching bones: {len(matching_bones)} of {len(model_bone_names)} model bones")

if len(matching_bones) < 10:
    print(f"[Blender Apply Animation] WARNING: Very few matching bones ({len(matching_bones)})!")
    print(f"[Blender Apply Animation] Model bones sample: {sorted(model_bone_names)[:5]}")
    print(f"[Blender Apply Animation] Animation bones sample: {sorted(anim_bone_names)[:5]}")

    # This is a fatal mismatch
    if len(matching_bones) == 0:
        print(f"[Blender Apply Animation] ERROR: No matching bone names found!")
        print(f"[Blender Apply Animation] The model skeleton does not match the animation skeleton.")
        sys.exit(1)

# Step 3: Align armatures - make animation armature match model's location/rotation
# This is crucial for proper retargeting
print(f"[Blender Apply Animation] Aligning armatures...")

# Copy transforms from model to animation armature
anim_armature.location = model_armature.location.copy()
anim_armature.rotation_euler = model_armature.rotation_euler.copy()
anim_armature.rotation_quaternion = model_armature.rotation_quaternion.copy()
anim_armature.scale = model_armature.scale.copy()

# Step 4: Add Copy Location/Rotation constraints for retargeting
# Using separate constraints gives better control than Copy Transforms
print(f"[Blender Apply Animation] Setting up constraints for retargeting...")

bpy.context.view_layer.objects.active = model_armature
bpy.ops.object.mode_set(mode='POSE')

constraint_count = 0
for bone_name in matching_bones:
    if bone_name in model_armature.pose.bones and bone_name in anim_armature.pose.bones:
        pose_bone = model_armature.pose.bones[bone_name]

        # Add Copy Rotation constraint (most important for animation)
        rot_constraint = pose_bone.constraints.new('COPY_ROTATION')
        rot_constraint.name = "AnimRetarget_Rot"
        rot_constraint.target = anim_armature
        rot_constraint.subtarget = bone_name
        rot_constraint.mix_mode = 'REPLACE'
        rot_constraint.target_space = 'LOCAL'
        rot_constraint.owner_space = 'LOCAL'

        # Add Copy Location only for root bone (Hips)
        if bone_name in ["mixamorig:Hips", "Hips"]:
            loc_constraint = pose_bone.constraints.new('COPY_LOCATION')
            loc_constraint.name = "AnimRetarget_Loc"
            loc_constraint.target = anim_armature
            loc_constraint.subtarget = bone_name
            loc_constraint.target_space = 'LOCAL'
            loc_constraint.owner_space = 'LOCAL'

        constraint_count += 1

print(f"[Blender Apply Animation] Added constraints to {constraint_count} bones")

bpy.ops.object.mode_set(mode='OBJECT')

# Step 5: Bake the animation onto the model
print(f"[Blender Apply Animation] Baking animation (frames {frame_start} to {frame_end})...")

bpy.context.scene.frame_start = frame_start
bpy.context.scene.frame_end = frame_end
bpy.context.scene.frame_set(frame_start)

# Select only model armature for baking
bpy.ops.object.select_all(action='DESELECT')
model_armature.select_set(True)
bpy.context.view_layer.objects.active = model_armature

# Bake action
bpy.ops.nla.bake(
    frame_start=frame_start,
    frame_end=frame_end,
    only_selected=True,
    visual_keying=True,
    clear_constraints=True,
    use_current_action=False,
    bake_types={'POSE'}
)

print(f"[Blender Apply Animation] Animation baked successfully")

# Step 6: Clean up - remove animation armature
print(f"[Blender Apply Animation] Cleaning up...")
bpy.data.objects.remove(anim_armature, do_unlink=True)

# Step 7: Export the animated model
print(f"[Blender Apply Animation] Exporting animated FBX...")
os.makedirs(os.path.dirname(output_fbx) if os.path.dirname(output_fbx) else '.', exist_ok=True)

try:
    # Select model objects for export
    bpy.ops.object.select_all(action='DESELECT')
    model_armature.select_set(True)
    for mesh in model_meshes:
        if mesh.name in bpy.data.objects:
            mesh.select_set(True)

    bpy.context.view_layer.objects.active = model_armature

    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        use_selection=True,
        check_existing=False,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        path_mode='COPY',
        embed_textures=True,
        # Important: preserve materials
        mesh_smooth_type='FACE',
        use_mesh_modifiers=True,
    )

    print(f"[Blender Apply Animation] Saved to: {output_fbx}")
    print("[Blender Apply Animation] Done!")

except Exception as e:
    print(f"[Blender Apply Animation] Export failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
