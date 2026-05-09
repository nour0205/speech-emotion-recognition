"""
TestMorphTargetDirect.py
────────────────────────────────────────────────────────────────────────────────
Directly tests USkeletalMeshComponent.set_morph_target on Amelia's face mesh,
bypassing the EmotionBridge driver entirely.

Purpose: isolate whether SetMorphTarget works on this mesh AT ALL.  If this
script makes Amelia's brows drop, then SetMorphTarget + MetaHuman morphs are
fine and the issue is in the driver.  If this script does NOT make the brows
drop, then MetaHuman's Post-Process ABP (RigLogic) is overwriting our writes
and we need a different approach.

HOW TO RUN
  1. Select the Amelia actor in the viewport (the Amelia_FaceMesh actor).
  2. Output Log → switch console dropdown from "Cmd" to "Python".
  3. Paste:
       import importlib, sys
       if 'TestMorphTargetDirect' in sys.modules:
           importlib.reload(sys.modules['TestMorphTargetDirect'])
       else:
           import TestMorphTargetDirect
       TestMorphTargetDirect.run()

WHAT TO WATCH
  - Amelia's brows.  After running, they should drop visibly (angry brow).
  - The Output Log for readback values.  If `read=0.00` everywhere, RigLogic is
    clobbering our writes.  If `read=1.00`, writes are sticking.
"""

import unreal


# Morph targets to stress.  These are the verbatim primary morph names for
# Amelia's face mesh, straight from the MorphTargetDump.
TEST_MORPHS = [
    ('head_lod0_mesh__brow_down_L',          1.0),
    ('head_lod0_mesh__brow_down_R',          1.0),
    ('head_lod0_mesh__eye_squintInner_L',    1.0),
    ('head_lod0_mesh__eye_squintInner_R',    1.0),
    ('head_lod0_mesh__nose_wrinkle_left',    1.0),
    ('head_lod0_mesh__nose_wrinkle_right',   1.0),
]


def _find_face_component(actor):
    """Return the SkeletalMeshComponent whose name contains 'Face'."""
    for comp in actor.get_components_by_class(unreal.SkeletalMeshComponent):
        if 'face' in comp.get_name().lower():
            return comp
    # Fallback: first SkeletalMeshComponent.
    skels = actor.get_components_by_class(unreal.SkeletalMeshComponent)
    return skels[0] if skels else None


def _enable_editor_ticking(mesh):
    """Make sure the mesh evaluates its pose in the Level Editor viewport."""
    try:
        mesh.set_editor_property('visibility_based_anim_tick_option',
                                  unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES)
    except Exception as e:
        unreal.log_warning(f'[MorphTest] Could not set VisibilityBasedAnimTickOption: {e}')
    for prop in ('update_animation_in_editor', 'update_cloth_in_editor'):
        try:
            mesh.set_editor_property(prop, True)
        except Exception:
            pass


def run():
    subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    selected = subsys.get_selected_level_actors() if subsys else []
    if not selected:
        unreal.log_error('[MorphTest] Select the Amelia actor in the viewport first.')
        return

    actor = selected[0]
    unreal.log(f'[MorphTest] Target actor: {actor.get_actor_label()}')

    mesh = _find_face_component(actor)
    if mesh is None:
        unreal.log_error(f'[MorphTest] No SkeletalMeshComponent on {actor.get_actor_label()}.')
        return

    unreal.log(f'[MorphTest] Using mesh component: {mesh.get_name()}')
    _enable_editor_ticking(mesh)

    skel_asset = mesh.get_editor_property('skeletal_mesh_asset')
    if skel_asset:
        unreal.log(f'[MorphTest] Skeletal asset: {skel_asset.get_name()}')

    # Write + immediate readback.
    unreal.log('[MorphTest] Writing morph weights (1.0 each) ...')
    for name, weight in TEST_MORPHS:
        mesh.set_morph_target(name, weight)
        read = mesh.get_morph_target(name)
        marker = 'OK' if abs(read - weight) < 0.01 else 'MISMATCH (downstream overwriting!)'
        unreal.log(f'[MorphTest]   {name}  wrote={weight:.2f}  read={read:.2f}  {marker}')

    unreal.log('[MorphTest] Done. Look at Amelia in the viewport — brows should be dropped, '
               'eyes squinted, nose wrinkled. If nothing moved but readback says "OK", '
               'the anim pipeline is clobbering weights at mesh-evaluation time.')


if __name__ == '__main__':
    run()
