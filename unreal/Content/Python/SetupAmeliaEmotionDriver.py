"""
SetupAmeliaEmotionDriver.py
───────────────────────────────────────────────────────────────────────────────
One-time setup script for the EmotionBridge → MetaHuman face system.

WHAT IT DOES
  1. Restores ABP_Amelia_FaceMesh_PostProcess as the PostProcess AnimBP on
     the Amelia_FaceMesh asset.  This step is required because the PostProcess
     ABP was cleared during debugging.  Without it the MetaHuman DNA solver
     never runs and the face shows zero deformation.

  NOTE: The Face SkeletalMeshComponent's Anim Class (MetaHumanEmotionAnimInstance)
  is now installed automatically by UMetaHumanEmotionDriverComponent at runtime —
  no manual Blueprint editing is needed for that part.

HOW TO RUN
  In the Unreal Editor Output Log, switch the console dropdown to "Python" and type:
      import importlib, sys
      if 'SetupAmeliaEmotionDriver' in sys.modules:
          importlib.reload(sys.modules['SetupAmeliaEmotionDriver'])
      else:
          import SetupAmeliaEmotionDriver

  OR run from the menu: Tools → Execute Python Script → select this file.

PREREQUISITES
  • The EmotionBridge plugin must be built (Visual Studio rebuild).
  • The Python plugin must be enabled in the project (Project Settings → Plugins).
───────────────────────────────────────────────────────────────────────────────
"""

import unreal

# ── Asset paths ───────────────────────────────────────────────────────────────

FACE_MESH_PATH     = '/Game/MetaHumans/Amelia/Face/Amelia_FaceMesh'
POST_PROCESS_PATH  = '/Game/MetaHumans/Amelia/Face/ABP_Amelia_FaceMesh_PostProcess'

# ── Helper ────────────────────────────────────────────────────────────────────

def _load_anim_bp_class(asset_path: str):
    """Return the generated UClass for a Blueprint asset, or None on failure."""
    bp_asset = unreal.load_asset(asset_path)
    if bp_asset is None:
        unreal.log_error(f'[EmotionSetup] Cannot load asset: {asset_path}')
        return None
    generated = bp_asset.generated_class()
    if generated is None:
        unreal.log_error(
            f'[EmotionSetup] {asset_path} has no generated class. '
            f'Compile the Blueprint first.'
        )
    return generated


# ── Step 1: Restore PostProcess ABP ──────────────────────────────────────────

def restore_post_process_abp() -> bool:
    """
    Sets ABP_Amelia_FaceMesh_PostProcess as the PostProcess AnimBP on the
    Amelia_FaceMesh SkeletalMesh asset and saves it.

    Returns True on success, False on failure.
    """
    face_mesh = unreal.load_asset(FACE_MESH_PATH)
    if face_mesh is None:
        unreal.log_error(
            f'[EmotionSetup] Cannot find {FACE_MESH_PATH}. '
            f'Check that MetaHumans/Amelia is imported.'
        )
        return False

    post_process_class = _load_anim_bp_class(POST_PROCESS_PATH)
    if post_process_class is None:
        return False

    # Check current value — skip if already set correctly.
    current_pp = face_mesh.get_editor_property('post_process_anim_blueprint')
    if current_pp is not None and current_pp == post_process_class:
        unreal.log('[EmotionSetup] PostProcess ABP already set — no change needed.')
        return True

    with unreal.ScopedEditorTransaction('EmotionBridge: Restore PostProcess ABP'):
        face_mesh.set_editor_property('post_process_anim_blueprint', post_process_class)

    saved = unreal.EditorAssetLibrary.save_asset(FACE_MESH_PATH, only_if_is_dirty=False)
    if saved:
        unreal.log(f'[EmotionSetup] Restored PostProcess ABP on Amelia_FaceMesh. Saved.')
    else:
        unreal.log_warning(
            '[EmotionSetup] PostProcess ABP set in memory but save failed. '
            'Save the asset manually from the Content Browser.'
        )
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    unreal.log('=' * 70)
    unreal.log('[EmotionSetup] Starting Amelia Emotion Driver setup...')
    unreal.log('=' * 70)

    ok = restore_post_process_abp()

    unreal.log('─' * 70)
    if ok:
        unreal.log('[EmotionSetup] Setup complete!')
        unreal.log('')
        unreal.log('Next steps:')
        unreal.log('  1. Rebuild the EmotionBridge plugin (Visual Studio → Build).')
        unreal.log('     MetaHumanEmotionAnimInstance will be installed automatically')
        unreal.log('     on the Face mesh when Play Demo runs in the Emotion Bridge panel.')
        unreal.log('  2. In UE5, open the Emotion Bridge panel and click Play Demo.')
        unreal.log('     The face should now animate with the correct CTRL_ curves.')
        unreal.log('  3. Watch the Output Log for:')
        unreal.log('       MetaHumanEmotionDriver: installed MetaHumanEmotionAnimInstance on ...')
        unreal.log('       MetaHumanEmotionDriver: auto-detected face mesh ...')
    else:
        unreal.log_error('[EmotionSetup] Setup FAILED. Check the errors above.')
    unreal.log('=' * 70)


run()
