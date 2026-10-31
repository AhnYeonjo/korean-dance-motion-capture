"""Create a mocap skeleton in a character .blend and make the character's rig follow it.

python build_blend.py <character.blend> <profile.json> <targets.json> <out.blend>

 1. Keyframed target empties from the fused front+side motion (temporary).
 2. New armature 'MOCAP_Skeleton' copying the rest layout of the character's FK controls,
    driven by Damped/Locked Track constraints to the targets, then baked to plain keyframes.
 3. Character rig switched to FK; each FK control copies the world rotation of its MOCAP bone
    (identical rest orientation => identical pose). Torso master copies location, top root bone
    carries the travel so every helper bone moves along.
"""
import bpy, json, sys
from mathutils import Vector
try:
    from bpy_extras import anim_utils
except Exception:
    anim_utils = None

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
src, prof_path, tgt_json, out_path = args[:4]
bpy.ops.wm.open_mainfile(filepath=src)
prof = json.load(open(prof_path)); BM = prof['bones']
data = json.load(open(tgt_json)); N = data['frames']; T = data['targets']
scene = bpy.context.scene
scene.render.fps = 30; scene.render.fps_base = 1.001
scene.frame_start = 1; scene.frame_end = N
rig = bpy.data.objects[prof['rig']]
if rig.animation_data: rig.animation_data.action = None

# ---------- 1. target empties
coll = bpy.data.collections.new('MOCAP'); scene.collection.children.link(coll)
tcoll = bpy.data.collections.new('MOCAP_targets'); coll.children.link(tcoll)

def set_curve(obj, path, pts):
    setattr(obj, path, pts[0]); obj.keyframe_insert(path, frame=1)
    ad = obj.animation_data
    try:  # Blender 4.4+ (slotted actions)
        fcs = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot).fcurves
    except Exception:  # older Blender
        fcs = ad.action.fcurves
    for ax in range(3):
        fc = fcs.find(path, index=ax)
        fc.keyframe_points.clear(); fc.keyframe_points.add(N)
        co = []
        for f in range(N): co += [f + 1, pts[f][ax]]
        fc.keyframe_points.foreach_set('co', co)
        fc.keyframe_points.foreach_set('interpolation', [1] * N)
        fc.update()

E = {}
for name, pts in T.items():
    e = bpy.data.objects.new('TGT-' + name, None); e.empty_display_size = 0.015
    tcoll.objects.link(e); set_curve(e, 'location', pts); E[name] = e

# ---------- 2. MOCAP skeleton
spine = sorted([r for r in BM if r.startswith('spine')], key=lambda s: int(s[5:]))
MN = {'hips': 'Hips', 'neck': 'Neck', 'head': 'Head'}
for k, r in enumerate(spine): MN[r] = f'Spine{k}'
for s in 'LR':
    MN.update({f'clav.{s}': f'Shoulder.{s}', f'upper.{s}': f'UpperArm.{s}', f'fore.{s}': f'Forearm.{s}', f'hand.{s}': f'Hand.{s}',
               f'thigh.{s}': f'Thigh.{s}', f'shin.{s}': f'Shin.{s}', f'foot.{s}': f'Foot.{s}'})
PAR = {'neck': spine[-1], 'head': 'neck'}
for k, r in enumerate(spine): PAR[r] = 'hips' if k == 0 else spine[k - 1]
for s in 'LR':
    clav = f'clav.{s}' if f'clav.{s}' in BM else None
    if clav: PAR[clav] = spine[-1]
    PAR.update({f'upper.{s}': clav or spine[-1], f'fore.{s}': f'upper.{s}', f'hand.{s}': f'fore.{s}',
                f'thigh.{s}': 'hips', f'shin.{s}': f'thigh.{s}', f'foot.{s}': f'shin.{s}'})

arm = bpy.data.armatures.new('MOCAP_Skeleton'); mc = bpy.data.objects.new('MOCAP_Skeleton', arm)
coll.objects.link(mc); mc.matrix_world = rig.matrix_world
mc.show_in_front = True
for o in bpy.context.selected_objects: o.select_set(False)
bpy.context.view_layer.objects.active = mc; mc.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
for r, vb in BM.items():
    b = rig.data.bones[vb]; eb = arm.edit_bones.new(MN[r])
    eb.head = b.head_local; eb.tail = b.tail_local; eb.align_roll(b.matrix_local.col[2].to_3d())
for r in BM:
    if r in PAR: arm.edit_bones[MN[r]].parent = arm.edit_bones[MN[PAR[r]]]
bpy.ops.object.mode_set(mode='OBJECT')

def axis_to(bone, v):
    m = bone.matrix_local.to_3x3()
    c = {'TRACK_X': m.col[0], 'TRACK_NEGATIVE_X': -m.col[0], 'TRACK_Z': m.col[2], 'TRACK_NEGATIVE_Z': -m.col[2]}
    return max(c, key=lambda k: c[k].dot(Vector(v)))
def damped(r, t):
    c = mc.pose.bones[MN[r]].constraints.new('DAMPED_TRACK'); c.target = E[t]
def locked(r, t):
    c = mc.pose.bones[MN[r]].constraints.new('LOCKED_TRACK'); c.target = E[t]
    c.lock_axis = 'LOCK_Y'; c.track_axis = axis_to(arm.bones[MN[r]], (1, 0, 0))

c = mc.pose.bones['Hips'].constraints.new('COPY_LOCATION'); c.target = E['hips']
damped('hips', 'hips_up'); locked('hips', 'hips_left')
for r in spine: damped(r, r)
locked(spine[-1], 'chest_left')
damped('neck', 'neck'); damped('head', 'head'); locked('head', 'head_left')
for s in 'LR':
    if f'clav.{s}' in BM: damped(f'clav.{s}', f'clav.{s}')
    damped(f'upper.{s}', f'elbow.{s}'); damped(f'fore.{s}', f'wrist.{s}')
    damped(f'thigh.{s}', f'knee.{s}'); damped(f'shin.{s}', f'ankle.{s}'); damped(f'foot.{s}', f'toe.{s}')

bpy.ops.object.mode_set(mode='POSE')
for pb in mc.pose.bones:
    if hasattr(pb, 'select'): pb.select = True
    else: pb.bone.select = True
bpy.ops.nla.bake(frame_start=1, frame_end=N, only_selected=True, visual_keying=True,
                 clear_constraints=True, use_current_action=False, bake_types={'POSE'})
bpy.ops.object.mode_set(mode='OBJECT')
if mc.animation_data and mc.animation_data.action: mc.animation_data.action.name = 'MOCAP_dance'
for e in list(tcoll.objects): bpy.data.objects.remove(e, do_unlink=True)
bpy.data.collections.remove(tcoll)

# ---------- 3. character follows the skeleton
for bone, key, val in prof['props']:
    rig.pose.bones[bone][key] = val
# freeze the IK/FK switch drivers at their FK values so the result doesn't depend on
# driver evaluation in whichever Blender version opens the file
rig.update_tag(); bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get(); rig_ev = rig.evaluated_get(dg)
frozen = 0
if rig.animation_data:
    for fc in list(rig.animation_data.drivers):
        paths = [t.data_path for v in fc.driver.variables for t in v.targets if t.data_path]
        if not any(('ik_' in p or 'fk_hinge' in p) for p in paths): continue
        try:
            owner_path, prop = fc.data_path.rsplit('.', 1)
            ev_owner = rig_ev.path_resolve(owner_path); val = getattr(ev_owner, prop)
            if hasattr(val, '__len__'): val = val[fc.array_index]
            og_owner = rig.path_resolve(owner_path)
        except Exception:
            continue
        rig.driver_remove(fc.data_path, fc.array_index) if fc.array_index or not hasattr(getattr(og_owner, prop), '__len__') else rig.driver_remove(fc.data_path, fc.array_index)
        cur = getattr(og_owner, prop)
        if hasattr(cur, '__len__'):
            cur[fc.array_index] = val
        else:
            setattr(og_owner, prop, val)
        frozen += 1
print('frozen IK/FK drivers:', frozen)
if prof.get('hair_parent_bone'):
    bpy.context.view_layer.update()
    for o in bpy.data.objects:
        if o.parent == rig and o.name.endswith('.scalp') and o.parent_type == 'OBJECT':
            mw = o.matrix_world.copy()
            o.parent_type = 'BONE'; o.parent_bone = prof['hair_parent_bone']
            bpy.context.view_layer.update(); o.matrix_world = mw
for r, vb in BM.items():
    pb = rig.pose.bones[vb]
    if prof.get('lock_feet') and r.startswith('foot.'):
        # ankle locked: foot keeps its rest orientation (flat, pointing straight ahead)
        lk = bpy.data.objects.new('MOCAP_footlock_' + r[-1], None); coll.objects.link(lk)
        lk.matrix_world = rig.matrix_world @ rig.data.bones[vb].matrix_local
        lk.hide_render = True
        try: lk.hide_set(True)
        except Exception: pass
        c = pb.constraints.new('COPY_ROTATION'); c.name = 'MOCAP_rot'; c.target = lk
        c.target_space = 'WORLD'; c.owner_space = 'WORLD'
        continue
    if r == 'hips':
        c = pb.constraints.new('COPY_LOCATION'); c.name = 'MOCAP_loc'; c.target = mc; c.subtarget = MN[r]
    c = pb.constraints.new('COPY_ROTATION'); c.name = 'MOCAP_rot'; c.target = mc; c.subtarget = MN[r]
    c.target_space = 'WORLD'; c.owner_space = 'WORLD'
rootb = rig.pose.bones[prof['root']]
R3 = rig.data.bones[prof['root']].matrix_local.to_3x3().transposed()
hip_rest = rig.data.bones[BM['hips']].head_local
for f in range(N):
    rootb.location = R3 @ (Vector(T['hips'][f]) - hip_rest)
    rootb.keyframe_insert('location', frame=f + 1)

# hair that should hang with gravity (e.g. a ponytail modelled sticking straight out)
if prof.get('gravity'):
    g = bpy.data.objects.new('MOCAP_hair_gravity', None); coll.objects.link(g)
    g.parent = rig; g.parent_type = 'BONE'; g.parent_bone = prof['root']
    bpy.context.view_layer.update()
    g.matrix_world = rig.matrix_world @ __import__('mathutils').Matrix.Translation(prof['gravity']['target'])
    g.hide_render = True
    for bn, infl in prof['gravity']['bones']:
        c = rig.pose.bones[bn].constraints.new('DAMPED_TRACK'); c.name = 'MOCAP_gravity'
        c.target = g; c.influence = infl
    try: g.hide_set(True)
    except Exception: pass
# hide the helper skeleton (still evaluated, just not drawn/rendered)
mc.hide_render = True
try: mc.hide_set(True)
except Exception: pass

scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=out_path, relative_remap=True, compress=True)
print('saved', out_path)
