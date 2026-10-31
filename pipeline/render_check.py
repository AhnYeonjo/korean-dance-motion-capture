"""Render front & side views of the animated character at given frames.
python render_check.py file.blend outprefix frame1 frame2 ..."""
import bpy, sys, math
from mathutils import Vector
path, prefix = sys.argv[1], sys.argv[2]; frames = [int(f) for f in sys.argv[3:]]
bpy.ops.wm.open_mainfile(filepath=path)
sc = bpy.context.scene
rig = next(o for o in bpy.data.objects if o.type == 'ARMATURE' and o.name.startswith('RIG'))
try: [pb.__setitem__('Quality', 0) for pb in rig.pose.bones if 'Quality' in pb.keys()]
except Exception: pass
for o in sc.objects:
    for m in o.modifiers:
        if m.type == 'SUBSURF': m.show_render = False
sc.render.engine = 'BLENDER_WORKBENCH'
sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'MATERIAL'
sc.render.resolution_x = 540; sc.render.resolution_y = 960; sc.render.resolution_percentage = 100
sc.render.film_transparent = False
cams = {}
for name, loc in [('front', (0, -4.2, 0.85)), ('side', (4.2, 0, 0.85))]:
    cd = bpy.data.cameras.new('CHK_' + name); cd.lens = 50
    co = bpy.data.objects.new('CHK_' + name, cd); sc.collection.objects.link(co)
    co.location = loc
    d = Vector((0, 0, 0.75)) - Vector(loc); co.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    cams[name] = co
for f in frames:
    sc.frame_set(f)
    for name, co in cams.items():
        sc.camera = co
        sc.render.filepath = f'{prefix}_{name}_{f:04d}.png'
        bpy.ops.render.render(write_still=True)
print('ok')
