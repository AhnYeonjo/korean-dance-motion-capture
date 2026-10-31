import bpy, sys
from mathutils import Vector
path, outdir, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
bpy.ops.wm.open_mainfile(filepath=path)
sc = bpy.context.scene
for o in sc.objects:
    for m in o.modifiers:
        if m.type == 'SUBSURF': m.show_render = False
sc.render.engine = 'BLENDER_WORKBENCH'
sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'MATERIAL'
sc.render.resolution_x = 360; sc.render.resolution_y = 560
cams = {}
for name, loc in [('front', (0, -4.6, 0.9)), ('side', (4.6, -0.4, 0.9))]:
    cd = bpy.data.cameras.new(name); cd.lens = 50
    co = bpy.data.objects.new(name, cd); sc.collection.objects.link(co); co.location = loc
    co.rotation_euler = (Vector((0, -0.4 if name == 'side' else 0, 0.8)) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    cams[name] = co
for f in range(1, sc.frame_end + 1, step):
    sc.frame_set(f)
    for n, co in cams.items():
        sc.camera = co; sc.render.filepath = f'{outdir}/{n}_{f:04d}.png'
        bpy.ops.render.render(write_still=True)
print('done')
