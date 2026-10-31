"""python export_rest.py char.blend profile.json out_rest.json  -> role:{head,tail} rest positions"""
import bpy, sys, json
src, prof, out = sys.argv[1:4]
bpy.ops.wm.open_mainfile(filepath=src)
p = json.load(open(prof)); rig = bpy.data.objects[p['rig']]
json.dump({r: {'head': list(rig.matrix_world @ rig.data.bones[b].head_local), 'tail': list(rig.matrix_world @ rig.data.bones[b].tail_local)}
           for r, b in p['bones'].items()}, open(out, 'w'), indent=1)
