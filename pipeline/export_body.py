"""python export_body.py char.blend out_body.json OBJ1 OBJ2 ... -> torso cross-section ellipses per height (rest pose)"""
import bpy, sys, json, numpy as np
src, out = sys.argv[1], sys.argv[2]; names = sys.argv[3:]
bpy.ops.wm.open_mainfile(filepath=src)
V = []
for n in names:
    o = bpy.data.objects[n]; mw = np.array(o.matrix_world)
    co = np.array([v.co for v in o.data.vertices]); co = co @ mw[:3, :3].T + mw[:3, 3]; V.append(co)
V = np.concatenate(V)
sh = None
rows = []
for z in np.arange(0.55, 1.45, 0.02):
    s = V[(np.abs(V[:, 2] - z) < 0.012) & (np.abs(V[:, 0]) < 0.17)]
    if len(s) < 10: continue
    yc = (s[:, 1].max() + s[:, 1].min()) / 2
    rows.append([float(z), float(np.abs(s[:, 0]).max()), float(yc), float((s[:, 1].max() - s[:, 1].min()) / 2)])
json.dump(rows, open(out, 'w')); print(len(rows), 'slices'); [print(np.round(r, 3)) for r in rows[::5]]
