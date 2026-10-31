"""Animated GIF of the fused, retargeted 3D skeleton (front + side projections) — no source video needed.
python skeleton_gif.py targets.json out.gif"""
import sys, json, numpy as np
from PIL import Image, ImageDraw
tg = json.load(open(sys.argv[1])); T = {k: np.array(v) for k, v in tg['targets'].items()}
spine = sorted([k for k in T if k.startswith('spine')], key=lambda s: int(s[5:]))
top = spine[-1]
CH = [('hips', top), (top, 'neck'), ('neck', 'head')]
for s in 'LR':
    CH += [(top, f'elbow.{s}'), (f'elbow.{s}', f'wrist.{s}'), ('hips', f'knee.{s}'), (f'knee.{s}', f'ankle.{s}'), (f'ankle.{s}', f'toe.{s}')]
W, H = 260, 400
col = lambda a, b: (46, 160, 67) if '.L' in a + b else ((214, 69, 65) if '.R' in a + b else (40, 40, 40))
frames = []
for i in range(0, tg['frames'], 2):
    im = Image.new('RGB', (W * 2, H), (248, 248, 246)); d = ImageDraw.Draw(im)
    for p, (axis, title) in enumerate([(0, 'front'), (1, 'side')]):
        ox = p * W
        cy = T['hips'][i][1]
        pr = lambda q: (ox + W / 2 + (q[0] if axis == 0 else q[1] - cy) * 200, H - 30 - q[2] * 200)
        d.line([(ox, H - 30), (ox + W, H - 30)], fill=(200, 200, 200))
        for a, b in CH: d.line([pr(T[a][i]), pr(T[b][i])], fill=col(a, b), width=4)
        d.text((ox + 8, 8), f'{title}  t={i / tg["fps"]:.1f}s', fill=(0, 0, 0))
    frames.append(im)
frames[0].save(sys.argv[2], save_all=True, append_images=frames[1:], duration=int(2000 / tg['fps']), loop=0, optimize=True)
print('saved', sys.argv[2], len(frames), 'frames')
