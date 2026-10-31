"""Crop a reference clip to the dancer, blur everyone else (segmentation masks), mute, encode.
python ref_clip.py in.mp4 out.mp4"""
import cv2, sys, numpy as np, subprocess
from ultralytics import YOLO
src, dst = sys.argv[1], sys.argv[2]
m = YOLO('/home/claude/mt/yolo11m-seg.pt')
cap = cv2.VideoCapture(src); fps = cap.get(5); W, H = int(cap.get(3)), int(cap.get(4))
frames, dboxes, others = [], [], []
prev = None
while True:
    ok, f = cap.read()
    if not ok: break
    r = m.predict(f, imgsz=960, classes=[0], verbose=False, retina_masks=True)[0]
    om = None; db = None
    if r.masks is not None and len(r.boxes):
        b = r.boxes.xyxy.cpu().numpy(); ms = r.masks.data.cpu().numpy() > 0.5
        c = (b[:, :2] + b[:, 2:]) / 2
        j = int(np.argmin(np.abs(c[:, 0] - W / 2))) if prev is None else int(np.argmin(np.linalg.norm(c - prev, axis=1)))
        prev = c[j]; db = b[j]
        dancer = cv2.dilate(ms[j].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        o = np.zeros((H, W), bool)
        for k in range(len(b)):
            if k != j: o |= ms[k]
        if o.any():
            om = cv2.dilate(o.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool) & ~dancer
    if om is not None:
        bl = cv2.GaussianBlur(f, (0, 0), 20); f[om] = bl[om]
    frames.append(f); dboxes.append(db)
db = np.array([d for d in dboxes if d is not None])
pad = 0.12 * (db[:, 3] - db[:, 1]).max()
x0, y0 = max(0, int(db[:, 0].min() - pad)), max(0, int(db[:, 1].min() - pad))
x1, y1 = min(W, int(db[:, 2].max() + pad)), min(H, int(db[:, 3].max() + pad))
x1 -= (x1 - x0) % 2; y1 -= (y1 - y0) % 2
tmp = dst + '.raw.mp4'
out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (x1 - x0, y1 - y0))
for f in frames: out.write(f[y0:y1, x0:x1])
out.release()
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', tmp, '-an', '-vf', 'scale=640:-2', '-c:v', 'libx264', '-preset', 'veryfast',
                '-crf', '24', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', dst], check=True)
print('frames', len(frames), 'crop', (x0, y0, x1, y1))
