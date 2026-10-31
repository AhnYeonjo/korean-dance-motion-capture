"""2D pose detection (YOLO11x-pose, 17 COCO keypoints) on every frame of a video.
Usage: python detect.py video.mp4 out.npz
Saves kp (T,17,3: x_px, y_px, conf), fps, w, h. Picks the largest person per frame."""
import sys, time, cv2, numpy as np
from ultralytics import YOLO

vid, out = sys.argv[1], sys.argv[2]
model = YOLO('yolo11x-pose.pt')
cap = cv2.VideoCapture(vid)
fps = cap.get(cv2.CAP_PROP_FPS)
kps = []
t0 = time.time(); i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    r = model.predict(f, imgsz=640, verbose=False)[0]
    if r.keypoints is not None and len(r.boxes):
        b = r.boxes.xyxy.cpu().numpy()
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        j = int(area.argmax())
        xy = r.keypoints.xy.cpu().numpy()[j]; c = r.keypoints.conf.cpu().numpy()[j]
        kps.append(np.concatenate([xy, c[:, None]], 1))
    else:
        kps.append(np.full((17, 3), np.nan))
    i += 1
    if i % 50 == 0: print(vid, i, f'{time.time()-t0:.0f}s', flush=True)
np.savez(out, kp=np.array(kps), fps=fps, w=int(cap.get(3)), h=int(cap.get(4)))
print('done', vid, len(kps), time.time() - t0)
