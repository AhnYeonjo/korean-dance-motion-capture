"""Turn the fused 3D motion (fused.npz) into per-frame target points sized to a character.
Usage: python retarget.py <rest.json> <out_targets.json>
rest.json: role -> {head, tail} in rig space (exported by export_rest.py). Roles:
  hips, spine0..spineK, neck, head, [clav.L/R], upper.L/R, fore.L/R, thigh.L/R, shin.L/R, foot.L/R
"""
import sys, json, numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot, Slerp
from fuse import unit, frame_from, NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKN, RKN, LAN, RAN

rest_path, out_path = sys.argv[1], sys.argv[2]
FIXED_HEAD = '--fixed-head' in sys.argv
d = np.load('fused.npz'); P = d['P']; hf = d['hf']; hs = d['hs']; tau = d['tau']; fps = float(d['fps'])
Tf = len(P)
R = {k: (np.array(v['head']), np.array(v['tail'])) for k, v in json.load(open(rest_path)).items()}
L = {k: np.linalg.norm(t - h) for k, (h, t) in R.items()}
rdir = {k: unit(t - h) for k, (h, t) in R.items()}
spine = sorted([k for k in R if k.startswith('spine')], key=lambda s: int(s[5:]))
has_clav = 'clav.L' in R

hipc = (P[:, LHIP] + P[:, RHIP]) / 2; shc = (P[:, LSH] + P[:, RSH]) / 2
up = unit(shc - hipc)
R_hip = frame_from(P[:, LHIP] - P[:, RHIP], up)
R_chest = frame_from(P[:, LSH] - P[:, RSH], up)
earc = (P[:, LEAR] + P[:, REAR]) / 2
fwd_h = unit(P[:, NOSE] - earc); left_h = unit((P[:, LEAR] - P[:, REAR]) + (P[:, LEYE] - P[:, REYE]))
R_head = frame_from(left_h, np.cross(fwd_h, left_h))
n0 = int(1.0 * fps)
rel = lambda Rs: Rs * Rs[:n0].mean().inv()
R_hip, R_chest, R_head = rel(R_hip), rel(R_chest), rel(R_head)
FACE_FRONT = '--face-front' in sys.argv
def no_yaw(Rs):  # keep lean/tilt, drop turning about the vertical axis
    e = Rs.as_euler('ZXY'); e[:, 0] = 0
    return Rot.from_euler('ZXY', e)
if FACE_FRONT:
    R_hip, R_chest = no_yaw(R_hip), no_yaw(R_chest)
LEGS_SAG = '--legs-sagittal' in sys.argv
if LEGS_SAG:  # pelvis stays level: keep only forward/back tilt
    e = R_hip.as_euler('XYZ'); e[:, 1:] = 0; R_hip = Rot.from_euler('XYZ', e)
def leg_dir(v, rest_d):
    v = unit(v)
    if not LEGS_SAG: return v
    x = rest_d[0]; yz = np.array([v[1], v[2]]); n = np.linalg.norm(yz) + 1e-9
    s = np.sqrt(max(1 - x * x, 0)) / n
    return np.array([x, v[1] * s, v[2] * s])   # bend only forward/back + up/down, no sideways

ARM_FIX = '--arm-clearance' in sys.argv
def arm_dir(v, s):
    # keep hanging arms clear of the (wider) character body and swing them further forward
    v = unit(v)
    if not ARM_FIX: return v
    w = max(0.0, -v[2])                      # 1 = arm straight down, 0 = horizontal or raised
    sign = 1.0 if s == 'L' else -1.0
    x, y, z = v
    y = y * (1 + 0.8 * w) - 0.18 * w          # bigger front/back swing + slight forward lift
    x = sign * max(sign * x, 0.36 * w)        # at least ~20 deg away from the body when hanging
    return unit(np.array([x, y, z]))
blend = lambda A, B, t: Rot.from_rotvec(np.array([Slerp([0, 1], Rot.concatenate([A[i], B[i]]))(t).as_rotvec() for i in range(Tf)]))
R_head = blend(R_chest, R_head, 0.3)   # face keypoints are noisy -> damp extra head motion
R_neck = blend(R_chest, R_head, 0.5)
if FIXED_HEAD:  # neck/head stay upright, looking straight ahead
    R_head = Rot.identity(Tf); R_neck = Rot.identity(Tf)

BODY = None
for a in sys.argv:
    if a.startswith('--body='): BODY = np.array(json.load(open(a[7:])))
ARM_PAD = 0.045  # arm thickness margin (m)
def push_out(pt, root, rot):
    """If pt is inside the character's torso/hip cross-section, push it out horizontally."""
    if BODY is None: return pt
    q = rot.inv().apply(pt - root) + hip0
    if not (0.6 <= q[2] <= 1.22): return pt
    rx = np.interp(q[2], BODY[:, 0], BODY[:, 1]) + ARM_PAD
    yc = np.interp(q[2], BODY[:, 0], BODY[:, 2])
    ry = np.interp(q[2], BODY[:, 0], BODY[:, 3]) + ARM_PAD
    dx, dy = q[0], q[1] - yc
    e = np.hypot(dx / rx, dy / ry)
    if e >= 1: return pt
    if e < 1e-6: dx, dy, e = 0.0, ry, 1.0
    q = np.array([dx / e, yc + dy / e, q[2]])
    return rot.apply(q - hip0) + root
def solve_seg(start, d, length, root, rot, iters=4):
    """Segment from start with direction d; nudge so its middle and end stay outside the body."""
    end = start + d * length
    for _ in range(iters):
        mid = start + (end - start) * 0.5
        m2 = push_out(mid, root, rot); e2 = push_out(end, root, rot)
        tgt = e2 + (m2 - mid) * 1.0
        end = start + unit(tgt - start) * length
    return end

out = {}
UH = {}
def put(k, i, v):
    out.setdefault(k, np.zeros((Tf, 3)))[i] = v
hip0 = R['hips'][0]
for i in range(Tf):
    rh, rc = R_hip[i], R_chest[i]
    p = hip0.copy()
    put('hips', i, p); put('hips_up', i, p + rh.apply(rdir['hips']) * 0.1); put('hips_left', i, p + rh.apply([1, 0, 0]) * 0.1)
    sl = Slerp([0, max(len(spine) - 1, 1)], Rot.concatenate([rh, rc]))
    prev_tail_rest, prev_tail = hip0, p
    heads = {}
    for k, b in enumerate(spine):
        rk = sl(k)
        h = prev_tail + rk.apply(R[b][0] - prev_tail_rest)
        heads[b] = h
        t = h + rk.apply(rdir[b]) * L[b]
        put(b, i, t); prev_tail_rest, prev_tail = R[b][1], t
    chest = spine[-1]
    put('chest_left', i, prev_tail + rc.apply([1, 0, 0]) * 0.1)
    nh = prev_tail + rc.apply(R['neck'][0] - R[chest][1])
    nt = nh + R_neck[i].apply(rdir['neck']) * L['neck']; put('neck', i, nt)
    hh = nt + R_neck[i].apply(R['head'][0] - R['neck'][1])
    put('head', i, hh + R_head[i].apply(rdir['head']) * L['head'])
    put('head_left', i, hh + R_head[i].apply([1, 0, 0]) * 0.1)
    for s, (sh, el, wr) in {'L': (LSH, LEL, LWR), 'R': (RSH, REL, RWR)}.items():
        if has_clav:
            ch = heads[chest] + rc.apply(R[f'clav.{s}'][0] - R[chest][0])
            ce = ch + rc.apply(rdir[f'clav.{s}']) * L[f'clav.{s}']; put(f'clav.{s}', i, ce)
            uh = ce + rc.apply(R[f'upper.{s}'][0] - R[f'clav.{s}'][1])
        else:
            uh = heads[chest] + rc.apply(R[f'upper.{s}'][0] - R[chest][0])
        elbow = solve_seg(uh, arm_dir(P[i, el] - P[i, sh], s), L[f'upper.{s}'], p, rh)
        wrist = solve_seg(elbow, arm_dir(P[i, wr] - P[i, el], s), L[f'fore.{s}'], p, rh)
        put(f'elbow.{s}', i, elbow); put(f'wrist.{s}', i, wrist)
        UH.setdefault(s, np.zeros((Tf, 3)))[i] = uh
    yl = rh.apply([1, 0, 0]); yl[2] = 0; R_yaw = frame_from(unit(yl), np.array([0, 0, 1.0]))
    for s, (hp, kn, an) in {'L': (LHIP, LKN, LAN), 'R': (RHIP, RKN, RAN)}.items():
        th = p + rh.apply(R[f'thigh.{s}'][0] - hip0)
        knee = th + leg_dir(P[i, kn] - P[i, hp], rdir[f'thigh.{s}']) * L[f'thigh.{s}']
        ank = knee + leg_dir(P[i, an] - P[i, kn], rdir[f'shin.{s}']) * L[f'shin.{s}']
        fh = ank + R_yaw.apply(R[f'foot.{s}'][0] - R[f'shin.{s}'][1])
        put(f'knee.{s}', i, knee); put(f'ankle.{s}', i, ank)
        put(f'toe.{s}', i, fh + R_yaw.apply(R[f'foot.{s}'][1] - R[f'foot.{s}'][0]))

# opening section: scale the sideways arm lift so its peak reaches the requested angle (left arm;
# the right arm follows through the mirroring below)
RAISE = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('--swing-raise=')), None)
if RAISE is not None:
    zl = P[:, LWR, 2] - P[:, LSH, 2]; zr = P[:, RWR, 2] - P[:, RSH, 2]
    first_raise = int(np.argmax((zl > -0.45) & (zr > -0.45)))
    def abd(i):
        v = R_hip[i].inv().apply(out['elbow.L'][i] - UH['L'][i]); return np.arctan2(v[0], -v[2])
    A = np.array([abd(i) for i in range(first_raise)])
    from scipy.signal import find_peaks
    lim = max(1, first_raise - int(1.2 * fps))           # ignore the rise into the arms-up section
    pk, _ = find_peaks(A[:lim], prominence=np.radians(8))
    peak = np.median(A[pk]) if len(pk) else A[:lim].max()
    k = np.radians(RAISE) / peak
    ramp = np.clip((lim - np.arange(first_raise)) / (0.4 * fps), 0, 1)  # fade out before the raised section
    for i in range(lim):
        if A[i] <= 0: continue
        dlt = A[i] * (k - 1) * ramp[i]
        rh = R_hip[i]
        for j in ['elbow', 'wrist']:
            v = rh.inv().apply(out[f'{j}.L'][i] - UH['L'][i])
            a = np.arctan2(v[0], -v[2]) + dlt; rad = np.hypot(v[0], v[2])
            v = np.array([rad * np.sin(a), v[1], -rad * np.cos(a)])
            out[f'{j}.L'][i] = UH['L'][i] + rh.apply(v)
    print(f'swing lift: peaks {np.degrees(A[pk]).round(1) if len(pk) else []} median {np.degrees(peak):.1f} deg -> {RAISE} deg (x{k:.2f})')

# opening arm-swing section: right arm = left arm mirrored left/right with the front/back swing
# flipped (natural alternating swing), measured relative to each shoulder
if '--swing-from-left' in sys.argv:
    zl = P[:, LWR, 2] - P[:, LSH, 2]; zr = P[:, RWR, 2] - P[:, RSH, 2]
    first_raise = int(np.argmax((zl > -0.45) & (zr > -0.45)))
    k = int(0.4 * fps) | 1
    sw = (np.arange(Tf) < first_raise).astype(float)
    w = np.convolve(np.pad(sw, k // 2, mode='edge'), np.ones(k) / k, mode='valid')
    fl = np.array([-1.0, -1.0, 1.0])
    for i in np.where(w > 0)[0]:
        rh = R_hip[i]
        new = {}
        for j in ['elbow', 'wrist']:
            v = rh.inv().apply(out[f'{j}.L'][i] - UH['L'][i]) * fl
            new[j] = UH['R'][i] + rh.apply(v)
        e = solve_seg(UH['R'][i], unit(new['elbow'] - UH['R'][i]), L['upper.R'], hip0, rh)
        wr = solve_seg(e, unit(new['wrist'] - new['elbow']), L['fore.R'], hip0, rh)
        out['elbow.R'][i] = (1 - w[i]) * out['elbow.R'][i] + w[i] * e
        out['wrist.R'][i] = (1 - w[i]) * out['wrist.R'][i] + w[i] * wr
    print('swing frames from left arm:', int((w > 0.99).sum()))

# both-arms-raised section: make one arm an exact mirror of the other (default: copy the LEFT arm)
if any(a.startswith('--mirror-raised') for a in sys.argv):
    SRC = 'R' if '--mirror-raised=R' in sys.argv else 'L'   # source arm (character's side)
    DST = 'L' if SRC == 'R' else 'R'
    zl = P[:, LWR, 2] - P[:, LSH, 2]; zr = P[:, RWR, 2] - P[:, RSH, 2]
    raised = ((zl > -0.45) & (zr > -0.45)).astype(float)
    k = int(0.4 * fps) | 1
    w = np.convolve(np.pad(raised, k // 2, mode='edge'), np.ones(k) / k, mode='valid')  # soft blend in/out
    flip = np.array([-1.0, 1.0, 1.0])
    for i in np.where(w > 0)[0]:
        rh = R_hip[i]
        for j in ['elbow', 'wrist']:
            mr = rh.apply(rh.inv().apply(out[f'{j}.{SRC}'][i] - hip0) * flip) + hip0
            out[f'{j}.{DST}'][i] = (1 - w[i]) * out[f'{j}.{DST}'][i] + w[i] * mr
    print('mirrored frames:', int((w > 0.99).sum()))

# anatomical clean-up: no elbow bending backwards, limited backward reach of the upper arm
if '--arm-limits' in sys.argv:
    MAX_BACK = np.sin(np.radians(30))
    for i in range(Tf):
        rh = R_hip[i]; fwd = rh.apply([0, -1.0, 0])
        for s in 'LR':
            uh = UH[s][i]
            u = unit(out[f'elbow.{s}'][i] - uh)
            back = -u @ fwd
            if u[2] < 0 and back > MAX_BACK:            # reaching too far behind -> swing it forward
                horiz = u - (u @ fwd) * fwd
                u = unit(horiz * np.sqrt(1 - MAX_BACK**2) / max(np.linalg.norm(horiz), 1e-6) - fwd * MAX_BACK)
            e = solve_seg(uh, u, L[f'upper.{s}'], hip0, rh)
            f = unit(out[f'wrist.{s}'][i] - out[f'elbow.{s}'][i]); u = unit(e - uh)
            bend = f - (f @ u) * u
            if bend @ fwd < 0:                          # elbow hyper-extended -> remove the backward bend
                f = unit(f - (bend @ fwd) * fwd)
            out[f'elbow.{s}'][i] = e
            out[f'wrist.{s}'][i] = solve_seg(e, f, L[f'fore.{s}'], hip0, rh)

# root translation: horizontal from image hip motion, vertical by keeping the lower ankle on its rest height
char_torso = np.linalg.norm((R['upper.L'][0] + R['upper.R'][0]) / 2 - hip0)
def trend(x, win_s=2.0, passes=2):
    # travel path = smooth trend line (repeated wide Savitzky-Golay, order 2)
    for _ in range(passes): x = savgol_filter(x, int(win_s * fps) | 1, 2, mode='nearest')
    return x
lat = trend((hf[:, 0] - hf[:n0, 0].mean()) * char_torso)
dep = trend((hs[:, 0] - hs[:n0, 0].mean()) * char_torso)
if FACE_FRONT:  # walk straight ahead only: no sideways drift, forward travel never reverses
    lat = np.zeros_like(lat)
    dep = trend(np.minimum.accumulate(dep), 1.0, 1)
FWD = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('--forward=')), None)
if FWD is not None:  # steady gradual travel forward (-Y) over the whole clip, eased at both ends
    # stand in place during the opening arm swing; start travelling when the arms go up
    zl = P[:, LWR, 2] - P[:, LSH, 2]; zr = P[:, RWR, 2] - P[:, RSH, 2]
    t0 = int(np.argmax((zl > -0.45) & (zr > -0.45)))
    u = np.clip((np.arange(Tf) - t0) / max(Tf - 1 - t0, 1), 0, 1); ease = u * u * (3 - 2 * u)
    dep = -FWD * (0.6 * u + 0.4 * ease)
    print(f'forward travel starts at {t0 / fps:.2f}s')
dz = savgol_filter(R['shin.L'][1][2] - np.minimum(out['ankle.L'][:, 2], out['ankle.R'][:, 2]), int(0.7 * fps) | 1, 2, mode='nearest')
shift = np.stack([lat, dep, dz], 1)
for k in out: out[k] = savgol_filter(out[k], int(0.3 * fps) | 1, 2, axis=0, mode='nearest') + shift
json.dump({'fps': fps, 'frames': Tf, 'targets': {k: np.round(v, 5).tolist() for k, v in out.items()},
           'sync_side_time': tau.round(4).tolist()}, open(out_path, 'w'))
print('saved', out_path, 'targets:', len(out), 'shift range', shift.min(0).round(3), shift.max(0).round(3))
