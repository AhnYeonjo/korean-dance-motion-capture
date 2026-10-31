"""Fuse front + side 2D keypoints into a 3D motion retargeted to Victoria's proportions.

Coordinate convention (Blender): +X = character's left, +Y = backward, +Z = up (character faces -Y).
Front video: image x right = her left (+X), image y down = -Z.
Side video:  she faces image-left, so image x right = backward (+Y), image y down = -Z.
Output: mocap_targets.json  (per-frame target points for the Blender rig, front-video timeline)
"""
import json, numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d, median_filter
from scipy.spatial.transform import Rotation as Rot, Slerp

NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKN, RKN, LAN, RAN = range(17)
CONF = 0.3

def load(path):
    d = np.load(path)
    return d['kp'].astype(float), float(d['fps'])

def clean(kp, fps, win_s=0.3):
    """Mask low-confidence points, interpolate gaps, smooth."""
    T = len(kp); xy = kp[..., :2].copy(); c = kp[..., 2].copy()
    c = np.nan_to_num(c)
    xy[c < CONF] = np.nan
    t = np.arange(T)
    for j in range(17):
        for k in range(2):
            v = xy[:, j, k]; ok = ~np.isnan(v)
            if ok.sum() >= 2: xy[:, j, k] = np.interp(t, t[ok], v[ok])
            else: xy[:, j, k] = np.nanmean(xy[:, :, k])
    w = int(win_s * fps) | 1
    xy = savgol_filter(median_filter(xy, size=(5, 1, 1)), w, 2, axis=0)
    return xy, c

def normalize(xy, fps):
    """Hip-centred coordinates in 'torso length' units, scale smoothed over time."""
    hc = (xy[:, LHIP] + xy[:, RHIP]) / 2; sc = (xy[:, LSH] + xy[:, RSH]) / 2
    torso = np.linalg.norm(sc - hc, axis=1)
    # rolling high percentile -> robust to forward-lean foreshortening
    w = int(4 * fps)
    pad = np.pad(torso, (w // 2, w // 2), mode='edge')
    s = np.array([np.percentile(pad[i:i + w], 80) for i in range(len(torso))])
    s = uniform_filter1d(s, int(fps))
    u = (xy - hc[:, None]) / s[:, None, None]
    return u, hc / s[:, None], s

def sync_features(u):
    """View-independent features: vertical positions (L/R-symmetric)."""
    y = u[..., 1]
    f = [np.minimum(y[:, LWR], y[:, RWR]), np.maximum(y[:, LWR], y[:, RWR]),
         np.minimum(y[:, LEL], y[:, REL]), np.maximum(y[:, LEL], y[:, REL]),
         (y[:, LKN] + y[:, RKN]) / 2, (y[:, LAN] + y[:, RAN]) / 2, y[:, NOSE]]
    return np.stack(f, 1)

def dtw_path(A, B, band):
    n, m = len(A), len(B)
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0
    cost = np.linalg.norm(A[:, None] - B[None], axis=2)
    for i in range(1, n + 1):
        lo = max(1, int(i * m / n) - band); hi = min(m, int(i * m / n) + band)
        for j in range(lo, hi + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j = n, m; path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        if k == 0: i, j = i - 1, j - 1
        elif k == 1: i -= 1
        else: j -= 1
        path.append((i - 1, j - 1))
    return path[::-1], cost

def unit(v): return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)

def frame_from(left, up):
    """Rotation whose columns are (left, back, up) — rest frame is identity."""
    z = unit(up); x = unit(left - (left * z).sum(-1, keepdims=True) * z); y = np.cross(z, x)
    return Rot.from_matrix(np.stack([x, y, z], -1))

def main():
    kf, fps_f = load('front.npz'); ks, fps_s = load('side.npz')
    xf, cf = clean(kf, fps_f); xs, cs = clean(ks, fps_s)
    uf, hf, sf = normalize(xf, fps_f); us, hs, ss = normalize(xs, fps_s)
    Tf = len(uf)
    tf = np.arange(Tf) / fps_f

    # ---- resample side to front frame rate for syncing
    ts = np.arange(len(us)) / fps_s
    tgrid = np.arange(0, ts[-1], 1 / fps_f)
    def resamp(a, tsrc, tdst):
        flat = a.reshape(len(a), -1)
        return np.stack([np.interp(tdst, tsrc, flat[:, i]) for i in range(flat.shape[1])], 1).reshape((len(tdst),) + a.shape[1:])
    us_r = resamp(us, ts, tgrid)

    # ---- temporal alignment (front timeline is master)
    Ff = sync_features(uf); Fs = sync_features(us_r)
    # coarse lag by cross-correlation of the feature sequences
    best = None
    for lag in range(-int(4 * fps_f), int(4 * fps_f) + 1):
        i0 = max(0, -lag); i1 = min(Tf, len(Fs) - lag)
        if i1 - i0 < Tf * 0.5: continue
        e = np.mean(np.linalg.norm(Ff[i0:i1] - Fs[i0 + lag:i1 + lag], axis=1))
        if best is None or e < best[0]: best = (e, lag)
    print('coarse lag (s):', best[1] / fps_f, 'err', best[0])
    path, cost = dtw_path(Ff, Fs, band=int(3 * fps_f))
    pi = np.array(path)
    # map each front frame -> mean side index, then smooth & force monotone
    m = np.array([pi[pi[:, 0] == i, 1].mean() for i in range(Tf)])
    m = savgol_filter(m, int(1.0 * fps_f) | 1, 2)
    m = np.maximum.accumulate(np.clip(m, 0, len(Fs) - 1))
    tau = m / fps_f  # side time (s) for each front frame
    e_dtw = np.mean([cost[i, int(round(m[i]))] for i in range(Tf)])
    print('dtw mean err', e_dtw, 'side time range', tau[0], tau[-1])
    us_a = resamp(us, ts, tau); hs_a = resamp(hs, ts, tau); cs_a = resamp(np.nan_to_num(cs), ts, tau)

    # ---- resolve left/right labels in the side view (Viterbi, 2 states per limb pair)
    def swap_states(pairs):
        cost = np.zeros((Tf, 2))
        for (l, r) in pairs:
            cost[:, 0] += np.abs(uf[:, l, 1] - us_a[:, l, 1]) + np.abs(uf[:, r, 1] - us_a[:, r, 1])
            cost[:, 1] += np.abs(uf[:, l, 1] - us_a[:, r, 1]) + np.abs(uf[:, r, 1] - us_a[:, l, 1])
        pen = 0.6
        acc = cost[0].copy(); back = np.zeros((Tf, 2), int)
        for t in range(1, Tf):
            new = np.empty(2)
            for s in range(2):
                opts = [acc[0] + (pen if s != 0 else 0), acc[1] + (pen if s != 1 else 0)]
                back[t, s] = int(np.argmin(opts)); new[s] = min(opts) + cost[t, s]
            acc = new
        st = np.zeros(Tf, int); st[-1] = int(np.argmin(acc))
        for t in range(Tf - 1, 0, -1): st[t - 1] = back[t, st[t]]
        return st
    # arms: in the side view the camera-side (left) arm is always clearly visible and labelled
    # correctly, and both wrists hang at the same height during arm swings, so height matching
    # can't tell them apart -> trust the detector's labels for arms
    sw_arm = np.zeros(Tf, int)
    sw_leg = swap_states([(LKN, RKN), (LAN, RAN)])
    print('side L/R swapped frames: arms', sw_arm.mean().round(2), 'legs', sw_leg.mean().round(2))
    us_f = us_a.copy(); cs_f = cs_a.copy()
    for st, pairs in [(sw_arm, [(LSH, RSH), (LEL, REL), (LWR, RWR)]), (sw_leg, [(LKN, RKN), (LAN, RAN)])]:
        for (l, r) in pairs:
            idx = st == 1
            us_f[idx, l], us_f[idx, r] = us_a[idx, r], us_a[idx, l]
            cs_f[idx, l], cs_f[idx, r] = cs_a[idx, r], cs_a[idx, l]

    # ---- 3D fusion (torso-length units, hip-centred)
    P = np.zeros((Tf, 17, 3))
    P[..., 0] = uf[..., 0]                      # lateral from front
    P[..., 1] = us_f[..., 0]                    # depth from side
    P[..., 2] = -(0.5 * uf[..., 1] + 0.5 * us_f[..., 1])  # height from both
    # far-side joints poorly seen in side view -> lean toward the opposite joint's depth
    # far-side joints poorly seen in the side view:
    #  arms -> mirror the visible arm's swing (alternating front/back), legs -> copy (crouch bends both)
    sh_d = {LEL: (LSH, RSH), REL: (RSH, LSH), LWR: (LSH, RSH), RWR: (RSH, LSH)}
    for (l, r) in [(LEL, REL), (LWR, RWR), (LKN, RKN), (LAN, RAN)]:
        for a, b in [(l, r), (r, l)]:
            w = np.clip((cs_f[:, a] - 0.2) / 0.4, 0, 1)
            if a in sh_d:
                sa, sb = sh_d[a]
                guess = P[:, sa, 1] - (P[:, b, 1] - P[:, sb, 1])
            else:
                guess = P[:, b, 1]
            P[:, a, 1] = w * P[:, a, 1] + (1 - w) * guess
    P = savgol_filter(P, int(0.45 * fps_f) | 1, 2, axis=0)  # stronger de-jitter

    np.savez('fused.npz', P=P, uf=uf, us=us_f, tau=tau, sw_arm=sw_arm, sw_leg=sw_leg, hf=hf, hs=hs_a, fps=fps_f)
    print('saved fused.npz', Tf, 'frames')

if __name__ == '__main__':
    main()
