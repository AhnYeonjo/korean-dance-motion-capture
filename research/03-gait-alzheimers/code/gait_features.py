"""Gait biomarkers from 2D pose keypoints.

Implements the biomarkers defined in the gait-analysis paper (step time variability, stride length,
cadence, left/right asymmetry, postural sway, arm swing) for keypoint time series from either
YOLO/COCO-17 or MediaPipe-33 pose estimators.

    from gait_features import gait_features
    feats = gait_features(kp, fps=30, layout='coco17', view='side')

kp: array (T, J, 2) or (T, J, 3) in image pixels (x right, y down).
view: 'side' (walking across the frame) or 'front' (walking toward / away from the camera).
"""
import numpy as np
from scipy.signal import find_peaks, savgol_filter

LAYOUTS = {  # name -> index
    'coco17': dict(l_sh=5, r_sh=6, l_wr=9, r_wr=10, l_hip=11, r_hip=12, l_ank=15, r_ank=16),
    'mediapipe33': dict(l_sh=11, r_sh=12, l_wr=15, r_wr=16, l_hip=23, r_hip=24, l_ank=27, r_ank=28),
}


def _smooth(x, fps, win_s=0.2):
    w = max(5, int(win_s * fps) | 1)
    return savgol_filter(x, w, 2, axis=0) if len(x) > w else x


def gait_features(kp, fps, layout='coco17', view='side'):
    J = LAYOUTS[layout]
    p = _smooth(np.asarray(kp, float)[..., :2], fps)
    g = lambda k: p[:, J[k]]
    hip = (g('l_hip') + g('r_hip')) / 2
    sh = (g('l_sh') + g('r_sh')) / 2
    scale = np.median(np.linalg.norm(sh - hip, axis=1))      # torso length -> body-size normalisation
    walk_ax = 0 if view == 'side' else 1                      # axis along the walking direction in the image
    lat_ax = 1 - walk_ax if view == 'front' else None

    # step events = peaks of the (signed) distance between the ankles along the walking direction
    sep = (g('l_ank')[:, walk_ax] - g('r_ank')[:, walk_ax]) / scale
    pk_l, _ = find_peaks(sep, distance=int(0.3 * fps), prominence=0.1)     # left foot leading
    pk_r, _ = find_peaks(-sep, distance=int(0.3 * fps), prominence=0.1)    # right foot leading
    steps = np.sort(np.concatenate([pk_l, pk_r]))
    out = {'n_steps': int(len(steps))}
    if len(steps) >= 4:
        st = np.diff(steps) / fps
        out['cadence_steps_per_min'] = 60.0 / st.mean()
        out['step_time_cv'] = st.std() / st.mean()                          # step time variability
        side = np.isin(steps[1:], pk_l)                                     # which foot ended each step
        if side.any() and (~side).any():
            tl, tr = st[side].mean(), st[~side].mean()
            out['step_time_asymmetry'] = abs(tl - tr) / ((tl + tr) / 2)
        amp = np.abs(sep[steps])
        out['stride_length_norm'] = amp.mean()                              # in torso lengths
        ampl, ampr = np.abs(sep[pk_l]), np.abs(sep[pk_r])
        if len(ampl) and len(ampr):
            out['step_length_asymmetry'] = abs(ampl.mean() - ampr.mean()) / ((ampl.mean() + ampr.mean()) / 2)

    # postural sway: lateral oscillation of the hip centre after removing the walking trend
    if lat_ax is not None:
        x = hip[:, 0] / scale
        trend = savgol_filter(x, max(5, int(2 * fps) | 1), 1) if len(x) > 2 * fps else x.mean()
        out['sway_amplitude_norm'] = float(np.std(x - trend) * 2)

    # arm swing: wrist motion relative to the shoulder along the walking direction
    for s in 'lr':
        rel = (g(f'{s}_wr')[:, walk_ax] - g(f'{s}_sh')[:, walk_ax]) / scale
        out[f'arm_swing_{s}'] = float(np.percentile(rel, 95) - np.percentile(rel, 5))
    out['arm_swing_asymmetry'] = abs(out['arm_swing_l'] - out['arm_swing_r']) / (
        (out['arm_swing_l'] + out['arm_swing_r']) / 2 + 1e-9)
    return {k: float(v) for k, v in out.items()}


def synthetic_walker(fps=30, secs=10, cadence=110, stride=0.9, asym=0.0, jitter=0.0, arm=0.6, seed=0):
    """Side-view COCO-17 stick figure walking in place (for tests / demos). asym slows the left step."""
    rng = np.random.default_rng(seed)
    T = int(fps * secs); kp = np.zeros((T, 17, 2))
    phase = np.zeros(T); f0 = cadence / 120.0                   # stride cycles per second
    for t in range(1, T):
        f = f0 * (1 + jitter * rng.standard_normal())
        phase[t] = phase[t - 1] + 2 * np.pi * f / fps
    ph = phase + asym * np.sin(phase)                             # warps timing of one half-cycle
    torso = 100.0
    hip = np.array([300.0, 300.0]); shp = hip + [0, -torso]
    kp[:, 11] = kp[:, 12] = hip; kp[:, 5] = kp[:, 6] = shp
    kp[:, 15, 0] = hip[0] + stride / 2 * torso * np.sin(ph); kp[:, 16, 0] = hip[0] - stride / 2 * torso * np.sin(ph)
    kp[:, 15, 1] = kp[:, 16, 1] = hip[1] + 1.8 * torso
    kp[:, 9, 0] = shp[0] - arm / 2 * torso * np.sin(ph); kp[:, 10, 0] = shp[0] + arm / 2 * torso * np.sin(ph)
    kp[:, 9, 1] = kp[:, 10, 1] = shp[1] + 1.2 * torso
    return kp


if __name__ == '__main__':
    import json
    print('normal :', json.dumps({k: round(v, 3) for k, v in gait_features(synthetic_walker(), 30).items()}))
    print('impaired:', json.dumps({k: round(v, 3) for k, v in gait_features(
        synthetic_walker(stride=0.6, asym=0.25, jitter=0.08, arm=0.3), 30).items()}))
