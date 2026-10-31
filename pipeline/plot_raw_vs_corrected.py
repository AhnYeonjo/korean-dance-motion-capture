"""Raw detections vs. corrected motion, from the real recording.
Run from data/:  python ../pipeline/plot_raw_vs_corrected.py
Writes ../media/raw_vs_corrected.png and prints jitter numbers."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fuse import clean, normalize, LSH, RSH, LWR, RWR, LHIP, RHIP

kf = np.load('front.npz'); ks = np.load('side.npz'); fu = np.load('fused.npz')
tg = json.load(open('targets_rain.json')); T = {k: np.array(v) for k, v in tg['targets'].items()}
fps_f, fps_s = float(kf['fps']), float(ks['fps'])
tau = fu['tau']; N = len(tau); t = np.arange(N) / fps_f

def raw_norm(kp, fps):
    """Raw keypoints in hip-centred torso units, no smoothing; low-confidence points -> NaN."""
    xy = kp[..., :2].astype(float).copy(); c = np.nan_to_num(kp[..., 2])
    _, _, s = normalize(clean(kp.astype(float), fps)[0], fps)      # same body scale as the pipeline
    hc = (xy[:, LHIP] + xy[:, RHIP]) / 2
    u = (xy - hc[:, None]) / s[:, None, None]
    u[c < 0.3] = np.nan
    return u, hc / s[:, None], c

uf_raw, hf_raw, cf = raw_norm(kf['kp'], fps_f)
us_raw, hs_raw, cs = raw_norm(ks['kp'], fps_s)
side_idx = np.clip(np.round(tau * fps_s).astype(int), 0, len(us_raw) - 1)   # same time sync as the pipeline
P = fu['P']

fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
# (a) jitter: left wrist height, front view
raw = -(uf_raw[:, LWR, 1] - uf_raw[:, LSH, 1])
ax[0].plot(t, raw, '.', ms=3, color='#9aa7b8', label='raw detection (per frame)')
ax[0].plot(t, P[:, LWR, 2] - P[:, LSH, 2], color='#2a6fdb', lw=2, label='cleaned + fused (front & aligned side views averaged, smoothed)')
ax[0].set_ylabel('left wrist − shoulder\nheight (torso units)'); ax[0].legend(fontsize=8, loc='upper left')
ax[0].set_title('(a) Frame-to-frame jitter removed, movement kept', fontsize=10, loc='left')

# (b) occlusion: right wrist depth, side view
rd = us_raw[side_idx, RWR, 0] - us_raw[side_idx, RSH, 0]
lowc = cs[side_idx, RWR] < 0.3
ax[1].plot(t[~lowc], rd[~lowc], '.', ms=3, color='#e8873a', label='raw side-view detection (visible)')
ax[1].plot(t, P[:, RWR, 1] - P[:, RSH, 1], color='#c0392b', lw=2, label='fused estimate (hidden / low-confidence frames inferred from the left arm)')
ax[1].fill_between(t, -1.5, 1.5, where=lowc, color='#e8873a', alpha=.12, label='right wrist hidden (conf < 0.3)')
ax[1].set_ylim(-1.2, 1.2); ax[1].set_ylabel('right wrist depth\n(+ = behind, torso units)'); ax[1].legend(fontsize=8, loc='upper left')
ax[1].set_title('(b) Hidden far-side arm filled in', fontsize=10, loc='left')

# (c) forward travel: raw side-view hip motion vs trend vs final
R = json.load(open('rest_rain.json'))
scale_m = np.linalg.norm((np.array(R['upper.L']['head']) + np.array(R['upper.R']['head'])) / 2 - np.array(R['hips']['head']))  # Rain torso length (m), same scaling as retarget.py
dep_raw = (hs_raw[side_idx, 0] - np.nanmean(hs_raw[side_idx[:30], 0])) * scale_m
from scipy.signal import savgol_filter
dep_trend = savgol_filter(savgol_filter(dep_raw, int(2 * fps_f) | 1, 2, mode='nearest'), int(2 * fps_f) | 1, 2, mode='nearest')
ax[2].plot(t, -dep_raw, color='#9aa7b8', lw=1, label='raw: hip position in side video')
ax[2].plot(t, -dep_trend, color='#e8873a', lw=2, label='trend line (2 s smoothing)')
ax[2].plot(t, -(T['hips'][:, 1] - T['hips'][0, 1]), color='#2a6fdb', lw=2.5, label='final animation: stands during the swing, then a steady 0.8 m walk forward (design choice, not measured)')
ax[2].set_ylabel('forward travel (m)'); ax[2].legend(fontsize=8, loc='upper left')
ax[2].set_title('(c) Forward movement: measured path vs. trend vs. final animation path', fontsize=10, loc='left')

# (d) sideways drift: raw front-view hip motion vs final
lat_raw = (hf_raw[:, 0] - np.nanmean(hf_raw[:30, 0])) * scale_m
ax[3].plot(t, lat_raw, color='#9aa7b8', lw=1, label='raw: hip position in front video')
ax[3].plot(t, savgol_filter(lat_raw, int(2 * fps_f) | 1, 2, mode='nearest'), color='#e8873a', lw=2, label='trend line')
ax[3].plot(t, T['hips'][:, 0] - T['hips'][0, 0], color='#2a6fdb', lw=2.5, label='final animation: sideways drift removed (design choice)')
ax[3].set_ylabel('sideways drift (m)'); ax[3].set_xlabel('time (s)'); ax[3].legend(fontsize=8, loc='lower left')
ax[3].set_title('(d) Sideways drift: measured vs. final animation', fontsize=10, loc='left')
fig.tight_layout(); fig.savefig('../media/raw_vs_corrected.png', dpi=120)

jit = lambda x: np.nanmean(np.abs(np.diff(x, 2)))
print('left wrist height jitter  raw %.4f -> corrected %.4f' % (jit(raw), jit(P[:, LWR, 2] - P[:, LSH, 2])))
print('right wrist hidden in %.0f%% of frames' % (100 * lowc.mean()))
print('hip forward path jitter   raw %.5f -> final %.5f' % (jit(dep_raw), jit(T['hips'][:, 1])))
print('raw sideways drift range %.3f m -> final 0' % (np.nanmax(lat_raw) - np.nanmin(lat_raw)))
