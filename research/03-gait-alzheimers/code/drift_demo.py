"""Long-term gait drift detection — demo on SYNTHETIC data.

Simulates one daily walking recording for 90 days. Gait is stable for 45 days, then deteriorates
gradually (longer and more variable steps times, shorter strides, smaller arm swing), following the
early-Alzheimer's biomarkers in the paper. Each day's skeleton goes through gait_features.py; a
baseline model learned on the first 21 days scores each day; CUSUM accumulates only sustained
deviation so single noisy days don't raise an alarm.

The paper proposes an LSTM autoencoder / VAE as the anomaly scorer. This demo uses a simpler
baseline (Mahalanobis distance on standardized features) so it runs anywhere; swapping in a learned
model is the next step. No real patient data is used here.
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gait_features import gait_features, synthetic_walker

DAYS, ONSET, BASE = 90, 45, 21
FEATS = ['step_time_cv', 'step_time_asymmetry', 'stride_length_norm', 'arm_swing_l', 'arm_swing_r']
rng = np.random.default_rng(7)

rows = []
for d in range(DAYS):
    s = np.clip((d - ONSET) / (DAYS - ONSET), 0, 1)          # 0 before onset -> 1 at the end
    kp = synthetic_walker(stride=0.9 - 0.25 * s + 0.03 * rng.standard_normal(),
                          asym=0.2 * s + 0.02 * abs(rng.standard_normal()),
                          jitter=0.02 + 0.06 * s, arm=0.6 - 0.25 * s + 0.03 * rng.standard_normal(),
                          seed=d)
    kp = kp + rng.normal(0, 1.5, kp.shape)                    # keypoint detection noise (pixels)
    f = gait_features(kp, 30)
    rows.append([f.get(k, np.nan) for k in FEATS])
X = np.array(rows)

mu, sd = X[:BASE].mean(0), X[:BASE].std(0) + 1e-9
Z = (X - mu) / sd
C = np.cov(Z[:BASE].T) + 1e-3 * np.eye(len(FEATS))
score = np.sqrt(np.einsum('ij,jk,ik->i', Z, np.linalg.inv(C), Z))   # Mahalanobis distance per day

k = np.median(score[:BASE]) + 0.5                              # allowance: ignore ordinary day-to-day noise
h = 8.0                                                        # alarm threshold
cusum = np.zeros(DAYS)
for d in range(1, DAYS):
    cusum[d] = max(0.0, cusum[d - 1] + score[d] - k)
alarm = int(np.argmax(cusum > h)) if (cusum > h).any() else None

fig, ax = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
ax[0].plot(X[:, 0], label='step time CV'); ax[0].plot(X[:, 2], label='stride length (torso units)')
ax[0].plot((X[:, 3] + X[:, 4]) / 2, label='arm swing'); ax[0].legend(fontsize=8); ax[0].set_title('Daily gait biomarkers (synthetic)')
ax[1].plot(score, color='tab:gray'); ax[1].axhline(k, ls=':', color='k'); ax[1].set_ylabel('anomaly score')
ax[2].plot(cusum, color='tab:red'); ax[2].axhline(h, ls='--', color='tab:red'); ax[2].set_ylabel('CUSUM drift'); ax[2].set_xlabel('day')
for a in ax:
    a.axvline(ONSET, color='tab:blue', alpha=.4); a.axvspan(0, BASE, color='tab:green', alpha=.08)
if alarm is not None:
    for a in ax: a.axvline(alarm, color='tab:red', alpha=.5)
fig.text(0.01, 0.005, f'green: baseline days 0-{BASE - 1} | blue: simulated onset day {ONSET} | red: drift alarm day {alarm}', fontsize=8)
fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig('../figures/drift_demo.png', dpi=120)

false_before = int((cusum[:ONSET] > h).sum())
print(f'onset day {ONSET}, alarm day {alarm}, days in alarm before onset: {false_before}')
