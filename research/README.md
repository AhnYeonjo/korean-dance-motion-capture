# Research Journey

How this project grew, step by step: from recovering 3D shape out of photos, to reading meaning from movement, to capturing Korean dance in 3D.

| # | Project | Type | Key idea carried forward |
|---|---|---|---|
| 01 | [3D Reconstruction from Photos (Meshroom)](01-3d-reconstruction/) | School convergence project, grade 10 | 3D structure can be recovered from several 2D views. |
| 02 | [Gait Recognition for Drone/CCTV Missing-Person Search](02-gait-recognition-drone/) | School STEAM R&E, 2024 (team of 2) | Movement is a signal: a person can be recognized from how they walk. |
| 03 | [AI Gait Analysis for Early Alzheimer's Detection](03-gait-alzheimers/) | Team paper (KSFI International STEAM), plus working code | Skeleton-only gait biomarkers and long-term drift as early warning. |
| 04 | [Korean Dance Motion Capture → Blender](../README.md) | Independent build, October 2026 | Front and side videos fused into 3D motion that drives a rigged character. |

## Where it is heading
Projects 03 and 04 share one pipeline: video → pose keypoints → cleaned time series → features. The current work connects them.
- **Preservation and teaching:** capture traditional Korean dance in 3D and replay it on characters, so learners can study a movement from any angle.
- **Cognitive use:** use the same pose pipeline to measure gait and movement quality (`03-gait-alzheimers/code`), then compare a learner's movement against a reference.

The next milestones are listed under *Limitations and next steps* in the [main README](../README.md) and *Next steps* in [03](03-gait-alzheimers/).
