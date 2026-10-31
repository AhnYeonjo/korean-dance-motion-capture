# 01 · 3D Reconstruction from Photos (Meshroom)

*School convergence project, grade 10 (final report written in Korean; summarized here in English).*

## Question
Can usable 3D models for games and avatars be built from ordinary photos instead of modeling them by hand? And which shooting conditions keep reconstruction errors to a minimum?

## Method
- Photographed real objects from many angles with a phone camera.
- Reconstructed them in **Meshroom**, an open-source photogrammetry tool:
  1. **Feature extraction**: finds distinctive points in each photo.
  2. **Feature matching**: finds the same points across different photos and estimates each camera's position.
  3. **Point cloud → mesh**: turns the matched 3D points into a triangle mesh.
  4. **Texturing**: projects the best-matching photo onto each face of the mesh, using the camera poses.
- Cleaned up stray geometry and edited the textures in a separate editor.

## Experiments and findings
| Object | Result |
|---|---|
| Stuffed rabbit (matte, textured) | Reconstructed cleanly with little difficulty. |
| Honey jar in clear plastic wrap, with a thin stick | The transparent wrap didn't reconstruct, and the thin stick came out broken. |

Photogrammetry works when a surface has visible texture that can be matched between photos. Transparent, reflective and very thin parts give the matcher nothing to lock onto.

## What it led to
The core idea, recovering 3D structure from several 2D views, comes back in [04 · dance motion capture](../../README.md), where two 2D camera views of a moving body are combined into 3D motion.
