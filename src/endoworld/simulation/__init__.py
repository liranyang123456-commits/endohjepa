"""Stage 3 - Simulation: image/video -> physically-consistent, renderable 3D world.

Three complementary components:
  A. reconstruct3d      - depth/stereo -> point cloud / 3D Gaussian Splatting (renderable geometry)
  B. latent_world_model - action-conditioned latent dynamics ("imagine the future")
  C. physics            - deformable soft-body simulation coupled to reconstructed geometry
"""
