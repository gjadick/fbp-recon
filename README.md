# fbp-recon
A Python framework for filtered backprojection reconstruction (FBP) of 3D CT data,
with an adjustable ramp filter and conjugate ray weighting cone beam correction.

![Example Recon](example/matrix.gif)

# Usage
Acquisition params are set within main().
Reconstruction params are passed as arguments to main().

One key option is "use_GPU", which determines the function called to do the reconstruction.
If Nvidia GPUs are available, this option should be set to "True".
GPU acceleration will reduce the reconstruction time by several orders of magnitude.
(a single 512x512 slice will take a few seconds instead of a half hour).

An example main() call is:

```main(proj_dir, z_width, FOV, N_matrix, ramp_percent, kl, detail_mode=detail_mode, use_GPU=False)```

where the parameters are:

  - proj_dir: the name of the directory with the projection data in dcm format
  - z_width: reconstruction slice thickness [mm]
  - FOV: reconstruction field of view [mm]
  - N_matrix: number of pixels in one dimension of reconstruction matrix
  - ramp_percent: (0 to 1) percentage of Nyquist frequency for projection filtering, higher gives noiser data but better detail
  - kl: (0 to 1) cone beam correction strength, higher improves resolution but may cause artifacts
  - detail_mode: (T/F) whether cone beam correction will scale from 0 to 1 (detailed) or just 0.5 to 1 (softer)
  - use_GPU: (T/F) whether an Nvidia GPU should be used for this recon

