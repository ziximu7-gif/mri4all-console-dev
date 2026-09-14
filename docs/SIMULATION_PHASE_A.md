# Phase A Geometry-Aware Hardware Simulation

Status: Phase A (ideal Cartesian, geometry-aware). This document records
the contract, conventions and explicit limitations of the simulator that
was added in `common/simulation/`.

## Goal

Simulate MRI acquisition data from ONE fixed 3D digital phantom defined
in physical Scanner XYZ coordinates (meters, isocenter at origin). The
object never changes; changing FOV center, FOV size, FOV rotation,
Orientation or Readout Direction changes only how that same fixed object
is sampled into the logical Read/Phase/Third acquisition volume.

This replaces the previous plumbing-level simulation, which invented one
independent 2D phantom per Localizer orientation and built the 3D
phantom directly in logical coordinates (no geometry awareness).

## Module boundary

```
common/simulation/
    phantom.py    the single fixed scanner-space phantom (pure numpy)
    context.py    typed plain-number contexts (dataclasses, no ScanTask)
    kspace.py     forward Cartesian k-space / raw-ADC synthesis
```

Rules:

- Pure numerical boundary: no PyQt, no marcos/flocra, no Pulseq, no
  ScanTask persistence knowledge, and NO import of reconstruction code.
- Adapter layers live in the sequences:
  `sequences/gre_3D.py::build_simulation_context` extracts
  `other["resolved_geometry"]` / `other["resolved_encoding"]` into a
  `GRE3DSimulationContext`;
  `sequences/localizer.py::build_simulation_context` builds a
  `Localizer2DSimulationContext` from its own parameters.
- `run_pulseq(..., sim_context=None)` (external/seq/adjustments_acq/
  scripts.py) is a thin dispatcher. `sim_context=None` preserves the
  legacy simulation branches bit-for-bit (TSE3D/bssfp/FID and the legacy
  filename-sniffing Localizer branch remain untouched). The real
  hardware path never sees simulation code.

## The phantom (`common/simulation/phantom.py`)

Deterministic, deliberately asymmetric, hard-edged ellipsoids in meter
coordinates (`PHANTOM_MARKERS`):

| marker        | center (X, Y, Z) [m]        | intensity |
|---------------|-----------------------------|-----------|
| body          | ( 0.004, -0.006,  0.002)    | 1.0       |
| bright        | ( 0.030,  0.020, -0.020)    | 2.0       |
| dark          | (-0.026,  0.030,  0.022)    | 0.45      |
| chiral_high   | ( 0.020, -0.032,  0.034)    | 3.0       |
| chiral_low    | (-0.034, -0.012, -0.030)    | 0.2       |

Semi-axes differ per axis and markers sit at generic non-symmetric
positions, so the object is asymmetric and chiral: +90 and -90 degree
FOV rotations produce distinguishable logical volumes. Sampling
(`sample_phantom`) is a pure hard-threshold mapping from scanner-space
points to intensity, shared by GRE3D and Localizer simulation.

## Forward model and conventions

For GRE3D, with resolved planning geometry
(`logical_to_scanner` = M, orthogonal, det may be -1):

```
sampled_fov_logical = [fov_read * oversampling_read, fov_phase, fov_third]
coord(i)            = (i - center_index) * sampled_fov / n
center_index:  read = n_readout // 2
               phase = n_phase - n_phase // 2
               third = n_slice - n_slice // 2
r_logical     = (read, phase, third) grid coordinates
r_scanner     = center_scanner + M @ r_logical
image(r_logical) = sample_phantom(r_scanner)      # ideal FOV-centered volume
kspace        = ifftshift(ifftn(ifftshift(image))) # exact inverse of recon's
                                                   # fftshift(fftn(fftshift))
kspace       *= exp(+2*pi*i * k_axis * center_logical_axis)   # forward ramp
k_axis        = (index - center_index) / sampled_fov_axis
raw[line]     = kspace[:, pe_index, slice_index] * exp(-1j * adc_phase)
pe_index      = (center_phase   - order[line,0]) % n_phase
slice_index   = (center_slice   - order[line,1]) % n_slice
```

Key points:

- The forward center phase `exp(+2*pi*i*k*center_logical)` is
  implemented INDEPENDENTLY in the simulator (no import of
  `recon.recon_utils.cartesian_translation`). It is the exact algebraic
  inverse of reconstruction's `exp(-2*pi*i*k*center_logical)` ramp, using
  the same k definitions, so the two cancel and reconstruction returns
  the ideal FOV-centered volume. The inverse property is locked by
  regression tests, not by shared code.
- Readout oversampling is explicit: the pre-crop volume spans
  `fov_read * oversampling_read` with `n_readout = base_resolution *
  oversampling_read` samples. Reconstruction crops the central
  `n_readout/4 : 3*n_readout/4` read region back to the prescribed Read
  FOV, so final Read voxel size = fov_read / base_resolution.
- Raw ordering matches the real pipeline: `pe_order.npy` (col0 = phase,
  col1 = slice, center-out from `choose_pe_order`) and `adc_phase.npy`
  in degrees; reconstruction restores the phase with `exp(+1j*adc)`.
- Reconstruction's empirical readout phase ramp only affects phase
  (unit modulus), so simulated magnitude images are unaffected.

### Translation sign convention (locked by test)

With identity encoding, moving the FOV center to +X moves a fixed
scanner-space object toward NEGATIVE logical Read
(`r_logical = M^T (r_scanner - center)`).

### Localizer

The current Localizer sequence uses non-selective block RF pulses with
only Read + Phase encoding (no slice-selective excitation). Therefore
the third scanner axis (`ORIENTATION_CHANNELS[orientation][2]`) is
unencoded and the simulated image is the PROJECTION of the same fixed 3D
phantom along that axis (deterministic Riemann summation over a range
that fully contains the object), not a zero-thickness slice.

- Axial: Read=X, Phase=Y, projected along Z
- Coronal: Read=X, Phase=Z, projected along Y
- Sagittal: Read=Y, Phase=Z, projected along X

Raw contract preserved: `raw.reshape(nsa, ny, nx)` with ny = Phase rows,
nx = Read columns, consumed by `run_reconstruction_localizer2d` via
`fftshift(fft2(fftshift(.)))`. All three views come from the SAME 3D
phantom (marginal-projection consistency is locked by test).

## Worked example (locked by test)

Phantom marker `bright` at scanner coordinates
`(0.030, 0.020, -0.020) m`.

FOV rotation rotation_z = +90 deg with `R = Rz @ Ry @ Rx`
(plananner convention, local -> scanner):

```
logical_to_scanner M = Rz(+90)
r_logical = M^T @ r_scanner = Rz(-90) @ (0.030, 0.020, -0.020)
          = (y, -x, z) = (0.020, -0.030, -0.020) m
```

With fov_logical = [0.16, 0.24, 0.12] m and matrix 24(read, 2x
oversampled to 48) x 24 x 12, the reconstructed voxel is
(0.16/24, 0.24/24, 0.12/12) = (6.67, 10, 10) mm. Indexing with the
read DC bin at `n//2` and the phase/third DC bins at `n - n//2`:

```
read  index = round(0.020 / 6.67e-3) + 12 = 3 + 12 = 15
phase index = round(-0.030 / 10e-3) + 12  = -3 + 12 = 9
third index = round(-0.020 / 10e-3) + 6   = -2 + 6  = 4
```

So the marker lands on reconstructed index (15, 9, 4) with intensity
2.0. For rotation_z = -90 deg, `r_logical = (-0.020, 0.030, -0.020)`,
giving index (9, 15, 4) — the two rotations are distinguishable. Both
positions were verified against the real
`reconstruct_cartesian_3d_complex` output, not guessed.

## PHASE A LIMITATIONS (explicit)

1. NOT an independent validation of physical gradient-transform
   correctness. The simulator samples the phantom with the RESOLVED
   acquisition geometry and reconstruction uses the same resolved
   geometry; a wrong `logical_to_scanner` semantic would be invisible to
   these tests. What IS validated: planning -> raw ordering -> k-space
   conventions -> reconstruction -> crop/display geometry consistency.
2. No aliasing model. The simulated acquisition represents an ideal
   selected FOV; object parts outside the prescribed FOV are simply not
   sampled (real hardware would wrap them).
3. No physics: no Bloch simulation, T1/T2, B0/B1, coil sensitivity,
   noise, eddy currents, RF excitation profile. Piecewise-constant
   phantom, hard ellipsoid edges.
4. Localizer is an ideal projection (non-selective RF approximation);
   no slice profile, no relaxation weighting (the real sequence is SE).
5. Simulation uses resolved-geometry values; it does not read the
   generated Pulseq waveform. PE order / ADC phases come from the files
   the sequence already writes, so ordering contracts are shared with
   the real path, but gradient/ADC timing is not.

## Phase B outlook

Derive k(t) by integrating the physical Gx/Gy/Gz waveforms from the
Pulseq file against ADC sample timing, making the simulation an
independent check of `sequences/common/gradient_transform.py` and of
trajectory/FOV consistency. Optionally add relaxation, noise and
aliasing.

## Tests

`tests/test_simulation_geometry.py`:

- phantom determinism / asymmetry
- identity zero-center GRE round-trip (exact, vs independent sampling)
- readout oversampling contract (n_readout, sampled FOV, final read
  size, voxel spacing via a hand-calculated marker peak)
- +90 deg marker position, hand-calculated (index (5,3,2))
- +90 vs -90 distinguishability (chirality)
- translation sign (+X FOV center -> object moves to negative Read)
- Localizer Axial/Coronal/Sagittal from the same 3D phantom
  (marginal-projection consistency)
- orthogonal `logical_to_scanner` with det = -1
- run_pulseq dispatch with GRE3D / Localizer2D contexts
- legacy `sim_context=None` behavior unchanged (bit-exact legacy
  phantom; geometry-aware path asserted NOT called)
- `run_pulseq` signature defaults preserved (real hardware path)
