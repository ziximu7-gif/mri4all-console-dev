"""
Phase A Cartesian k-space / raw-ADC synthesis from the fixed
scanner-space phantom (pure numerical, no reconstruction import).

FORWARD CENTER-PHASE (independent of reconstruction)
----------------------------------------------------
Reconstruction re-centers k-space with apply_cartesian_center_translation,
multiplyng by::

    exp(-2*pi*i * k_axis * center_logical_axis)

using, per axis,

    k = (index - center_index) / sampled_fov,
    center_index_read  = n_readout // 2
    center_index_phase = n_phase   - n_phase // 2
    center_index_third = n_slice   - n_slice // 2
    sampled_fov = [fov_read*oversampling_read, fov_phase, fov_third]

Per the approved plan the simulator must NOT import or call that function.
It instead implements the FORWARD model that is exactly its algebraic
inverse: starting from an ideal FOV-centered k-space it encodes::

    exp(+2*pi*i * k_axis * center_logical_axis)

with the SAME k-axis definition as above, so that reconstruction's
forward (-) ramp cancels it and the ideal centered volume is recovered.

The forward model:
    r_scanner = center_scanner + logical_to_scanner @ r_logical
sample the fixed phantom at r_scanner to obtain the ideal FOV-centered
volume img(r_logical). Reconstruction's ramp then shifts it by
-center_logical along each logical axis, re-centering the physical object
at its scanner location.

FFT CONVENTION ALIGNMENT
------------------------
img is built on an fftshift-centered grid; its Cartesian k-space is

    kspace = ifftshift(ifftn(ifftshift(img)))

which is the exact inverse of reconstruction's

    image = fftshift(fftn(fftshift(kspace)))

(verified by the numpy FFT-shift algebra and locked by the identity
round-trip regression test). Per reconstruction, raw lines are
kspace[:, pe_index, slice_index] * exp(-1j*adc_phase), consumed back as
raw*exp(+1j*adc_phase).
"""

from typing import Sequence

import numpy as np

from common.geometry import ORIENTATION_CHANNELS
from common.simulation.context import (
    GRE3DSimulationContext,
    Localizer2DSimulationContext,
)
from common.simulation.phantom import (
    sample_phantom,
)

__all__ = [
    "_forward_center_ramp",
    "synthesize_gre3d_raw",
    "synthesize_localizer_kspace",
]


def _logical_axis_coords(
    n: int,
    fov: float,
    center_index: int,
) -> np.ndarray:
    """
    Centered logical coordinates [m] for one axis.

    Coordinate zero is anchored at ``center_index``, chosen per axis to
    match the k-space DC-bin index used by the consuming reconstruction
    (GRE: read n//2, phase/third n - n//2, identical to
    reconstruct_cartesian_3d_complex; even matrix sizes make both
    conventions coincide). The exact inverse-transform property is locked
    by the identity round-trip regression test, not assumed.

        coord(i) = (i - center_index) * fov / n
    """
    return (
        (np.arange(n, dtype=float) - center_index) * (fov / n)
    )


def _forward_center_ramp(
    kspace: np.ndarray,
    center_logical_m,
    sampled_fov_logical_m,
    center_indices,
) -> np.ndarray:
    """
    Independent forward center phase: exp(+2*pi*i * k * center_logical).

    k-axes are built with the SAME index convention as reconstruction's
    center correction, so this is its exact algebraic inverse. This code
    does not import reconstruction; the sign is the inverse of recon's
    exp(-2*pi*i*k*c) by construction.

    kspace axes: (read, phase, third).
    """
    kspace = np.asarray(kspace, dtype=np.complex128)
    center_logical_m = np.asarray(
        center_logical_m, dtype=float
    )
    sampled_fov = np.asarray(
        sampled_fov_logical_m, dtype=float
    )
    center_indices = np.asarray(center_indices, dtype=float)

    k_read = (
        np.arange(kspace.shape[0], dtype=float)
        - center_indices[0]
    ) / sampled_fov[0]
    k_phase = (
        np.arange(kspace.shape[1], dtype=float)
        - center_indices[1]
    ) / sampled_fov[1]
    k_third = (
        np.arange(kspace.shape[2], dtype=float)
        - center_indices[2]
    ) / sampled_fov[2]

    # FORWARD (+) ramp: inverse of reconstruction's (-) ramp.
    ramp_read = np.exp(
        +2j * np.pi * k_read * center_logical_m[0]
    )[:, None, None]
    ramp_phase = np.exp(
        +2j * np.pi * k_phase * center_logical_m[1]
    )[None, :, None]
    ramp_third = np.exp(
        +2j * np.pi * k_third * center_logical_m[2]
    )[None, None, :]

    return kspace * ramp_read * ramp_phase * ramp_third


def synthesize_gre3d_raw(
    ctx: GRE3DSimulationContext,
    pe_order,
    adc_phases,
) -> np.ndarray:
    """
    Synthesize a flat complex raw-ADC stream for a 3D GRE acquisition of
    the fixed scanner-space phantom, compatible with run_reconstruction
    -> reconstruct_cartesian_3d_complex.

    Parameters
    ----------
    ctx:
        GRE3D geometry context (plain numbers).
    pe_order:
        (n_lines, 2) integer table; col0 = phase encode index,
        col1 = slice encode index (matches make_gre_3D / get_trajectory).
    adc_phases:
        RF-spoiling phase per line in DEGREES (matches make_gre_3D).

    Returns
    -------
    np.ndarray:
        Flat complex128 stream of length n_lines * n_readout, in the same
        row order reconstruction expects (line order from pe_order).
    """
    pe_order = np.asarray(pe_order)
    adc_phases = np.asarray(adc_phases, dtype=float)

    n_read = ctx.n_readout
    n_phase = ctx.n_phase
    n_slice = ctx.n_slice

    if pe_order.shape[0] != n_phase * n_slice:
        raise ValueError(
            "PE order length does not match "
            f"n_phase*n_slice = {n_phase * n_slice}"
        )
    if pe_order.shape[0] != adc_phases.shape[0]:
        raise ValueError(
            "PE order and ADC phases must have one "
            "entry per k-space line"
        )

    # DC-bin convention used by reconstruct_cartesian_3d_complex for the
    # forward center phase: read = n//2, phase/third = n - n//2.
    center_read = n_read // 2
    center_phase = n_phase - n_phase // 2
    center_slice = n_slice - n_slice // 2
    center_indices = (
        center_read,
        center_phase,
        center_slice,
    )

    # ---- 1. Ideal FOV-centered logical volume (pre-crop, oversampled).
    #
    # The image is sampled on an ifftshift-centered grid: coordinate zero
    # at index n//2 on every axis. This is the exact pivot of the
    # ifftshift(ifftn(ifftshift(.))) transform below, which is the
    # algebraic inverse of reconstruction's
    # fftshift(fftn(fftshift(.))) (locked by the identity round-trip
    # test). For the even matrix sizes used by GRE3D the ifftshift pivot
    # n//2 equals the DC-bin convention n - n//2 used by the forward ramp.
    sampled_fov = ctx.sampled_fov_logical_m
    read_c = _logical_axis_coords(
        n_read, sampled_fov[0], center_read
    )
    phase_c = _logical_axis_coords(
        n_phase, sampled_fov[1], center_phase
    )
    third_c = _logical_axis_coords(
        n_slice, sampled_fov[2], center_slice
    )

    rr, pp, tt = np.meshgrid(
        read_c, phase_c, third_c, indexing="ij"
    )
    # r_logical axes stacked as (read, phase, third, 3).
    r_logical = np.stack([rr, pp, tt], axis=-1)

    # r_scanner = center + M @ r_logical  (M = logical_to_scanner).
    r_scanner = (
        ctx.center_scanner_m
        + r_logical @ ctx.logical_to_scanner.T
    )

    image = sample_phantom(r_scanner).astype(
        np.complex128
    )

    # ---- 2. Cartesian k-space of the centered volume (exact inverse of
    #         reconstruction's fftshift(fftn(fftshift(.)))).
    kspace = np.fft.ifftshift(
        np.fft.ifftn(np.fft.ifftshift(image))
    )

    # ---- 3. Independent forward center phase (inverse of recon ramp).
    kspace = _forward_center_ramp(
        kspace=kspace,
        center_logical_m=ctx.center_logical_m,
        sampled_fov_logical_m=sampled_fov,
        center_indices=center_indices,
    )

    # ---- 4. Read lines in PE order, applying the negated ADC phase that
    #         reconstruction later restores with exp(+1j*adc_phase).

    raw = np.zeros(
        (pe_order.shape[0], n_read), dtype=np.complex128
    )
    for counter, line in enumerate(pe_order):
        pe_index = (center_phase - int(line[0])) % n_phase
        slice_index = (center_slice - int(line[1])) % n_slice
        phase = float(adc_phases[counter]) / 180.0 * np.pi
        raw[counter, :] = (
            kspace[:, pe_index, slice_index]
            * np.exp(-1j * phase)
        )

    return raw.reshape(-1)


def synthesize_localizer_kspace(
    ctx: Localizer2DSimulationContext,
) -> np.ndarray:
    """
    Synthesize a flat complex Localizer raw stream of length
    nsa * n_phase * n_read, preserving the existing reconstruction
    contract::

        raw.reshape(nsa, ny, nx)

    where nx (= n_read) are Read columns and ny (= n_phase) are Phase rows.
    This output/raw contract is unchanged by the slice-selective model.

    The Localizer is scanner-base and centered at isocenter: per
    ORIENTATION_CHANNELS the read axis is ch0 and the phase axis is
    ch1, while ch2 is the SLICE-SELECT axis. The excited slab is a
    finite slice centered on the isocenter (slice coordinate 0), and
    the signal is integrated only over ``slice_thickness_m``.

    Phase A approximates the slice profile as an ideal rectangular
    (boxcar) profile - a geometry-level stand-in for the real
    slice-selective sinc RF plus slice gradient. No Bloch simulation
    is performed and no Pulseq waveform is read back.

    The 2D centered image -> k-space uses ifftshift(ifft2(ifftshift(img)))
    matching run_reconstruction_localizer2d's fftshift(fft2(fftshift(.))).
    """
    read_axis, phase_axis, slice_axis = ORIENTATION_CHANNELS[
        ctx.orientation
    ]
    axis_index = {"x": 0, "y": 1, "z": 2}
    read_dim = axis_index[read_axis]
    phase_dim = axis_index[phase_axis]
    slice_dim = axis_index[slice_axis]

    n_read = ctx.n_read
    n_phase = ctx.n_phase

    # Square Read x Phase FOV centered at isocenter (Localizer is
    # scanner-base and centered). run_reconstruction_localizer2d uses
    # fftshift(fft2(fftshift(.))), whose pivot is n//2 on both axes.
    read_c = _logical_axis_coords(
        n_read, ctx.fov_m, n_read // 2
    )
    phase_c = _logical_axis_coords(
        n_phase, ctx.fov_m, n_phase // 2
    )

    # Finite slice centered at scanner isocenter: the slice-select
    # coordinate spans -slice_thickness/2 .. +slice_thickness/2. The
    # Localizer slice is centered by construction (no slice offset),
    # so the slab is symmetric about coordinate 0.
    slice_half = (
        0.5
        * ctx.slice_thickness_m
    )

    slice_c = np.linspace(
        -slice_half,
        slice_half,
        ctx.n_slice_samples,
        dtype=float,
    )

    # Build (phase, read, slice) scanner coordinate grids.
    # Raw contract: raw.reshape(nsa, ny, nx) with ny = Phase rows and
    # nx = Read columns (run_reconstruction_localizer2d).
    image = np.zeros((n_phase, n_read), dtype=np.complex128)

    rr, pph = np.meshgrid(
        read_c, phase_c, indexing="xy"
    )  # shape (n_phase, n_read): rr varies along columns,
    # pph along rows.

    # Integrate the phantom through the finite slice only, using the
    # trapezoidal rule (endpoint half weights) - a deterministic
    # numerical integration. Phase A treats the slice profile as an
    # ideal rectangular (boxcar) profile.
    slice_spacing = (
        ctx.slice_thickness_m
        / (ctx.n_slice_samples - 1)
    )

    for sample, slice_position in enumerate(
        slice_c
    ):
        points = np.zeros(
            (n_phase, n_read, 3),
            dtype=float,
        )

        points[:, :, read_dim] = rr
        points[:, :, phase_dim] = pph
        points[:, :, slice_dim] = (
            slice_position
        )

        weight = (
            0.5
            if sample in (
                0,
                len(slice_c) - 1,
            )
            else 1.0
        )

        image += (
            weight
            * sample_phantom(points)
        )

    image = (
        image
        * slice_spacing
    )

    # Centered image -> k-space (flat stream, NSA=1 row repeated).
    kspace = np.fft.ifftshift(
        np.fft.ifft2(np.fft.ifftshift(image))
    )

    flat = kspace.reshape(-1)
    # Replicate across NSA so raw.reshape(nsa, ny, nx) is well-formed.
    return np.tile(flat, ctx.nsa)
