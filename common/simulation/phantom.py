"""
Fixed scanner-space digital phantom for Phase A hardware simulation.

ARCHITECTURAL CONTRACT
----------------------
The phantom is defined ONCE, deterministically, in physical Scanner XYZ
coordinates measured in meters, with the isocenter at the origin. It is a
property of the simulated *object*, completely independent of any FOV,
orientation, gradient transform, reconstruction or display choice.

Changing FOV center / size / rotation / Orientation / Readout Direction
changes how this SAME fixed object is sampled into the logical
Read/Phase/Third volume; it never changes the object itself.

This is pure numerical code: no PyQt, no marcos/flocra hardware, no Pulseq,
no ScanTask, and no reconstruction imports.

PHASE A LIMITATION
------------------
This phantom plus the k-space synthesizer in ``kspace.py`` validate the
planning -> raw -> reconstruction -> display GEOMETRY chain. They do NOT
independently validate physical gradient-transform correctness, because the
sampled object and the reconstruction use the same ``logical_to_scanner``
matrix. An independent validation would derive k(t) from physical Gx/Gy/Gz
and ADC timing (a possible Phase B). No aliasing model is included in
Phase A; the simulated acquisition represents an ideal selected FOV.

PHASE A NON-GOALS (deliberately excluded):
Bloch simulation, T1/T2, B0/B1 effects, coil sensitivity, noise, eddy
currents, RF physics, physical-gradient trajectory integration.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from common.geometry import ORIENTATION_CHANNELS

__all__ = [
    "PhantomMarker",
    "PHANTOM_MARKERS",
    "sample_phantom",
    "phantom_bounds_m",
    "localizer_plane_axes",
]


@dataclass(frozen=True)
class PhantomMarker:
    """
    One named ellipsoidal feature of the fixed scanner-space phantom.

    center_m:
        Feature center in scanner XYZ [m].
    semi_axes_m:
        Ellipsoid semi-axes along scanner X/Y/Z [m]. The phantom is
        deliberately asymmetric (semi-axes differ) and chiral (markers
        are arranged so that +rotation and -rotation renders are
        distinguishable).
    intensity:
        Feature signal intensity (arbitrary simulation units).
    """

    name: str
    center_m: np.ndarray
    semi_axes_m: np.ndarray
    intensity: float


# ---------------------------------------------------------------------
# The single fixed scanner-space phantom.
#
# Coordinates are in METERS in scanner XYZ with the isocenter at origin.
# These constants are the ground truth object; they must not depend on
# any acquisition geometry.
#
# A main asymmetric body plus several distinct-intensity point markers in
# a chiral (handed) arrangement. Because the markers have different
# intensities at generic non-symmetric positions, a +90 degree and a -90
# degree FOV rotation of this object produce different logical volumes.
# ---------------------------------------------------------------------
_M = np.array  # local alias for compact literal arrays

PHANTOM_MARKERS: Dict[str, PhantomMarker] = {
    "body": PhantomMarker(
        name="body",
        center_m=_M([0.004, -0.006, 0.002]),
        semi_axes_m=_M([0.055, 0.072, 0.088]),
        intensity=1.0,
    ),
    "bright": PhantomMarker(
        name="bright",
        center_m=_M([0.030, 0.020, -0.020]),
        semi_axes_m=_M([0.012, 0.012, 0.012]),
        intensity=2.0,
    ),
    "dark": PhantomMarker(
        name="dark",
        center_m=_M([-0.026, 0.030, 0.022]),
        semi_axes_m=_M([0.012, 0.012, 0.012]),
        intensity=0.45,
    ),
    # Chiral markers: distinct intensities at non-collinear positions so
    # handedness (+90 vs -90 rotation) is observable. Radii are chosen to
    # remain resolvable by nearest-grid-point sampling even at coarse
    # matrices (a feature smaller than one voxel could otherwise be
    # missed entirely).
    "chiral_high": PhantomMarker(
        name="chiral_high",
        center_m=_M([0.020, -0.032, 0.034]),
        semi_axes_m=_M([0.010, 0.010, 0.010]),
        intensity=3.0,
    ),
    "chiral_low": PhantomMarker(
        name="chiral_low",
        center_m=_M([-0.034, -0.012, -0.030]),
        semi_axes_m=_M([0.010, 0.010, 0.010]),
        intensity=0.2,
    ),
}

# Deterministic paint order. Later features overwrite earlier ones inside
# overlap, giving well-defined, reproducible intensities.
_MARKER_ORDER: Tuple[str, ...] = (
    "body",
    "dark",
    "bright",
    "chiral_low",
    "chiral_high",
)


def sample_phantom(points_scanner_m):
    """
    Evaluate the fixed phantom at arbitrary scanner-space points.

    Parameters
    ----------
    points_scanner_m:
        Array of shape (..., 3) with scanner X/Y/Z coordinates [m].

    Returns
    -------
    np.ndarray:
        Real intensity array with the leading shape of the input points.

    The phantom is piecewise constant (hard ellipsoid boundaries, no
    interpolation) so the result is fully deterministic. The function is
    a pure mapping from scanner-space coordinates to intensity and is the
    single source of truth shared by GRE3D and Localizer simulation.
    """
    points = np.asarray(points_scanner_m, dtype=float)

    if points.shape[-1] != 3:
        raise ValueError(
            "sample_phantom expects scanner points "
            "with a trailing axis of length 3, "
            f"got shape {points.shape}"
        )

    leading = points.shape[:-1]
    values = np.zeros(leading, dtype=np.float64)

    for key in _MARKER_ORDER:
        marker = PHANTOM_MARKERS[key]
        delta = points - marker.center_m
        # Ellipsoid interior test: sum((delta/semi_axis)^2) <= 1.
        normalized = delta / marker.semi_axes_m
        inside = np.sum(normalized * normalized, axis=-1) <= 1.0
        values = np.where(inside, marker.intensity, values)

    return values


def phantom_bounds_m():
    """
    Axis-aligned bounding box of the fixed phantom in scanner XYZ [m].

    Returns (min_corner, max_corner), each a length-3 array. Used by the
    Localizer projection to choose an integration range along the
    unencoded axis that fully contains the same fixed 3D object (the
    Localizer RF is non-selective, so all spins along the unencoded axis
    contribute).
    """
    mins = []
    maxs = []
    for key in _MARKER_ORDER:
        marker = PHANTOM_MARKERS[key]
        mins.append(marker.center_m - marker.semi_axes_m)
        maxs.append(marker.center_m + marker.semi_axes_m)
    return (
        np.min(np.stack(mins), axis=0),
        np.max(np.stack(maxs), axis=0),
    )


def localizer_plane_axes(orientation: str) -> Tuple[str, str, str]:
    """
    Return (read_axis, phase_axis, projection_axis) scanner axis names.

    The current Localizer uses non-selective block RF pulses with only
    Read + Phase encoding and NO slice-selective excitation. Therefore the
    third scanner axis (ch2) is the unencoded direction along which the
    fixed 3D phantom is projected by deterministic summation.
    """
    return ORIENTATION_CHANNELS[orientation]
