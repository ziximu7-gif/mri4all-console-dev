"""
Typed, plain-numerical simulation contexts (Phase A).

These dataclasses are the contract between the sequence/adapter layer and
the pure numerical simulation math in ``kspace.py``/``phantom.py``.

ARCHITECTURAL RULE (per approved plan):
The core simulation layer must NOT know about ScanTask persistence or the
``other["resolved_*"]`` JSON contract. The adapter layer (the sequence
modules ``gre_3D.py`` / ``localizer.py``) extracts plain numerical values
from ScanTask and constructs one of these typed contexts. Serialization is
deliberately NOT added here: these are in-process objects passed directly
as ``run_pulseq(..., sim_context=<context>)``.

Coordinate conventions (mirror ``common/geometry.py`` and reconstruction):
- center_scanner_m:    FOV center in scanner XYZ [m], isocenter at origin.
- center_logical_m:    = logical_to_scanner.T @ center_scanner_m  [m]
                       along logical Read/Phase/Third.
- fov_logical_m:       prescribed FOV along logical Read/Phase/Third [m]
                       (Read is NOT oversampled here).
- logical_to_scanner:  3x3 orthogonal matrix mapping logical
                       Read/Phase/Third -> scanner X/Y/Z. Columns are the
                       logical axes expressed in scanner coordinates.
                       ``logical_to_scanner`` may be any orthogonal matrix;
                       the simulator does NOT assume determinant +1.
"""

from dataclasses import dataclass

import numpy as np

__all__ = [
    "GRE3DSimulationContext",
    "Localizer2DSimulationContext",
]


@dataclass(frozen=True)
class GRE3DSimulationContext:
    """
    Geometry for a 3D GRE Cartesian acquisition simulation.

    n_readout is the OVERSAMPLED read length (base_resolution *
    oversampling_read); the reconstruction crops the central read region
    back to base_resolution afterward.
    """

    kind: str
    center_scanner_m: np.ndarray
    center_logical_m: np.ndarray
    fov_logical_m: np.ndarray
    logical_to_scanner: np.ndarray
    base_resolution: int
    n_phase: int
    n_slice: int
    oversampling_read: int

    def __post_init__(self):
        # Validate shapes and finiteness only; no serialization.
        for name in (
            "center_scanner_m",
            "center_logical_m",
            "fov_logical_m",
        ):
            vec = np.asarray(getattr(self, name), dtype=float)
            if vec.shape != (3,):
                raise ValueError(f"{name} must have shape (3,)")
            object.__setattr__(self, name, vec)

        matrix = np.asarray(
            self.logical_to_scanner, dtype=float
        )
        if matrix.shape != (3, 3):
            raise ValueError(
                "logical_to_scanner must have shape (3, 3)"
            )
        # Orthogonal but allow det = -1 (mirrored logical frames).
        if not np.allclose(
            matrix @ matrix.T, np.eye(3), atol=1e-6
        ):
            raise ValueError(
                "logical_to_scanner must be orthogonal"
            )
        object.__setattr__(self, "logical_to_scanner", matrix)

        if self.oversampling_read <= 0:
            raise ValueError("oversampling_read must be positive")
        if self.base_resolution <= 0:
            raise ValueError("base_resolution must be positive")

    @property
    def n_readout(self) -> int:
        return int(self.base_resolution * self.oversampling_read)

    @property
    def sampled_fov_logical_m(self) -> np.ndarray:
        """
        Pre-crop sampled FOV along logical Read/Phase/Third [m].

        Read is enlarged by the oversampling factor; Phase/Third are not.
        This is the FOV over which the ideal (pre-crop) volume is sampled,
        matching reconstruction's view before it crops the read center.
        """
        return np.array(
            [
                self.fov_logical_m[0] * float(self.oversampling_read),
                self.fov_logical_m[1],
                self.fov_logical_m[2],
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class Localizer2DSimulationContext:
    """
    Geometry for one scanner-base 2D Localizer acquisition.

    Read and Phase axes come from ORIENTATION_CHANNELS; the third axis
    is the SLICE-SELECT axis. The slice is centered at scanner
    isocenter (slice coordinate 0) and the signal is integrated only
    through the finite ``slice_thickness_m`` around that center, so the
    image is a thin reconstructed slice - NOT a full-axis projection.

    Phase A models the slice profile as an ideal rectangular (boxcar)
    profile: a geometry-level approximation of the real slice-selective
    sinc RF plus slice gradient, with no Bloch simulation and no Pulseq
    waveform readback.
    """

    kind: str
    orientation: str
    fov_m: float
    slice_thickness_m: float
    n_read: int
    n_phase: int
    nsa: int
    n_slice_samples: int

    def __post_init__(self):
        if self.fov_m <= 0:
            raise ValueError("Localizer FOV must be positive")
        if self.n_read <= 0 or self.n_phase <= 0:
            raise ValueError(
                "Localizer matrix size must be positive"
            )
        if self.nsa <= 0:
            raise ValueError("Localizer NSA must be positive")
        if self.slice_thickness_m <= 0:
            raise ValueError(
                "Localizer slice thickness must be positive"
            )
        if self.n_slice_samples < 2:
            raise ValueError(
                "Localizer slice samples must be at least 2"
            )
