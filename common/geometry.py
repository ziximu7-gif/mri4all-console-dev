from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

MM_TO_M = 1e-3


def mm_to_m(value):
    return value * MM_TO_M


def m_to_mm(value):
    return value / MM_TO_M
# ---------------------------------------------------------
# Scanner coordinate convention
# ---------------------------------------------------------
#
# Scanner physical axes:
#
#   X -> Gx
#   Y -> Gy
#   Z -> Gz
#
# Imaging planes:
#
#   Axial    -> X-Y
#   Coronal  -> X-Z
#   Sagittal -> Y-Z
#
# Returned channel order is:
#
#   read, phase, third-axis
#
# For 3D Cartesian acquisition the third axis is the
# second phase-encoding direction.
#
ORIENTATION_CHANNELS = {
    "Axial": ("x", "y", "z"),
    "Coronal": ("x", "z", "y"),
    "Sagittal": ("y", "z", "x"),
}

ORIENTATION_PLANE_AXES = {
    "Axial": ("x", "y"),
    "Coronal": ("x", "z"),
    "Sagittal": ("y", "z"),
}


@dataclass
class ScanGeometry:
    """
    Physical scan geometry.

    center_scanner_m:
        FOV center in scanner X/Y/Z coordinates [m].

    fov_local_m:
        FOV dimensions along the local read/phase/third axes [m].

    rotation_local_to_scanner:
        3x3 rotation matrix mapping local FOV axes into
        scanner physical X/Y/Z axes.
    """

    center_scanner_m: np.ndarray
    fov_local_m: np.ndarray
    rotation_local_to_scanner: np.ndarray

    def as_dict(self):
        return {
            "center_scanner_m": (
                self.center_scanner_m.tolist()
            ),
            "fov_local_m": (
                self.fov_local_m.tolist()
            ),
            "rotation_local_to_scanner": (
                self.rotation_local_to_scanner.tolist()
            ),
        }


def orientation_channels(
    orientation: str,
):
    try:
        return ORIENTATION_CHANNELS[orientation]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported orientation: {orientation}"
        ) from exc


def orientation_plane_axes(
    orientation: str,
):
    try:
        return ORIENTATION_PLANE_AXES[orientation]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported orientation: {orientation}"
        ) from exc


def planning_euler_to_matrix(
    rotation_x_deg: float,
    rotation_y_deg: float,
    rotation_z_deg: float,
) -> np.ndarray:
    """
    Convert planning Euler angles into a local-to-scanner
    rotation matrix.

    Important:
    This preserves the sign convention currently used by
    ViewerWidget.

    Rotation order:
        R = Rz @ Ry @ Rx
    """

    rx = np.deg2rad(rotation_x_deg)
    ry = np.deg2rad(rotation_y_deg)
    rz = np.deg2rad(rotation_z_deg)

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ],
        dtype=float,
    )

    # Keep current planning/viewer Y sign convention.
    rotation_y = np.array(
        [
            [cy, 0.0, -sy],
            [0.0, 1.0, 0.0],
            [sy, 0.0, cy],
        ],
        dtype=float,
    )

    rotation_z = np.array(
        [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    return (
        rotation_z
        @ rotation_y
        @ rotation_x
    )


def planning_box_to_scan_geometry(
    fov_box: Mapping[str, float],
    reference_fov_mm: Sequence[float],
) -> ScanGeometry:
    """
    Convert normalized planning geometry into scanner-space
    physical geometry.

    Assumptions for v1:
      - localizer reference volume is centered at isocenter
      - normalized center 0.5 corresponds to scanner origin
      - normalized coordinates are scanner X/Y/Z
      - reference FOV is given in millimeters
    """

    reference_fov_m = (
        np.asarray(
            reference_fov_mm,
            dtype=float,
        )
        / 1000.0
    )

    if reference_fov_m.shape != (3,):
        raise ValueError(
            "reference_fov_mm must contain X/Y/Z"
        )

    if np.any(reference_fov_m <= 0):
        raise ValueError(
            "Reference FOV must be positive"
        )

    center_norm = np.array(
        [
            float(fov_box["center_x"]),
            float(fov_box["center_y"]),
            float(fov_box["center_z"]),
        ],
        dtype=float,
    )

    size_norm = np.array(
        [
            float(fov_box["size_x"]),
            float(fov_box["size_y"]),
            float(fov_box["size_z"]),
        ],
        dtype=float,
    )

    if np.any(center_norm < 0) or np.any(
        center_norm > 1
    ):
        raise ValueError(
            "Planning center must be normalized to 0..1"
        )

    if np.any(size_norm <= 0) or np.any(
        size_norm > 1
    ):
        raise ValueError(
            "Planning size must be normalized to 0..1"
        )

    center_scanner_m = (
        center_norm - 0.5
    ) * reference_fov_m

    fov_local_m = (
        size_norm
        * reference_fov_m
    )

    rotation = planning_euler_to_matrix(
        float(fov_box.get("rotation_x", 0.0)),
        float(fov_box.get("rotation_y", 0.0)),
        float(fov_box.get("rotation_z", 0.0)),
    )

    return ScanGeometry(
        center_scanner_m=center_scanner_m,
        fov_local_m=fov_local_m,
        rotation_local_to_scanner=rotation,
    )