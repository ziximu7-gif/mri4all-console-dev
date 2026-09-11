from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

MM_TO_M = 1e-3
CM_TO_M = 1e-2
CM_TO_MM = 10.0

def mm_to_m(value):
    if np.isscalar(value):
        return float(value) * MM_TO_M

    return (
        np.asarray(
            value,
            dtype=float,
        )
        * MM_TO_M
    )

def cm_to_m(value):
    if np.isscalar(value):
        return float(value) * CM_TO_M

    return (
        np.asarray(
            value,
            dtype=float,
        )
        * CM_TO_M
    )


def cm_to_mm(value):
    if np.isscalar(value):
        return float(value) * CM_TO_MM

    return (
        np.asarray(
            value,
            dtype=float,
        )
        * CM_TO_MM
    )
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
        FOV dimensions along the box-local
        X/Y/Z axes [m].

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
    
_AXIS_UNIT_VECTORS = {
    "x": np.array(
        [1.0, 0.0, 0.0],
        dtype=float,
    ),
    "y": np.array(
        [0.0, 1.0, 0.0],
        dtype=float,
    ),
    "z": np.array(
        [0.0, 0.0, 1.0],
        dtype=float,
    ),
}


def orientation_encoding_matrix(
    orientation: str,
    readout_direction: str = "Horizontal",
) -> np.ndarray:

    channels = list(
        orientation_channels(
            orientation
        )
    )

    if (
        readout_direction
        == "Horizontal"
    ):
        pass

    elif (
        readout_direction
        == "Vertical"
    ):
        channels[0], channels[1] = (
            channels[1],
            channels[0],
        )

    else:
        raise ValueError(
            "Unsupported readout "
            "direction: "
            + str(
                readout_direction
            )
        )

    return np.column_stack(
        [
            _AXIS_UNIT_VECTORS[
                channel
            ]
            for channel in channels
        ]
    )



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

def planning_matrix_to_euler(
    rotation,
):
    """
    Inverse of planning_euler_to_matrix().

    Rotation convention:

        R = Rz @ Ry @ Rx

    The Y-axis sign convention must remain
    consistent with planning_euler_to_matrix().
    """

    rotation = np.asarray(
        rotation,
        dtype=float,
    )

    if rotation.shape != (3, 3):
        raise ValueError(
            "Planning rotation matrix "
            "must be 3x3."
        )

    sin_ry = float(
        np.clip(
            rotation[2, 0],
            -1.0,
            1.0,
        )
    )

    ry = np.arcsin(
        sin_ry
    )

    cos_ry = np.cos(
        ry
    )

    if abs(cos_ry) > 1e-8:
        rx = np.arctan2(
            rotation[2, 1],
            rotation[2, 2],
        )

        rz = np.arctan2(
            rotation[1, 0],
            rotation[0, 0],
        )

    else:
        # Gimbal-lock fallback.
        #
        # One degree of freedom is not
        # uniquely recoverable. Keep Rx
        # at zero and absorb it into Rz.
        rx = 0.0

        rz = np.arctan2(
            -rotation[0, 1],
            rotation[1, 1],
        )

    def wrap_degrees(
        angle_rad,
    ):
        angle_deg = np.rad2deg(
            angle_rad
        )

        return float(
            (
                angle_deg
                + 180.0
            )
            % 360.0
            - 180.0
        )

    return (
        wrap_degrees(rx),
        wrap_degrees(ry),
        wrap_degrees(rz),
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

    reference_fov_m = mm_to_m(
        reference_fov_mm
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
@dataclass
class EncodingGeometry:
    """
    Encoding geometry in logical
    read/phase/third coordinates.

    center_logical_m:
        FOV center expressed along logical
        read/phase/third axes [m].

    fov_logical_m:
        FOV along logical read/phase/third
        directions [m].

    logical_to_scanner:
        Maps logical read/phase/third vectors
        into physical scanner X/Y/Z.
    """

    center_logical_m: np.ndarray
    fov_logical_m: np.ndarray
    logical_to_scanner: np.ndarray

    def as_dict(self):
        return {
            "center_logical_m": (
                self.center_logical_m.tolist()
            ),
            "fov_logical_m": (
                self.fov_logical_m.tolist()
            ),
            "logical_to_scanner": (
                self.logical_to_scanner.tolist()
            ),
        }
def resolve_encoding_geometry(
    scan_geometry: ScanGeometry,
    orientation: str,
    readout_direction: str = "Horizontal",
) -> EncodingGeometry:

    encoding_matrix = (
        orientation_encoding_matrix(
            orientation,
            readout_direction=(
                readout_direction
            ),
        )
    )

    box_fov_m = np.asarray(
        scan_geometry.fov_local_m,
        dtype=float,
    )

    if box_fov_m.shape != (3,):
        raise ValueError(
            "Scan FOV must contain X/Y/Z"
        )

    # Convert box X/Y/Z dimensions into
    # logical read/phase/third dimensions.
    fov_logical_m = (
        np.abs(
            encoding_matrix
        ).T
        @ box_fov_m
    )

    

    # Logical encoding axes
    # -> box-local axes
    # -> scanner physical axes
    logical_to_scanner = (
        scan_geometry
        .rotation_local_to_scanner
        @ encoding_matrix
    )

    center_scanner_m = np.asarray(
        scan_geometry.center_scanner_m,
        dtype=float,
    )

    if center_scanner_m.shape != (3,):
        raise ValueError(
            "Scan center must contain X/Y/Z"
        )

    # Scanner coordinates -> logical
    # read/phase/third coordinates.
    #
    # r_scanner = M @ r_logical
    # therefore:
    # r_logical = M.T @ r_scanner
    #
    # logical_to_scanner is orthogonal,
    # so transpose is its inverse.
    center_logical_m = (
        logical_to_scanner.T
        @ center_scanner_m
    )

    return EncodingGeometry(
        center_logical_m=(
            center_logical_m
        ),
        fov_logical_m=fov_logical_m,
        logical_to_scanner=(
            logical_to_scanner
        ),
    )