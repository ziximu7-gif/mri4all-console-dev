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


_BOX_CORNER_SIGNS = np.array(
    [
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1],
    ],
    dtype=float,
)
_BOX_EDGES = np.array(
    [
        [0, 1],
        [0, 2],
        [0, 4],

        [1, 3],
        [1, 5],

        [2, 3],
        [2, 6],

        [3, 7],

        [4, 5],
        [4, 6],

        [5, 7],
        [6, 7],
    ],
    dtype=int,
)


def scan_geometry_box_corners_scanner_m(
    scan_geometry: ScanGeometry,
) -> np.ndarray:
    """
    Return the eight corners of a ScanGeometry box
    in scanner X/Y/Z coordinates [m].

    Output shape:
        (8, 3)
    """

    center = np.asarray(
        scan_geometry.center_scanner_m,
        dtype=float,
    )

    sizes = np.asarray(
        scan_geometry.fov_local_m,
        dtype=float,
    )

    rotation = np.asarray(
        scan_geometry.rotation_local_to_scanner,
        dtype=float,
    )

    if center.shape != (3,):
        raise ValueError(
            "Scan center must have shape (3,)"
        )

    if sizes.shape != (3,):
        raise ValueError(
            "Scan FOV must have shape (3,)"
        )

    if rotation.shape != (3, 3):
        raise ValueError(
            "Scan rotation must have shape (3, 3)"
        )

    if (
        not np.all(np.isfinite(center))
        or not np.all(np.isfinite(sizes))
        or not np.all(np.isfinite(rotation))
    ):
        raise ValueError(
            "Scan geometry must contain finite values"
        )

    if np.any(sizes <= 0):
        raise ValueError(
            "Scan FOV dimensions must be positive"
        )

    half_sizes = (
        0.5 * sizes
    )

    local_corners = (
        _BOX_CORNER_SIGNS
        * half_sizes
    )

    scanner_corners = (
        center[None, :]
        + local_corners
        @ rotation.T
    )

    return scanner_corners


def scan_geometry_box_edges_scanner_m(
    scan_geometry: ScanGeometry,
) -> np.ndarray:
    """
    Return the twelve edges of a ScanGeometry box in
    scanner X/Y/Z coordinates [m].

    Output shape:
        (12, 2, 3)

    dimensions:
        edge
        endpoint
        scanner XYZ

    This is the single source of truth for box edge
    topology. Callers must not re-declare the corner
    index table.
    """

    corners = (
        scan_geometry_box_corners_scanner_m(
            scan_geometry
        )
    )

    return corners[_BOX_EDGES]


def planning_box_corners_in_encoding_m(
    box,
    reference_fov_mm,
    acquisition_center_scanner_m,
    logical_to_scanner,
):
    """
    Convert a planning box (FOV or Shim) into
    encoding-native logical corner coordinates
    [m], expressed relative to the acquisition
    FOV center.

    The reconstruction volume center is already
    the planned FOV center, so the Shim overlay
    uses coordinates relative to the FOV center,
    not absolute scanner-isocenter coordinates.

    Row-vector form of:

        r_logical = M.T @ r_scanner
    """

    box_geometry = (
        planning_box_to_scan_geometry(
            fov_box=box,
            reference_fov_mm=(
                reference_fov_mm
            ),
        )
    )

    half_sizes = (
        0.5
        * np.asarray(
            box_geometry.fov_local_m,
            dtype=float,
        )
    )

    local_corners = (
        _BOX_CORNER_SIGNS
        * half_sizes
    )

    scanner_corners = (
        box_geometry
        .center_scanner_m[
            None,
            :
        ]
        +
        local_corners
        @ box_geometry
        .rotation_local_to_scanner.T
    )

    logical_to_scanner = np.asarray(
        logical_to_scanner,
        dtype=float,
    )

    acquisition_center_scanner_m = (
        np.asarray(
            acquisition_center_scanner_m,
            dtype=float,
        )
    )

    relative_scanner = (
        scanner_corners
        - acquisition_center_scanner_m[
            None,
            :
        ]
    )

    # Row-vector form of:
    #
    # r_logical = M.T @ r_scanner
    logical_corners = (
        relative_scanner
        @ logical_to_scanner
    )

    return logical_corners


@dataclass(frozen=True)
class ImagePlaneGeometry:
    """
    Physical geometry of one displayed 2D image center plane.

    All vectors are expressed in scanner X/Y/Z coordinates.

    center_scanner_m:
        Scanner-space center of the image plane [m].

    horizontal_scanner:
        Unit scanner-space vector corresponding to increasing
        viewer/image x.

    vertical_scanner:
        Unit scanner-space vector corresponding to increasing
        viewer/image y.

    normal_scanner:
        Unit scanner-space slice normal.

    width_m / height_m:
        Physical image FOV along horizontal / vertical axes [m].

    columns / rows:
        Displayed image matrix dimensions.

    This represents the CENTER PLANE of the acquired finite slice.
    Slice thickness is acquisition information and is deliberately
    not part of this plane geometry.
    """

    center_scanner_m: np.ndarray
    horizontal_scanner: np.ndarray
    vertical_scanner: np.ndarray
    normal_scanner: np.ndarray
    width_m: float
    height_m: float
    rows: int
    columns: int

    def __post_init__(self):

        for name in (
            "center_scanner_m",
            "horizontal_scanner",
            "vertical_scanner",
            "normal_scanner",
        ):
            value = np.asarray(
                getattr(self, name),
                dtype=float,
            )

            if value.shape != (3,):
                raise ValueError(
                    f"{name} must have shape (3,)"
                )

            if not np.all(
                np.isfinite(value)
            ):
                raise ValueError(
                    f"{name} must contain only "
                    "finite values"
                )

            object.__setattr__(
                self,
                name,
                value,
            )

        for name in (
            "horizontal_scanner",
            "vertical_scanner",
            "normal_scanner",
        ):
            vector = getattr(self, name)

            if not np.isclose(
                float(np.linalg.norm(vector)),
                1.0,
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError(
                    f"{name} must be a unit vector"
                )

        # Pairwise orthogonality. Handedness is deliberately NOT
        # enforced: the repository convention defines the third
        # orientation channel explicitly, and Coronal is mirrored
        # relative to cross(horizontal, vertical).
        for first, second in (
            (
                "horizontal_scanner",
                "vertical_scanner",
            ),
            (
                "horizontal_scanner",
                "normal_scanner",
            ),
            (
                "vertical_scanner",
                "normal_scanner",
            ),
        ):
            dot = float(
                np.dot(
                    getattr(self, first),
                    getattr(self, second),
                )
            )

            if not np.isclose(
                dot,
                0.0,
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError(
                    f"{first} and {second} "
                    "must be orthogonal"
                )

        if (
            not np.isfinite(self.width_m)
            or self.width_m <= 0
        ):
            raise ValueError(
                "Image width must be finite and positive"
            )

        if (
            not np.isfinite(self.height_m)
            or self.height_m <= 0
        ):
            raise ValueError(
                "Image height must be finite and positive"
            )

        if self.rows <= 0:
            raise ValueError(
                "Image rows must be positive"
            )

        if self.columns <= 0:
            raise ValueError(
                "Image columns must be positive"
            )

    @property
    def basis_scanner(self):
        return np.column_stack(
            [
                self.horizontal_scanner,
                self.vertical_scanner,
                self.normal_scanner,
            ]
        )

    def scanner_to_plane_m(
        self,
        points_scanner_m,
    ):
        """
        Convert scanner X/Y/Z points into this plane's
        [horizontal, vertical, normal] coordinates [m].
        """

        points = np.asarray(
            points_scanner_m,
            dtype=float,
        )

        if (
            points.ndim == 0
            or points.shape[-1] != 3
        ):
            raise ValueError(
                "Points must have a final "
                "dimension of 3"
            )

        relative = (
            points
            - self.center_scanner_m
        )

        return (
            relative
            @ self.basis_scanner
        )

    def plane_to_scanner_m(
        self,
        plane_coordinates_m,
    ):
        """
        Inverse of scanner_to_plane_m(): convert
        [horizontal, vertical, normal] coordinates [m]
        into scanner X/Y/Z points.
        """

        plane_coordinates = np.asarray(
            plane_coordinates_m,
            dtype=float,
        )

        if (
            plane_coordinates.ndim == 0
            or plane_coordinates.shape[-1] != 3
        ):
            raise ValueError(
                "Plane coordinates must have a "
                "final dimension of 3"
            )

        return (
            self.center_scanner_m
            + plane_coordinates
            @ self.basis_scanner.T
        )

    def scanner_to_image_xy(
        self,
        points_scanner_m,
    ):
        """
        Orthographic scanner -> image mapping.

        The normal component is ignored: a point need not lie on
        the center plane for its image x/y to be well defined.
        Coordinates are NOT clipped to the image bounds.
        """

        plane_coordinates = (
            self.scanner_to_plane_m(
                points_scanner_m
            )
        )

        horizontal_m = plane_coordinates[..., 0]
        vertical_m = plane_coordinates[..., 1]

        image_x = (
            horizontal_m / self.width_m
            + 0.5
        ) * self.columns

        image_y = (
            vertical_m / self.height_m
            + 0.5
        ) * self.rows

        return np.stack(
            [image_x, image_y],
            axis=-1,
        )

    def image_xy_to_scanner_m(
        self,
        image_xy,
    ):
        """
        Inverse of scanner_to_image_xy() for points on the
        CENTER PLANE (normal coordinate 0).
        """

        image_xy = np.asarray(
            image_xy,
            dtype=float,
        )

        if (
            image_xy.ndim == 0
            or image_xy.shape[-1] != 2
        ):
            raise ValueError(
                "Image coordinates must have a "
                "final dimension of 2"
            )

        horizontal_m = (
            image_xy[..., 0] / self.columns
            - 0.5
        ) * self.width_m

        vertical_m = (
            image_xy[..., 1] / self.rows
            - 0.5
        ) * self.height_m

        normal_m = np.zeros_like(
            horizontal_m
        )

        plane_coordinates = np.stack(
            [
                horizontal_m,
                vertical_m,
                normal_m,
            ],
            axis=-1,
        )

        return self.plane_to_scanner_m(
            plane_coordinates
        )


def localizer_image_plane_geometry(
    orientation,
    fov_m,
    rows,
    columns,
):
    """
    Scanner-space center plane of one Localizer image.

    The Localizer is scanner-base and centered at isocenter, so the
    center is the origin. All three axes come from
    orientation_channels(): read -> horizontal, phase -> vertical,
    third channel -> normal. The normal is therefore taken from the
    explicit repository convention and NOT from
    cross(horizontal, vertical).

    The Localizer FOV is square, so width and height both equal
    fov_m. Slice thickness is acquisition information and is not
    part of this plane geometry.
    """

    read_axis, phase_axis, slice_axis = (
        orientation_channels(
            orientation
        )
    )

    return ImagePlaneGeometry(
        center_scanner_m=np.zeros(
            3,
            dtype=float,
        ),
        horizontal_scanner=(
            _AXIS_UNIT_VECTORS[
                read_axis
            ].copy()
        ),
        vertical_scanner=(
            _AXIS_UNIT_VECTORS[
                phase_axis
            ].copy()
        ),
        normal_scanner=(
            _AXIS_UNIT_VECTORS[
                slice_axis
            ].copy()
        ),
        width_m=float(fov_m),
        height_m=float(fov_m),
        rows=int(rows),
        columns=int(columns),
    )

def _append_unique_point(
    points,
    candidate,
    tolerance_m,
):
    """
    Append candidate only when no existing point is
    within tolerance_m.
    """

    candidate = np.asarray(
        candidate,
        dtype=float,
    )

    for existing in points:
        if (
            float(
                np.linalg.norm(
                    existing - candidate
                )
            )
            <= tolerance_m
        ):
            return

    points.append(
        candidate.copy()
    )


def intersect_box_with_plane_scanner_m(
    scan_geometry: ScanGeometry,
    image_plane: ImagePlaneGeometry,
    tolerance_m: float = 1e-9,
) -> np.ndarray:
    """
    Intersect a scanner-space oriented box with an image
    center plane.

    Returns unique intersection vertices in scanner X/Y/Z
    coordinates [m].

    For three or more points, vertices are ordered around
    the polygon perimeter in the image plane's horizontal /
    vertical coordinates.

    Possible outputs include:

        (0, 3): no intersection
        (1, 3): tangent at one vertex
        (2, 3): tangent/intersection segment
        (N, 3): polygon, normally N = 3..6
    """

    if (
        not np.isfinite(tolerance_m)
        or tolerance_m <= 0
    ):
        raise ValueError(
            "Plane intersection tolerance must be "
            "finite and positive"
        )

    corners = (
        scan_geometry_box_corners_scanner_m(
            scan_geometry
        )
    )

    plane_coordinates = (
        image_plane.scanner_to_plane_m(
            corners
        )
    )

    signed_distances = (
        plane_coordinates[:, 2]
    )

    points = []

    for start_index, end_index in _BOX_EDGES:

        p0 = corners[start_index]
        p1 = corners[end_index]

        d0 = float(
            signed_distances[start_index]
        )

        d1 = float(
            signed_distances[end_index]
        )

        on0 = abs(d0) <= tolerance_m
        on1 = abs(d1) <= tolerance_m

        # Entire box edge lies in the image plane.
        if on0 and on1:
            _append_unique_point(
                points,
                p0,
                tolerance_m,
            )

            _append_unique_point(
                points,
                p1,
                tolerance_m,
            )

            continue

        # Exactly one endpoint lies in the plane.
        if on0:
            _append_unique_point(
                points,
                p0,
                tolerance_m,
            )
            continue

        if on1:
            _append_unique_point(
                points,
                p1,
                tolerance_m,
            )
            continue

        # Opposite signs mean the edge crosses the plane.
        if (
            (d0 < 0.0 and d1 > 0.0)
            or
            (d0 > 0.0 and d1 < 0.0)
        ):
            t = (
                d0
                / (d0 - d1)
            )

            point = (
                p0
                + t * (p1 - p0)
            )

            _append_unique_point(
                points,
                point,
                tolerance_m,
            )

    if len(points) == 0:
        return np.empty(
            (0, 3),
            dtype=float,
        )

    result = np.asarray(
        points,
        dtype=float,
    )

    # For a polygon, order vertices around its centroid.
    #
    # Do this entirely in the explicit image-plane u/v basis;
    # do not make any handedness assumption.
    if result.shape[0] >= 3:

        plane_points = (
            image_plane.scanner_to_plane_m(
                result
            )
        )

        uv = plane_points[:, :2]

        centroid = np.mean(
            uv,
            axis=0,
        )

        angles = np.arctan2(
            uv[:, 1] - centroid[1],
            uv[:, 0] - centroid[0],
        )

        order = np.argsort(
            angles
        )

        result = result[order]

    return result


def project_box_edges_to_plane_scanner_m(
    scan_geometry: ScanGeometry,
    image_plane: ImagePlaneGeometry,
) -> np.ndarray:
    """
    Orthographically project all twelve 3D box edges onto
    the image center plane.

    No hidden-edge removal is performed.

    Output shape:
        (12, 2, 3)

    dimensions:
        edge
        endpoint
        scanner XYZ
    """

    corners = (
        scan_geometry_box_corners_scanner_m(
            scan_geometry
        )
    )

    plane_coordinates = (
        image_plane.scanner_to_plane_m(
            corners
        )
    )

    signed_distances = (
        plane_coordinates[:, 2]
    )

    projected_corners = (
        corners
        - signed_distances[:, None]
        * image_plane.normal_scanner[None, :]
    )

    return projected_corners[
        _BOX_EDGES
    ]