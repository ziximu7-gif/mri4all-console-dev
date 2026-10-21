import common.logger as logger
import numpy as np
from common.geometry import (cm_to_m,orientation_encoding_matrix,)
from common.geometry import (planning_box_to_scan_geometry,resolve_encoding_geometry,)

log = logger.get_logger()

def resolve_task_planning(
    scan_task,
    orientation,
    readout_direction="Horizontal",
):
    """
    Resolve Localizer planning data stored in
    scan_task.other into scanner geometry and
    logical encoding geometry.
    Returns:
        (scan_geometry, encoding_geometry)
    If no valid planning data exists:
        (None, None)
    """
    geometry_data = scan_task.other.get("geometry")
    if not isinstance(geometry_data,dict,):
        return None, None
    fov_box = geometry_data.get("fov_box")
    reference_fov_mm = (geometry_data.get("reference_fov_mm"))
    if not isinstance(fov_box,dict,):
        return None, None
    if reference_fov_mm is None:
        return None, None
    coordinate_system = (geometry_data.get("coordinate_system"))
    if (coordinate_system!= "scanner_xyz_v1"):
        log.warning("Unsupported planning coordinate ""system: "+ str(coordinate_system))
        return None, None
    scan_geometry = (planning_box_to_scan_geometry(fov_box=fov_box,reference_fov_mm=(reference_fov_mm),))
    encoding_geometry = (resolve_encoding_geometry(scan_geometry=scan_geometry,orientation=orientation,readout_direction=(readout_direction),))
    return (scan_geometry,encoding_geometry,)

def make_initial_gre_geometry(gre_parameters: dict,localizer_task,) -> dict:
    """
    Initialize an independent GRE planning FOV.
    Localizer:
        supplies the scanner-space reference and images.
    GRE:
        owns the actual acquisition FOV.
    Coordinate contract:
        fov_box is normalized scanner XYZ v1.
    """
    if localizer_task.sequence != "localizer":
        raise ValueError(
            "Planning source must be a Localizer"
        )

    reference_fov_m = float(
        cm_to_m(localizer_task.parameters["FOV"])
    )

    gre_fov_m = float(
        cm_to_m(gre_parameters["FOV"])
    )

    if reference_fov_m <= 0 or gre_fov_m <= 0:
        raise ValueError("FOV must be positive")

    # Preserve current legacy 3D GRE default:
    # Read = FOV, Phase = FOV, Third = FOV/2
    fov_logical_m = np.array([
        gre_fov_m,
        gre_fov_m,
        gre_fov_m / 2.0,
    ])

    # logical Read/Phase/Third -> FOV-local XYZ
    encoding = orientation_encoding_matrix(
        gre_parameters["orientation"],
        readout_direction=gre_parameters.get(
            "readout_direction", "Horizontal"
        ),
    )
    # Convert logical dimensions to FOV-local dimensions.
    # Existing E is a signed axis-permutation matrix.
    fov_local_m = (np.abs(encoding) @ fov_logical_m)
    size_norm = fov_local_m / reference_fov_m
    if np.any(size_norm < 0.02) or np.any(
        size_norm > 1.0
    ):
        raise ValueError(
            "GRE FOV is outside the current "
            "normalized planning reference. "
            "Physical-coordinate ROI support is needed "
            "for larger GRE FOVs."
        )
    return {
        "source_localizer_id": localizer_task.id,
        "reference_fov_mm": [
            reference_fov_m * 1000.0
        ] * 3,
        "coordinate_system": "scanner_xyz_v1",
        "reference_origin": "scanner_isocenter",
        "fov_box": {
            "center_x": 0.5,
            "center_y": 0.5,
            "center_z": 0.5,
            "size_x": float(size_norm[0]),
            "size_y": float(size_norm[1]),
            "size_z": float(size_norm[2]),
            "rotation_x": 0.0,
            "rotation_y": 0.0,
            "rotation_z": 0.0,
        },
    }
