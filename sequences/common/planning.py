import common.logger as logger

from common.geometry import (
    planning_box_to_scan_geometry,
    resolve_encoding_geometry,
)


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

    geometry_data = scan_task.other.get(
        "geometry"
    )

    if not isinstance(
        geometry_data,
        dict,
    ):
        return None, None

    fov_box = geometry_data.get(
        "fov_box"
    )

    reference_fov_mm = (
        geometry_data.get(
            "reference_fov_mm"
        )
    )

    if not isinstance(
        fov_box,
        dict,
    ):
        return None, None

    if reference_fov_mm is None:
        return None, None

    coordinate_system = (
        geometry_data.get(
            "coordinate_system"
        )
    )

    if (
        coordinate_system
        != "scanner_xyz_v1"
    ):
        log.warning(
            "Unsupported planning coordinate "
            "system: "
            + str(coordinate_system)
        )
        return None, None

    scan_geometry = (
        planning_box_to_scan_geometry(
            fov_box=fov_box,
            reference_fov_mm=(
                reference_fov_mm
            ),
        )
    )

    encoding_geometry = (
        resolve_encoding_geometry(
            scan_geometry=scan_geometry,
            orientation=orientation,
            readout_direction=(
                readout_direction
            ),
        )
    )

    return (
        scan_geometry,
        encoding_geometry,
    )