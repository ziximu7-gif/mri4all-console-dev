import common.logger as logger
import numpy as np
from common.geometry import (cm_to_m,orientation_encoding_matrix,)
from common.geometry import (planning_box_to_scan_geometry,resolve_encoding_geometry,)
from common.geometry import (ScanGeometry,)

from services.ui.spatialbox import (
    Box3D,
    PlanningState,
    PlanningRegion,
    PlanningScene,
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

def make_localizer_scan_boxes(parameters):
    """
    Build the three physical Localizer acquisition
    slabs in scanner XYZ coordinates.

    This describes the currently implemented
    isocenter-centered, orthogonal Localizer.
    It does not change the Pulseq sequence.
    """
    fov_m = float(cm_to_m(parameters["FOV"]))
    thickness_m = (
        float(parameters["Slice_Thickness"]) / 1000.0
    )

    if fov_m <= 0 or thickness_m <= 0:
        raise ValueError("Invalid Localizer geometry")

    dimensions = {
        "axial": [fov_m, fov_m, thickness_m],
        "coronal": [fov_m, thickness_m, fov_m],
        "sagittal": [thickness_m, fov_m, fov_m],
    }

    return {
        name: ScanGeometry(
            center_scanner_m=np.zeros(3),
            fov_local_m=np.asarray(
                size, dtype=float
            ),
            rotation_local_to_scanner=np.eye(3),
        )
        for name, size in dimensions.items()
    }


def make_next_localizer_scene(
    target_task,
    source_task,
    shared_shim_box,
) -> PlanningScene:
    """
    Construct three independently editable future
    Localizer slabs.

    The source Localizer provides reference images.
    The target Localizer owns the proposed geometry.
    """
    if source_task.sequence != "localizer":
        raise ValueError("Invalid reference sequence")

    reference_m = float(
        cm_to_m(source_task.parameters["FOV"])
    )
    target_m = float(
        cm_to_m(target_task.parameters["FOV"])
    )
    thickness_m = (
        float(target_task.parameters["Slice_Thickness"])
        / 1000.0
    )

    if min(reference_m, target_m, thickness_m) <= 0:
        raise ValueError("FOV/thickness must be positive")

    # Scanner X/Y/Z dimensions, normalized against
    # the source Localizer's reference FOV.
    f = target_m / reference_m
    t = thickness_m / reference_m

    defaults = {
        "Axial": (f, f, t),
        "Coronal": (f, t, f),
        "Sagittal": (t, f, f),
    }

    saved = target_task.other.get(
        "next_localizer", {}
    )
    if not isinstance(saved, dict):
        saved = {}

    if saved:
        if (
            saved.get("source_localizer_id")
            != source_task.id
        ):
            raise ValueError(
                "Saved planning uses another Localizer"
            )

        expected_reference_mm = reference_m * 1000.0
        stored_reference_mm = saved.get(
            "reference_fov_mm"
        )
        if (
            not isinstance(stored_reference_mm, list)
            or len(stored_reference_mm) != 3
            or not np.allclose(
                stored_reference_mm,
                [expected_reference_mm] * 3,
                rtol=0,
                atol=1e-6,
            )
        ):
            raise ValueError(
                "Planning reference FOV mismatch"
            )

    stored_boxes = saved.get("boxes", {})
    if not isinstance(stored_boxes, dict):
        stored_boxes = {}

    regions = {}

    for name, dimensions in defaults.items():
        if name in stored_boxes:
            box = Box3D(**stored_boxes[name])
        else:
            box = Box3D(
                center_x=0.5,
                center_y=0.5,
                center_z=0.5,
                size_x=dimensions[0],
                size_y=dimensions[1],
                size_z=dimensions[2],
            )

        box.allow_outside_reference = True
        box.clamp()

        regions[name] = PlanningRegion(
            id=name,
            state=PlanningState(
                fov_box=box,
                shim_box=shared_shim_box,
            ),
        )

    return PlanningScene(
        reference_fov_m=reference_m,
        source_localizer_id=source_task.id,
        regions=regions,
        active_id="Axial",
    )


def localizer_plan_matches_legacy_acquisition(scan_task):
    """
    True only if the proposed three slabs are identical
    to the currently implemented orthogonal Localizer.
    """
    plan = scan_task.other.get("next_localizer")

    if plan is None:
        return True

    if not isinstance(plan, dict):
        return False

    boxes = plan.get("boxes")
    reference = plan.get("reference_fov_mm")

    if (
        not isinstance(boxes, dict)
        or not isinstance(reference, list)
        or len(reference) != 3
    ):
        return False

    ref = np.asarray(reference, dtype=float)

    if (
        not np.all(np.isfinite(ref))
        or np.any(ref <= 0)
    ):
        return False

    fov_m = float(cm_to_m(scan_task.parameters["FOV"]))
    thickness_m = (
        float(scan_task.parameters["Slice_Thickness"])
        / 1000.0
    )

    if fov_m <= 0 or thickness_m <= 0:
        return False

    expected = {
        "Axial": [fov_m, fov_m, thickness_m],
        "Coronal": [fov_m, thickness_m, fov_m],
        "Sagittal": [thickness_m, fov_m, fov_m],
    }

    for name, physical_sizes in expected.items():
        box = boxes.get(name)
        if not isinstance(box, dict):
            return False

        try:
            center = np.array([
                box["center_x"],
                box["center_y"],
                box["center_z"],
            ], dtype=float)

            sizes = np.array([
                box["size_x"],
                box["size_y"],
                box["size_z"],
            ], dtype=float)

            angles = np.array([
                box["rotation_x"],
                box["rotation_y"],
                box["rotation_z"],
            ], dtype=float)
        except (KeyError, TypeError, ValueError):
            return False

        expected_sizes = (
            np.asarray(physical_sizes) /
            (ref / 1000.0)
        )

        if not np.allclose(
            center, 0.5, rtol=0, atol=1e-6
        ):
            return False

        if not np.allclose(
            sizes, expected_sizes,
            rtol=0, atol=1e-6,
        ):
            return False

        if not np.allclose(
            angles, 0.0, rtol=0, atol=1e-6
        ):
            return False

    return True
