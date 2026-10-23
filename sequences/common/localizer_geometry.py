"""Resolve a three-plane Localizer prescription without generating a sequence.

This module is independent of the UI models and the acquisition runtime.
All physical positions and dimensions are in meters. The persisted
``next_localizer`` schema is scanner_xyz_v1 (normalized scanner X/Y/Z).

Phase A only parses and validates. It does NOT authorize oblique scans;
the existing legacy-acquisition guard must remain active until the
Pulseq, simulation, reconstruction and image-plane paths are ready.
"""

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from common.geometry import (
    EncodingGeometry,
    ScanGeometry,
    cm_to_m,
    orientation_encoding_matrix,
    planning_box_to_scan_geometry,
    resolve_encoding_geometry,
)


LOCALIZER_ORIENTATIONS = ("Axial", "Coronal", "Sagittal")
_REQUIRED_BOX_FIELDS = (
    "center_x", "center_y", "center_z",
    "size_x", "size_y", "size_z",
    "rotation_x", "rotation_y", "rotation_z",
)


@dataclass(frozen=True)
class LocalizerSlicePrescription:
    """Geometry for one read/phase/slice acquisition, not a scan permit."""

    orientation: str
    scan_geometry: ScanGeometry
    encoding_geometry: EncodingGeometry
    source_localizer_id: Optional[str] = None


def _positive_scalar(value, field):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a positive finite number") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be a positive finite number")
    return result


def _task_dimensions(parameters):
    if not isinstance(parameters, Mapping):
        raise ValueError("Localizer parameters must be a mapping")
    try:
        fov_m = _positive_scalar(cm_to_m(parameters["FOV"]), "FOV")
        thickness_m = _positive_scalar(
            float(parameters.get("Slice_Thickness", 10.0)) * 1e-3,
            "Slice_Thickness",
        )
    except KeyError as exc:
        raise ValueError(f"Missing Localizer parameter: {exc.args[0]}") from exc
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ValueError) and "positive finite" in str(exc):
            raise
        raise ValueError("Invalid Localizer FOV or Slice_Thickness") from exc
    return {
        "Axial": np.array([fov_m, fov_m, thickness_m], dtype=float),
        "Coronal": np.array([fov_m, thickness_m, fov_m], dtype=float),
        "Sagittal": np.array([thickness_m, fov_m, fov_m], dtype=float),
    }


def _validate_matrices(orientation, scan_geometry, encoding_geometry):
    rotation = np.asarray(scan_geometry.rotation_local_to_scanner, dtype=float)
    matrix = np.asarray(encoding_geometry.logical_to_scanner, dtype=float)
    for label, value in (("rotation", rotation), ("encoding", matrix)):
        if value.shape != (3, 3) or not np.all(np.isfinite(value)):
            raise ValueError(f"{orientation}: invalid {label} matrix")
        if not np.allclose(value.T @ value, np.eye(3), rtol=0, atol=1e-10):
            raise ValueError(f"{orientation}: non-orthogonal {label} matrix")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10):
        raise ValueError(f"{orientation}: rotation matrix must be proper")
    # The legacy Coronal encoding basis has determinant -1 by convention.
    if not np.isclose(abs(np.linalg.det(matrix)), 1.0, atol=1e-10):
        raise ValueError(f"{orientation}: invalid encoding handedness")
    if not np.allclose(
        matrix,
        rotation @ orientation_encoding_matrix(orientation),
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError(f"{orientation}: inconsistent encoding axes")


def _validated_plan(plan, source_task):
    if not isinstance(plan, Mapping):
        raise ValueError("next_localizer must be an object")
    if type(plan.get("version")) is not int or plan["version"] != 1:
        raise ValueError("Unsupported next_localizer version")
    if plan.get("coordinate_system") != "scanner_xyz_v1":
        raise ValueError("Unsupported next_localizer coordinate_system")

    source_id = plan.get("source_localizer_id")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("next_localizer source_localizer_id is missing")

    reference = plan.get("reference_fov_mm")
    if not isinstance(reference, (list, tuple)) or len(reference) != 3:
        raise ValueError("reference_fov_mm must contain three positive values")
    try:
        reference_mm = np.asarray(reference, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("reference_fov_mm contains non-numeric values") from exc
    if not np.all(np.isfinite(reference_mm)) or np.any(reference_mm <= 0):
        raise ValueError("reference_fov_mm must be positive and finite")
    # The v1 reference is a cube defined by the source Localizer's scalar FOV.
    if not np.allclose(reference_mm, reference_mm[0], atol=1e-6, rtol=0):
        raise ValueError("scanner_xyz_v1 requires an isotropic reference FOV")

    if source_task is not None:
        if getattr(source_task, "sequence", None) != "localizer":
            raise ValueError("Reference scan must be a Localizer")
        if source_id != str(getattr(source_task, "id", "")):
            raise ValueError("source_localizer_id does not match reference scan")
        try:
            actual_reference_mm = (
                _positive_scalar(cm_to_m(source_task.parameters["FOV"]),
                                 "Reference Localizer FOV") * 1000.0
            )
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError("Reference Localizer FOV is unavailable") from exc
        if not np.allclose(reference_mm, actual_reference_mm, atol=1e-6, rtol=0):
            raise ValueError("reference_fov_mm does not match reference scan")

    boxes = plan.get("boxes")
    if not isinstance(boxes, Mapping) or set(boxes) != set(LOCALIZER_ORIENTATIONS):
        raise ValueError("next_localizer must contain exactly three named boxes")
    return source_id, reference_mm, boxes


def resolve_localizer_prescriptions(scan_task, source_task=None):
    """Resolve three independent Localizer boxes into physical scan geometry.

    Missing ``next_localizer`` uses the old centered orthogonal defaults.
    Explicit but invalid planning data raises ValueError; never fall back.

    This function does not apply a scan-capability restriction. Call
    ``validate_rotation_only_prescriptions`` for the planned first
    acquisition milestone. The existing legacy guard remains authoritative.
    """
    if getattr(scan_task, "sequence", None) != "localizer":
        raise ValueError("The scan task must be a Localizer")
    dimensions = _task_dimensions(scan_task.parameters)
    other = getattr(scan_task, "other", None)
    if not isinstance(other, Mapping):
        raise ValueError("scan_task.other must be a mapping")
    has_plan = "next_localizer" in other

    source_id = None
    if has_plan:
        source_id, reference_mm, boxes = _validated_plan(
            other["next_localizer"], source_task
        )

    prescriptions = {}
    for orientation in LOCALIZER_ORIENTATIONS:
        if has_plan:
            box = boxes[orientation]
            if not isinstance(box, Mapping):
                raise ValueError(f"{orientation}: box must be an object")
            if not all(field in box for field in _REQUIRED_BOX_FIELDS):
                missing = [field for field in _REQUIRED_BOX_FIELDS if field not in box]
                raise ValueError(f"{orientation}: missing box fields: {missing}")
            try:
                values = np.asarray([box[field] for field in _REQUIRED_BOX_FIELDS],
                                    dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{orientation}: box contains non-numeric values") from exc
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{orientation}: box values must be finite")
            if np.any(values[3:6] <= 0):
                raise ValueError(f"{orientation}: box dimensions must be positive")
            scan_geometry = planning_box_to_scan_geometry(
                fov_box={field: float(box[field]) for field in _REQUIRED_BOX_FIELDS},
                reference_fov_mm=reference_mm,
                allow_outside_reference=True,
            )
        else:
            scan_geometry = ScanGeometry(
                center_scanner_m=np.zeros(3, dtype=float),
                fov_local_m=dimensions[orientation].copy(),
                rotation_local_to_scanner=np.eye(3, dtype=float),
            )

        encoding = resolve_encoding_geometry(
            scan_geometry=scan_geometry,
            orientation=orientation,
            readout_direction="Horizontal",
        )
        _validate_matrices(orientation, scan_geometry, encoding)
        prescriptions[orientation] = LocalizerSlicePrescription(
            orientation=orientation,
            scan_geometry=scan_geometry,
            encoding_geometry=encoding,
            source_localizer_id=source_id,
        )
    return prescriptions


def validate_rotation_only_prescriptions(
    prescriptions,
    parameters,
    center_tolerance_m=1e-6,
    size_tolerance_m=1e-6,
):
    """Reject geometry outside the next *rotation-only* acquisition milestone.

    Translation, independent in-plane FOV and slice thickness remain
    unsupported, even though the general resolver represents them.
    Passing this validator is not permission to scan: the existing
    sequence-level legacy guard remains in place until Phase B.
    """
    expected = _task_dimensions(parameters)
    if not isinstance(prescriptions, Mapping) or set(prescriptions) != set(expected):
        raise ValueError("Exactly three Localizer prescriptions are required")
    for label, tolerance in (("center", center_tolerance_m),
                             ("size", size_tolerance_m)):
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError(f"Invalid {label} tolerance")
    for orientation in LOCALIZER_ORIENTATIONS:
        prescription = prescriptions[orientation]
        if (not isinstance(prescription, LocalizerSlicePrescription)
                or prescription.orientation != orientation):
            raise ValueError(f"{orientation}: invalid prescription")
        scan = prescription.scan_geometry
        center = np.asarray(scan.center_scanner_m, dtype=float)
        dims = np.asarray(scan.fov_local_m, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError(f"{orientation}: invalid scan center")
        if dims.shape != (3,) or not np.all(np.isfinite(dims)):
            raise ValueError(f"{orientation}: invalid scan dimensions")
        _validate_matrices(orientation, scan, prescription.encoding_geometry)
        if not np.allclose(center, 0, atol=center_tolerance_m, rtol=0):
            raise ValueError(f"{orientation}: off-isocenter translation not supported yet")
        if not np.allclose(dims, expected[orientation],
                           atol=size_tolerance_m, rtol=0):
            raise ValueError(
                f"{orientation}: independent FOV/thickness not supported yet"
            )
    return True
