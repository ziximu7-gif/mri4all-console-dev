"""Phase A: no Pulseq generation or hardware/simulation execution."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from common.geometry import orientation_encoding_matrix, planning_euler_to_matrix
from sequences.common.localizer_geometry import (
    LOCALIZER_ORIENTATIONS,
    resolve_localizer_prescriptions,
    validate_rotation_only_prescriptions,
)


def task(plan_marker=None, *, fov_cm=20.0, thickness_mm=10.0):
    other = {} if plan_marker is None else {"next_localizer": plan_marker}
    return SimpleNamespace(
        sequence="localizer", other=other,
        parameters={"FOV": fov_cm, "Slice_Thickness": thickness_mm},
    )


def plan(*, reference_mm=200.0, fov_cm=20.0, thickness_mm=10.0):
    dims = {
        "Axial": (fov_cm * 10, fov_cm * 10, thickness_mm),
        "Coronal": (fov_cm * 10, thickness_mm, fov_cm * 10),
        "Sagittal": (thickness_mm, fov_cm * 10, fov_cm * 10),
    }
    boxes = {}
    for orientation, sizes in dims.items():
        box = dict(center_x=0.5, center_y=0.5, center_z=0.5,
                   size_x=sizes[0] / reference_mm,
                   size_y=sizes[1] / reference_mm,
                   size_z=sizes[2] / reference_mm,
                   rotation_x=0.0, rotation_y=0.0, rotation_z=0.0)
        boxes[orientation] = box
    return {"version": 1, "coordinate_system": "scanner_xyz_v1",
            "source_localizer_id": "localizer-001",
            "reference_fov_mm": [reference_mm] * 3, "boxes": boxes}


@pytest.mark.parametrize("orientation,expected", [
    ("Axial", [0.2, 0.2, 0.010]),
    ("Coronal", [0.2, 0.2, 0.010]),
    ("Sagittal", [0.2, 0.2, 0.010]),
])
def test_legacy_default_fov_and_direction(orientation, expected):
    result = resolve_localizer_prescriptions(task())
    assert tuple(result) == LOCALIZER_ORIENTATIONS
    p = result[orientation]
    np.testing.assert_allclose(p.encoding_geometry.fov_logical_m, expected)
    np.testing.assert_allclose(p.encoding_geometry.center_logical_m, 0)
    np.testing.assert_allclose(p.encoding_geometry.logical_to_scanner,
                               orientation_encoding_matrix(orientation))
    assert validate_rotation_only_prescriptions(result, task().parameters)


def test_coronal_mirrored_encoding_is_legal():
    result = resolve_localizer_prescriptions(task(plan()))
    m = result["Coronal"].encoding_geometry.logical_to_scanner
    assert np.linalg.det(m) == pytest.approx(-1.0)
    assert validate_rotation_only_prescriptions(result, task().parameters)


def test_manual_z_rotation_changes_only_axial():
    d = plan()
    d["boxes"]["Axial"]["rotation_z"] = 90.0
    resolved = resolve_localizer_prescriptions(task(d))
    np.testing.assert_allclose(
        resolved["Axial"].encoding_geometry.logical_to_scanner,
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12
    )
    for orientation in ("Coronal", "Sagittal"):
        np.testing.assert_allclose(
            resolved[orientation].encoding_geometry.logical_to_scanner,
            orientation_encoding_matrix(orientation)
        )
    assert validate_rotation_only_prescriptions(resolved, task().parameters)


def test_manual_negative_y_rotation_coronal():
    d = plan()
    d["boxes"]["Coronal"]["rotation_y"] = -90.0
    resolved = resolve_localizer_prescriptions(task(d))
    # Current planning convention: R_y(-90) takes scanner +X to -Z.
    matrix = resolved["Coronal"].encoding_geometry.logical_to_scanner
    np.testing.assert_allclose(matrix[:, 0], [0, 0, -1], atol=1e-12)
    assert validate_rotation_only_prescriptions(resolved, task().parameters)


def test_manual_plus_x_rotation_sagittal():
    d = plan()
    d["boxes"]["Sagittal"]["rotation_x"] = 90.0
    m = resolve_localizer_prescriptions(task(d))["Sagittal"].encoding_geometry.logical_to_scanner
    # Sagittal read was scanner +Y; after +90 about +X, it becomes +Z.
    np.testing.assert_allclose(m[:, 0], [0, 0, 1], atol=1e-12)


def test_combined_rotation_uses_repo_euler_order():
    d = plan()
    values = (27.0, -38.0, 63.0)
    for name, angle in zip(("rotation_x", "rotation_y", "rotation_z"), values):
        d["boxes"]["Axial"][name] = angle
    result = resolve_localizer_prescriptions(task(d))
    matrix = result["Axial"].encoding_geometry.logical_to_scanner
    np.testing.assert_allclose(matrix, planning_euler_to_matrix(*values), atol=1e-12)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
    assert validate_rotation_only_prescriptions(result, task().parameters)


@pytest.mark.parametrize("angle", (-90.0, 90.0, 179.999, -179.999))
def test_rotation_boundaries_remain_finite(angle):
    d = plan()
    d["boxes"]["Axial"]["rotation_y"] = angle
    result = resolve_localizer_prescriptions(task(d))
    assert np.isfinite(result["Axial"].encoding_geometry.logical_to_scanner).all()
    assert validate_rotation_only_prescriptions(result, task().parameters)


def test_different_reference_fov_has_same_physical_prescription():
    d = plan(reference_mm=240)
    p = resolve_localizer_prescriptions(task(d))
    np.testing.assert_allclose(p["Axial"].encoding_geometry.fov_logical_m,
                               [0.2, 0.2, 0.01])
    assert validate_rotation_only_prescriptions(p, task().parameters)


def test_optional_source_reference_is_checked():
    t = task(plan(reference_mm=240))
    source = SimpleNamespace(id="localizer-001", sequence="localizer", parameters={"FOV": 24})
    assert resolve_localizer_prescriptions(t, source)["Axial"].source_localizer_id == "localizer-001"
    source.id = "wrong-source"
    with pytest.raises(ValueError, match="source_localizer_id"):
        resolve_localizer_prescriptions(t, source)
    source.id = "localizer-001"
    source.parameters["FOV"] = 20
    with pytest.raises(ValueError, match="reference_fov_mm"):
        resolve_localizer_prescriptions(t, source)


def test_translation_is_parsed_but_out_of_scope():
    d = plan()
    d["boxes"]["Coronal"]["center_x"] = 0.6
    result = resolve_localizer_prescriptions(task(d))
    np.testing.assert_allclose(result["Coronal"].scan_geometry.center_scanner_m,
                               [0.02, 0, 0], atol=1e-12)
    with pytest.raises(ValueError, match="Coronal.*off-isocenter"):
        validate_rotation_only_prescriptions(result, task().parameters)


@pytest.mark.parametrize("orientation,field,delta", [
    ("Axial", "size_x", 0.15),
    ("Coronal", "size_z", 0.15),
    ("Sagittal", "size_x", 0.01),
])
def test_independent_dimensions_parse_but_reject(orientation, field, delta):
    d = plan()
    d["boxes"][orientation][field] += delta
    r = resolve_localizer_prescriptions(task(d))
    with pytest.raises(ValueError, match="FOV/thickness"):
        validate_rotation_only_prescriptions(r, task().parameters)


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p.update(version=2), "version"),
    (lambda p: p.update(coordinate_system="unknown"), "coordinate_system"),
    (lambda p: p["boxes"].pop("Axial"), "three named boxes"),
    (lambda p: p["boxes"].update(Extra={}), "three named boxes"),
    (lambda p: p["boxes"]["Axial"].pop("center_x"), "missing box fields"),
    (lambda p: p["boxes"]["Axial"].update(size_z=0), "dimensions"),
    (lambda p: p["boxes"]["Axial"].update(rotation_y=float("nan")), "finite"),
    (lambda p: p["boxes"]["Axial"].update(rotation_z=float("inf")), "finite"),
    (lambda p: p.update(reference_fov_mm=[200, 0, 200]), "positive"),
    (lambda p: p.update(reference_fov_mm=[200, 201, 200]), "isotropic"),
    (lambda p: p.update(source_localizer_id=""), "source_localizer_id"),
])
def test_bad_plans_fail_closed(mutation, match):
    d = plan()
    mutation(d)
    with pytest.raises(ValueError, match=match):
        resolve_localizer_prescriptions(task(d))


def test_explicit_null_plan_does_not_fall_back_to_legacy():
    t = task()
    t.other["next_localizer"] = None
    with pytest.raises(ValueError, match="next_localizer"):
        resolve_localizer_prescriptions(t)


def test_default_values_validate_even_if_plan_is_absent():
    with pytest.raises(ValueError, match="positive finite"):
        resolve_localizer_prescriptions(task(fov_cm=float("nan")))
    with pytest.raises(ValueError, match="positive finite"):
        resolve_localizer_prescriptions(task(thickness_mm=-1))


def test_parser_does_not_mutate_task_plan():
    d = plan()
    d["boxes"]["Sagittal"]["rotation_z"] = 44
    before = deepcopy(d)
    resolve_localizer_prescriptions(task(d))
    assert d == before


def test_three_different_rotations_do_not_couple():
    d = plan()
    d["boxes"]["Axial"]["rotation_x"] = 12
    d["boxes"]["Coronal"]["rotation_y"] = -23
    d["boxes"]["Sagittal"]["rotation_z"] = 34
    r = resolve_localizer_prescriptions(task(d))
    assert all(np.linalg.norm(r[o].scan_geometry.rotation_local_to_scanner - np.eye(3)) > 0.1
               for o in LOCALIZER_ORIENTATIONS)
    assert validate_rotation_only_prescriptions(r, task().parameters)


def test_legacy_task_without_slice_thickness_uses_sequence_default():
    old = task()
    old.parameters.pop("Slice_Thickness")
    resolved = resolve_localizer_prescriptions(old)
    np.testing.assert_allclose(
        resolved["Axial"].encoding_geometry.fov_logical_m,
        [0.2, 0.2, 0.010],
    )
