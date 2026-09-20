import numpy as np
import pytest

from common.geometry import (
    ImagePlaneGeometry,
    ScanGeometry,
    cm_to_m,
    cm_to_mm,
    localizer_image_plane_geometry,
    mm_to_m,
    orientation_channels,
    orientation_encoding_matrix,
    planning_box_to_scan_geometry,
    planning_euler_to_matrix,
    resolve_encoding_geometry,
)
from common.geometry import (
    planning_euler_to_matrix,
    planning_matrix_to_euler,
)
def test_mm_to_m():
    np.testing.assert_allclose(
        mm_to_m(200.0),
        0.2,
    )

def test_cm_to_m():
    np.testing.assert_allclose(
        cm_to_m(20.0),
        0.2,
    )


def test_cm_to_mm():
    np.testing.assert_allclose(
        cm_to_mm(20.0),
        200.0,
    )

def test_orientation_channels():
    assert (
        orientation_channels(
            "Axial"
        )
        == (
            "x",
            "y",
            "z",
        )
    )

    assert (
        orientation_channels(
            "Coronal"
        )
        == (
            "x",
            "z",
            "y",
        )
    )

    assert (
        orientation_channels(
            "Sagittal"
        )
        == (
            "y",
            "z",
            "x",
        )
    )


def test_centered_planning_box():
    geometry = (
        planning_box_to_scan_geometry(
            fov_box={
                "center_x": 0.5,
                "center_y": 0.5,
                "center_z": 0.5,
                "size_x": 0.5,
                "size_y": 0.5,
                "size_z": 0.5,
                "rotation_x": 0.0,
                "rotation_y": 0.0,
                "rotation_z": 0.0,
            },
            reference_fov_mm=[
                200.0,
                200.0,
                200.0,
            ],
        )
    )

    np.testing.assert_allclose(
        geometry.center_scanner_m,
        [
            0.0,
            0.0,
            0.0,
        ],
    )

    np.testing.assert_allclose(
        geometry.fov_local_m,
        [
            0.1,
            0.1,
            0.1,
        ],
    )


def test_shifted_planning_box():
    geometry = (
        planning_box_to_scan_geometry(
            fov_box={
                "center_x": 0.6,
                "center_y": 0.4,
                "center_z": 0.5,
                "size_x": 0.5,
                "size_y": 0.4,
                "size_z": 0.3,
            },
            reference_fov_mm=[
                200.0,
                200.0,
                200.0,
            ],
        )
    )

    np.testing.assert_allclose(
        geometry.center_scanner_m,
        [
            0.02,
            -0.02,
            0.0,
        ],
    )

    np.testing.assert_allclose(
        geometry.fov_local_m,
        [
            0.1,
            0.08,
            0.06,
        ],
    )


def test_rotation_matrix():
    rotation = (
        planning_euler_to_matrix(
            10.0,
            20.0,
            30.0,
        )
    )

    np.testing.assert_allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.linalg.det(rotation),
        1.0,
        atol=1e-12,
    )
def test_orientation_encoding_matrix():

    axial = (
        orientation_encoding_matrix(
            "Axial"
        )
    )

    np.testing.assert_allclose(
        axial,
        np.eye(3),
    )

    coronal = (
        orientation_encoding_matrix(
            "Coronal"
        )
    )

    np.testing.assert_allclose(
        coronal,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    )

    sagittal = (
        orientation_encoding_matrix(
            "Sagittal"
        )
    )

    np.testing.assert_allclose(
        sagittal,
        np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    )
    
def test_coronal_encoding_geometry():

    scan_geometry = ScanGeometry(
        center_scanner_m=np.array(
            [
                0.01,
                0.02,
                0.03,
            ]
        ),
        fov_local_m=np.array(
            [
                0.12,
                0.10,
                0.08,
            ]
        ),
        rotation_local_to_scanner=(
            np.eye(3)
        ),
    )

    encoding = (
        resolve_encoding_geometry(
            scan_geometry=scan_geometry,
            orientation="Coronal",
        )
    )

    np.testing.assert_allclose(
        encoding.fov_logical_m,
        [
            0.12,
            0.08,
            0.10,
        ],
    )
    np.testing.assert_allclose(
        encoding.center_logical_m,
        [
            0.01,
            0.03,
            0.02,
        ],
    )
def test_rotated_encoding_center():
    rotation = (
        planning_euler_to_matrix(
            0.0,
            0.0,
            90.0,
        )
    )

    scan_geometry = ScanGeometry(
        center_scanner_m=np.array(
            [
                0.01,
                0.0,
                0.0,
            ]
        ),
        fov_local_m=np.array(
            [
                0.12,
                0.10,
                0.08,
            ]
        ),
        rotation_local_to_scanner=rotation,
    )

    encoding = (
        resolve_encoding_geometry(
            scan_geometry=scan_geometry,
            orientation="Axial",
        )
    )

    np.testing.assert_allclose(
        encoding.center_logical_m,
        [
            0.0,
            -0.01,
            0.0,
        ],
        atol=1e-12,
    )
def test_planning_rotation_roundtrip():
    original = (
        planning_euler_to_matrix(
            17.0,
            -23.0,
            41.0,
        )
    )

    angles = (
        planning_matrix_to_euler(
            original
        )
    )

    reconstructed = (
        planning_euler_to_matrix(
            *angles
        )
    )

    np.testing.assert_allclose(
        reconstructed,
        original,
        atol=1e-12,
    )


# =====================================================================
# Localizer scanner-space image plane geometry
# =====================================================================


def test_localizer_image_plane_axes():
    """All three axes come from orientation_channels(), not a cross
    product."""
    expected = {
        "Axial": (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ),
        "Coronal": (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
        ),
        "Sagittal": (
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
        ),
    }

    for orientation, (
        horizontal,
        vertical,
        normal,
    ) in expected.items():
        plane = localizer_image_plane_geometry(
            orientation=orientation,
            fov_m=0.2,
            rows=96,
            columns=96,
        )

        np.testing.assert_allclose(
            plane.horizontal_scanner,
            horizontal,
        )
        np.testing.assert_allclose(
            plane.vertical_scanner,
            vertical,
        )
        np.testing.assert_allclose(
            plane.normal_scanner,
            normal,
        )
        np.testing.assert_allclose(
            plane.center_scanner_m,
            np.zeros(3),
        )


def test_coronal_plane_does_not_infer_normal_from_cross_product():
    """Locks the repository convention: Coronal is mirrored, so
    cross(u, v) == -normal. Code must not derive the normal from the
    cross product."""
    plane = localizer_image_plane_geometry(
        orientation="Coronal",
        fov_m=0.2,
        rows=96,
        columns=96,
    )

    cross = np.cross(
        plane.horizontal_scanner,
        plane.vertical_scanner,
    )

    np.testing.assert_allclose(
        cross,
        -plane.normal_scanner,
        atol=1e-12,
    )


def test_image_plane_scanner_plane_roundtrip():
    """plane -> scanner -> plane must be an identity for every
    orientation."""
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.03, -0.02, 0.0],
            [-0.07, 0.04, 0.01],
        ]
    )

    for orientation in (
        "Axial",
        "Coronal",
        "Sagittal",
    ):
        plane = localizer_image_plane_geometry(
            orientation=orientation,
            fov_m=0.2,
            rows=96,
            columns=96,
        )

        scanner = plane.plane_to_scanner_m(
            coordinates
        )

        recovered = plane.scanner_to_plane_m(
            scanner
        )

        np.testing.assert_allclose(
            recovered,
            coordinates,
            atol=1e-12,
        )


def test_localizer_image_center_maps_to_isocenter():
    """The displayed image center is the scanner isocenter for every
    orientation."""
    for orientation in (
        "Axial",
        "Coronal",
        "Sagittal",
    ):
        plane = localizer_image_plane_geometry(
            orientation=orientation,
            fov_m=0.2,
            rows=96,
            columns=96,
        )

        scanner = plane.image_xy_to_scanner_m(
            np.array([48.0, 48.0])
        )

        np.testing.assert_allclose(
            scanner,
            np.zeros(3),
            atol=1e-12,
        )


def test_axial_image_edges_map_to_physical_fov():
    """The image rectangle corners span the physical FOV."""
    plane = localizer_image_plane_geometry(
        orientation="Axial",
        fov_m=0.2,
        rows=96,
        columns=96,
    )

    corner_lo = plane.image_xy_to_scanner_m(
        np.array([0.0, 0.0])
    )

    corner_hi = plane.image_xy_to_scanner_m(
        np.array([96.0, 96.0])
    )

    np.testing.assert_allclose(
        corner_lo,
        np.array([-0.1, -0.1, 0.0]),
        atol=1e-12,
    )

    np.testing.assert_allclose(
        corner_hi,
        np.array([0.1, 0.1, 0.0]),
        atol=1e-12,
    )


def test_scanner_to_image_xy_ignores_normal_distance():
    """scanner_to_image_xy is orthographic: the normal component does
    not affect the image coordinates."""
    plane = localizer_image_plane_geometry(
        orientation="Axial",
        fov_m=0.2,
        rows=96,
        columns=96,
    )

    on_plane = plane.scanner_to_image_xy(
        np.array([0.02, -0.03, 0.0])
    )

    off_plane = plane.scanner_to_image_xy(
        np.array([0.02, -0.03, 0.07])
    )

    np.testing.assert_allclose(
        on_plane,
        off_plane,
        atol=1e-12,
    )


def _valid_plane_kwargs(**overrides):
    kwargs = {
        "center_scanner_m": np.zeros(3),
        "horizontal_scanner": np.array([1.0, 0.0, 0.0]),
        "vertical_scanner": np.array([0.0, 1.0, 0.0]),
        "normal_scanner": np.array([0.0, 0.0, 1.0]),
        "width_m": 0.2,
        "height_m": 0.2,
        "rows": 96,
        "columns": 96,
    }
    kwargs.update(overrides)
    return kwargs


def test_image_plane_rejects_non_unit_direction():
    with pytest.raises(ValueError):
        ImagePlaneGeometry(
            **_valid_plane_kwargs(
                horizontal_scanner=np.array(
                    [2.0, 0.0, 0.0]
                )
            )
        )


def test_image_plane_rejects_non_orthogonal_axes():
    with pytest.raises(ValueError):
        ImagePlaneGeometry(
            **_valid_plane_kwargs(
                vertical_scanner=np.array(
                    [0.6, 0.8, 0.0]
                )
            )
        )


def test_image_plane_rejects_non_positive_size():
    with pytest.raises(ValueError):
        ImagePlaneGeometry(
            **_valid_plane_kwargs(width_m=0.0)
        )

    with pytest.raises(ValueError):
        ImagePlaneGeometry(
            **_valid_plane_kwargs(height_m=-0.2)
        )


def test_image_plane_rejects_non_positive_matrix():
    with pytest.raises(ValueError):
        ImagePlaneGeometry(
            **_valid_plane_kwargs(rows=0)
        )

    with pytest.raises(ValueError):
        ImagePlaneGeometry(
            **_valid_plane_kwargs(columns=-1)
        )


def test_image_plane_basis_scanner_columns():
    """basis_scanner columns are [horizontal, vertical, normal]."""
    plane = localizer_image_plane_geometry(
        orientation="Coronal",
        fov_m=0.2,
        rows=96,
        columns=96,
    )

    basis = plane.basis_scanner

    assert basis.shape == (3, 3)

    np.testing.assert_allclose(
        basis[:, 0],
        plane.horizontal_scanner,
    )
    np.testing.assert_allclose(
        basis[:, 1],
        plane.vertical_scanner,
    )
    np.testing.assert_allclose(
        basis[:, 2],
        plane.normal_scanner,
    )
    
def test_image_plane_rejects_non_finite_size():
    with pytest.raises(ValueError):
        ImagePlaneGeometry(
            **_valid_plane_kwargs(
                width_m=np.nan
            )
        )

    with pytest.raises(ValueError):
        ImagePlaneGeometry(
            **_valid_plane_kwargs(
                height_m=np.inf
            )
        )