import numpy as np

from common.geometry import (
    ScanGeometry,
    cm_to_m,
    cm_to_mm,
    mm_to_m,
    orientation_channels,
    orientation_encoding_matrix,
    planning_box_to_scan_geometry,
    planning_euler_to_matrix,
    resolve_encoding_geometry,
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