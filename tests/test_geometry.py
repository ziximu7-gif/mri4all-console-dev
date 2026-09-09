import numpy as np

from common.geometry import (
    orientation_channels,
    planning_box_to_scan_geometry,
    planning_euler_to_matrix,
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