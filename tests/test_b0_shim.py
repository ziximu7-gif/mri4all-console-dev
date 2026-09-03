import numpy as np

from recon.B0Shim.b0_shim import (
    calculate_b0_map,
    fit_first_order_b0,
)


def test_calculate_b0_map():
    delta_te_s = 0.002

    expected_b0_hz = 50.0

    phase_difference = (
        2.0
        * np.pi
        * expected_b0_hz
        * delta_te_s
    )

    image_te1 = np.ones(
        (4, 4, 4),
        dtype=np.complex128,
    )

    image_te2 = np.exp(
        1j * phase_difference
    ) * np.ones(
        (4, 4, 4),
        dtype=np.complex128,
    )

    b0_hz = calculate_b0_map(
        image_te1,
        image_te2,
        delta_te_s,
    )

    assert np.allclose(
        b0_hz,
        expected_b0_hz,
    )


def test_fit_first_order_b0():
    x = np.linspace(
        -0.1,
        0.1,
        8,
    )

    y = np.linspace(
        -0.1,
        0.1,
        7,
    )

    z = np.linspace(
        -0.05,
        0.05,
        6,
    )

    x_m, y_m, z_m = np.meshgrid(
        x,
        y,
        z,
        indexing="ij",
    )

    expected_offset = 12.0
    expected_gx = 80.0
    expected_gy = -40.0
    expected_gz = 25.0

    b0_hz = (
        expected_offset
        + expected_gx * x_m
        + expected_gy * y_m
        + expected_gz * z_m
    )

    roi_mask = np.ones(
        b0_hz.shape,
        dtype=bool,
    )

    result = fit_first_order_b0(
        b0_hz,
        x_m,
        y_m,
        z_m,
        roi_mask,
    )

    assert np.isclose(
        result["offset_hz"],
        expected_offset,
    )

    assert np.isclose(
        result["gradient_x_hz_per_m"],
        expected_gx,
    )

    assert np.isclose(
        result["gradient_y_hz_per_m"],
        expected_gy,
    )

    assert np.isclose(
        result["gradient_z_hz_per_m"],
        expected_gz,
    )

    assert np.isclose(
        result["rms_residual_hz"],
        0.0,
        atol=1e-10,
    )