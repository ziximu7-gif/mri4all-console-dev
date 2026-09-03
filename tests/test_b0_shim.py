import numpy as np

from recon.B0Shim.b0_shim import (
    calculate_b0_map,
    fit_first_order_b0,
    create_shim_roi_mask,
    create_physical_coordinate_grids,
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
def test_dual_echo_cartesian_reconstruction():
    from recon.recon_utils.cartesian3d import (
        reconstruct_cartesian_3d_complex,
    )

    num_readout = 8
    num_phase = 4
    num_slices = 3

    dims = [
        str(num_slices),
        str(num_phase),
        str(num_readout),
    ]

    order = []

    for slice_index in range(num_slices):
        for phase_index in range(num_phase):
            order.append(
                [
                    num_phase
                    - int(num_phase / 2)
                    - phase_index,
                    num_slices
                    - int(num_slices / 2)
                    - slice_index,
                ]
            )

    order = np.asarray(order)

    num_lines = len(order)

    adc_phases = np.zeros(
        num_lines * 2,
        dtype=float,
    )

    image_te1 = np.ones(
        (
            num_readout,
            num_phase,
            num_slices,
        ),
        dtype=np.complex128,
    )

    delta_te_s = 0.002
    expected_b0_hz = 40.0

    phase_difference = (
        2.0
        * np.pi
        * expected_b0_hz
        * delta_te_s
    )

    image_te2 = (
        image_te1
        * np.exp(
            1j * phase_difference
        )
    )

    kspace_te1 = np.fft.ifftshift(
        np.fft.ifftn(
            np.fft.ifftshift(
                image_te1
            )
        )
    )

    kspace_te2 = np.fft.ifftshift(
        np.fft.ifftn(
            np.fft.ifftshift(
                image_te2
            )
        )
    )

    raw = np.zeros(
        (
            num_lines,
            2,
            num_readout,
        ),
        dtype=np.complex128,
    )

    center_pe = (
        num_phase
        - int(num_phase / 2)
    )

    center_slice = (
        num_slices
        - int(num_slices / 2)
    )

    for line_index, line in enumerate(order):
        phase_index = (
            center_pe
            - int(line[0])
        ) % num_phase

        slice_index = (
            center_slice
            - int(line[1])
        ) % num_slices

        raw[
            line_index,
            0,
            :,
        ] = kspace_te1[
            :,
            phase_index,
            slice_index,
        ]

        raw[
            line_index,
            1,
            :,
        ] = kspace_te2[
            :,
            phase_index,
            slice_index,
        ]

    images, _ = (
        reconstruct_cartesian_3d_complex(
            raw=raw.reshape(-1),
            order=order,
            adc_phases=adc_phases,
            dims=dims,
            echo_count=2,
            oversampling_read=0,
        )
    )

    b0_hz = calculate_b0_map(
        images[0],
        images[1],
        delta_te_s,
    )

    assert np.allclose(
        b0_hz,
        expected_b0_hz,
        atol=1e-6,
    )
def test_create_shim_roi_mask():
    shim_box = {
        "center_x": 0.5,
        "center_y": 0.5,
        "center_z": 0.5,
        "size_x": 0.5,
        "size_y": 0.5,
        "size_z": 0.5,
        "rotation_x": 0.0,
        "rotation_y": 0.0,
        "rotation_z": 0.0,
    }

    mask = create_shim_roi_mask(
        volume_shape=(8, 8, 8),
        shim_box=shim_box,
    )

    assert mask.shape == (8, 8, 8)

    assert mask.dtype == bool

    assert np.count_nonzero(mask) == 64
def test_rotated_shim_roi_not_supported_yet():
    shim_box = {
        "center_x": 0.5,
        "center_y": 0.5,
        "center_z": 0.5,
        "size_x": 0.5,
        "size_y": 0.5,
        "size_z": 0.5,
        "rotation_x": 0.0,
        "rotation_y": 0.0,
        "rotation_z": 20.0,
    }

    try:
        create_shim_roi_mask(
            volume_shape=(8, 8, 8),
            shim_box=shim_box,
        )

    except NotImplementedError:
        return

    raise AssertionError(
        "Rotated shim ROI should not "
        "be accepted yet."
    )
def test_create_physical_coordinate_grids():
    x_m, y_m, z_m = (
        create_physical_coordinate_grids(
            volume_shape=(4, 4, 2),
            fov_x_m=0.2,
            fov_y_m=0.2,
            fov_z_m=0.1,
        )
    )

    assert x_m.shape == (4, 4, 2)
    assert y_m.shape == (4, 4, 2)
    assert z_m.shape == (4, 4, 2)

    assert np.allclose(
        x_m[:, 0, 0],
        [
            -0.075,
            -0.025,
            0.025,
            0.075,
        ],
    )

    assert np.allclose(
        y_m[0, :, 0],
        [
            -0.075,
            -0.025,
            0.025,
            0.075,
        ],
    )

    assert np.allclose(
        z_m[0, 0, :],
        [
            -0.025,
            0.025,
        ],
    )
