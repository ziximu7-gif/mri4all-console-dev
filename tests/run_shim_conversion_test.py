from external.seq.adjustments_acq.shim_conversion import (
    b0_fit_to_shim_delta,
)


def main():
    fit_result = {
        "gradient_x_hz_per_m": 80.0,
        "gradient_y_hz_per_m": -40.0,
        "gradient_z_hz_per_m": 25.0,
    }

    calibration_matrix = [
        [1000.0, 0.0, 0.0],
        [0.0, 800.0, 0.0],
        [0.0, 0.0, 500.0],
    ]

    delta = b0_fit_to_shim_delta(
        fit_result,
        calibration_matrix,
    )

    print(delta)

    assert abs(
        delta["shim_x"] - (-0.08)
    ) < 1e-12

    assert abs(
        delta["shim_y"] - 0.05
    ) < 1e-12

    assert abs(
        delta["shim_z"] - (-0.05)
    ) < 1e-12

    print(
        "Shim conversion test passed."
    )


if __name__ == "__main__":
    main()