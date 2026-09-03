import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)
import json
import tempfile
from pathlib import Path

import numpy as np

from common.constants import (
    mri4all_scanfiles,
    mri4all_taskdata,
)
from common.types import ScanTask
from services.recon.reconstruction import (
    run_reconstruction_b0_map,
)


def main():
    nx = 8
    ny = 6
    nz = 4

    fov_x_m = 0.20
    fov_y_m = 0.18
    fov_z_m = 0.10

    te1_ms = 10.0
    te2_ms = 12.0

    delta_te_s = (
        te2_ms - te1_ms
    ) / 1000.0

    expected_offset = 12.0
    expected_gx = 80.0
    expected_gy = -40.0
    expected_gz = 25.0

    x_axis = (
        (np.arange(nx) + 0.5) / nx
        - 0.5
    ) * fov_x_m

    y_axis = (
        (np.arange(ny) + 0.5) / ny
        - 0.5
    ) * fov_y_m

    z_axis = (
        (np.arange(nz) + 0.5) / nz
        - 0.5
    ) * fov_z_m

    x_m, y_m, z_m = np.meshgrid(
        x_axis,
        y_axis,
        z_axis,
        indexing="ij",
    )

    b0_expected = (
        expected_offset
        + expected_gx * x_m
        + expected_gy * y_m
        + expected_gz * z_m
    )

    image_te1 = np.ones(
        (nx, ny, nz),
        dtype=np.complex128,
    )

    image_te2 = np.exp(
        1j
        * 2.0
        * np.pi
        * b0_expected
        * delta_te_s
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

    order = []

    for slice_index in range(nz):
        for phase_index in range(ny):
            order.append(
                [
                    ny
                    - int(ny / 2)
                    - phase_index,
                    nz
                    - int(nz / 2)
                    - slice_index,
                ]
            )

    order = np.asarray(order)

    num_lines = len(order)

    raw = np.zeros(
        (
            num_lines,
            2,
            nx,
        ),
        dtype=np.complex128,
    )

    center_pe = (
        ny - int(ny / 2)
    )

    center_slice = (
        nz - int(nz / 2)
    )

    for line_index, line in enumerate(order):
        phase_index = (
            center_pe
            - int(line[0])
        ) % ny

        slice_index = (
            center_slice
            - int(line[1])
        ) % nz

        raw[line_index, 0, :] = (
            kspace_te1[
                :,
                phase_index,
                slice_index,
            ]
        )

        raw[line_index, 1, :] = (
            kspace_te2[
                :,
                phase_index,
                slice_index,
            ]
        )

    adc_phases = np.zeros(
        num_lines * 2,
        dtype=float,
    )

    scan_task = ScanTask()

    scan_task.processing.dim = 3
    scan_task.processing.dim_size = (
        f"{nz},{ny},{nx}"
    )

    # Keep this synthetic test simple.
    scan_task.processing.oversampling_read = 0

    scan_task.parameters = {
        "TE1": te1_ms,
        "TE2": te2_ms,
    }

    scan_task.other["planning"] = {
        "shim_box": {
            "center_x": 0.5,
            "center_y": 0.5,
            "center_z": 0.5,
            "size_x": 1.0,
            "size_y": 1.0,
            "size_z": 1.0,
            "rotation_x": 0.0,
            "rotation_y": 0.0,
            "rotation_z": 0.0,
        }
    }

    scan_task.other["b0_geometry"] = {
        "fov_x_m": fov_x_m,
        "fov_y_m": fov_y_m,
        "fov_z_m": fov_z_m,
    }

    with tempfile.TemporaryDirectory() as folder:
        raw_folder = (
            Path(folder)
            / mri4all_taskdata.RAWDATA
        )

        raw_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            raw_folder
            / mri4all_scanfiles.PE_ORDER,
            order,
        )

        np.save(
            raw_folder
            / mri4all_scanfiles.ADC_PHASE,
            adc_phases,
        )

        np.save(
            raw_folder
            / mri4all_scanfiles.RAWDATA,
            raw.reshape(-1),
        )

        success = run_reconstruction_b0_map(
            folder,
            scan_task,
        )

        assert success

        fit = scan_task.other["b0_fit"]

        print("B0 fit:")
        print(json.dumps(fit, indent=4))

        print()
        print("Expected:")
        print("offset =", expected_offset)
        print("gx =", expected_gx)
        print("gy =", expected_gy)
        print("gz =", expected_gz)

        assert np.isclose(
            fit["offset_hz"],
            expected_offset,
            atol=1e-6,
        )

        assert np.isclose(
            fit["gradient_x_hz_per_m"],
            expected_gx,
            atol=1e-6,
        )

        assert np.isclose(
            fit["gradient_y_hz_per_m"],
            expected_gy,
            atol=1e-6,
        )

        assert np.isclose(
            fit["gradient_z_hz_per_m"],
            expected_gz,
            atol=1e-6,
        )

        b0_file = (
            raw_folder
            / "b0_map.npy"
        )

        assert b0_file.exists()

        # Also make sure b0_fit can later be written
        # into scan.json.
        json.dumps(scan_task.other)

    print()
    print("B0 reconstruction pipeline test passed.")


if __name__ == "__main__":
    main()