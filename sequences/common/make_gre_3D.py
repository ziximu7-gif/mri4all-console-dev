import math
from pathlib import Path

import numpy as np
import pypulseq as pp

from sequences.common.get_trajectory import choose_pe_order
from common.constants import *
import common.logger as logger


log = logger.get_logger()


def pypulseq_gre3D(
    inputs,
    check_timing,
    output_file,
    working_folder,
):
    """
    Generate a Cartesian 3D GRE Pulseq sequence.

    This is the shared GRE generator used by imaging sequences
    and, later, by B0 mapping adjustments.
    """

    alpha1 = inputs["FA"]
    alpha1_duration = 80e-6

    TR = inputs["TR"] / 1000
    TE = inputs["TE"] / 1000

    fovx = inputs["FOV"] / 1000
    fovy = inputs["FOV"] / 1000

    # TODO: Expose FOV in Z independently.
    fovz = inputs["FOV"] / 1000 / 2

    Nx = inputs["baseresolution"]
    Ny = inputs["baseresolution"]
    Nz = inputs["slices"]

    dim0 = Ny
    dim1 = Nz

    num_averages = inputs["NSA"]
    orientation = inputs["orientation"]
    BW = inputs["BW"]
    ordering = inputs["ordering"]
    dummyshots = inputs["dummy_shots"]

    pe_order_file = (
        Path(working_folder)
        / mri4all_taskdata.RAWDATA
        / mri4all_scanfiles.PE_ORDER
    )

    adc_phase_file = (
        Path(working_folder)
        / mri4all_taskdata.RAWDATA
        / mri4all_scanfiles.ADC_PHASE
    )

    adc_dwell = 1 / BW
    adc_duration = Nx * adc_dwell

    ch0 = "x"
    ch1 = "y"
    ch2 = "z"

    if orientation == "Axial":
        ch0 = "x"
        ch1 = "y"
        ch2 = "z"
    elif orientation == "Sagittal":
        ch0 = "x"
        ch1 = "z"
        ch2 = "y"
    elif orientation == "Coronal":
        ch0 = "y"
        ch1 = "z"
        ch2 = "x"

    seq = pp.Sequence()
    n_shots = int(Ny * Nz)

    system = pp.Opts(
        max_grad=100,
        grad_unit="mT/m",
        max_slew=4000,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        rf_raster_time=1e-6,
        adc_dead_time=20e-6,
    )

    rf1 = pp.make_block_pulse(
        flip_angle=alpha1 * math.pi / 180,
        duration=alpha1_duration,
        delay=0e-6,
        system=system,
        use="excitation",
    )

    delta_kx = 1 / fovx
    delta_ky = 1 / fovy
    delta_kz = 1 / fovz

    gx = pp.make_trapezoid(
        channel=ch0,
        flat_area=Nx * delta_kx,
        flat_time=adc_duration,
        system=system,
    )

    adc = pp.make_adc(
        num_samples=2 * Nx,
        duration=gx.flat_time,
        delay=gx.rise_time,
        system=system,
    )

    gx_pre = pp.make_trapezoid(
        channel=ch0,
        area=gx.area / 2.0,
        system=system,
    )
    gx_pre.amplitude = -gx_pre.amplitude

    pe_order = choose_pe_order(
        ndims=3,
        npe=[dim0, dim1],
        traj=ordering,
        save_pe_order=True,
        save_path=str(pe_order_file),
    )

    phase_areas0 = (
        pe_order[:, 0]
        * delta_ky
    )

    phase_areas1 = (
        pe_order[:, 1]
        * delta_kz
    )

    gy_pre = pp.make_trapezoid(
        channel=ch1,
        area=1.0 * np.max(phase_areas0),
        system=system,
    )

    gz_pre = pp.make_trapezoid(
        channel=ch2,
        area=-1.0 * np.max(phase_areas1),
        system=system,
    )

    pre_duration = max(
        pp.calc_duration(gy_pre),
        pp.calc_duration(gz_pre),
    )

    pre_duration = max(
        pre_duration,
        pp.calc_duration(gx_pre),
    )

    gx_spoil = pp.make_trapezoid(
        channel=ch0,
        area=Nx * delta_kx,
        system=system,
    )

    if TE == 0:
        tau1 = (
            10
            * seq.grad_raster_time
        )

        TE = (
            tau1
            + 0.5 * pp.calc_duration(rf1)
            + pre_duration
            + 0.5 * pp.calc_duration(gx)
        )
    else:
        tau1 = (
            math.ceil(
                (
                    TE
                    - 0.5 * pp.calc_duration(rf1)
                    - pre_duration
                    - 0.5 * pp.calc_duration(gx)
                )
                / seq.grad_raster_time
            )
            * seq.grad_raster_time
        )

    delay_TR = (
        math.ceil(
            (
                TR
                - 0.5 * pp.calc_duration(rf1)
                - TE
                - 0.5 * pp.calc_duration(gx)
                - pp.calc_duration(gx_spoil)
            )
            / seq.grad_raster_time
        )
        * seq.grad_raster_time
    )

    assert np.all(tau1 >= 0)
    assert np.all(delay_TR >= 0)

    adc_phase = []

    rfspoil_phase = 0
    rfspoil_inc = 0
    rfspoil_incinc = 117.0

    for avg in range(num_averages):
        for i in range(
            n_shots + dummyshots
        ):
            rfspoil_inc = (
                rfspoil_inc
                + rfspoil_incinc
            )

            rfspoil_phase = (
                rfspoil_phase
                + rfspoil_inc
            )

            rfspoil_phase = np.mod(
                rfspoil_phase,
                360.0,
            )

            rfspoil_inc = np.mod(
                rfspoil_inc,
                360.0,
            )

            is_dummyshot = (
                i < dummyshots
            )

            rf1.phase_offset = (
                rfspoil_phase
                / 180
                * math.pi
            )

            seq.add_block(rf1)

            if is_dummyshot:
                pe_idx = 0
            else:
                pe_idx = (
                    i
                    - dummyshots
                )

            gy_pre = pp.make_trapezoid(
                channel=ch1,
                area=-1.0
                * phase_areas0[pe_idx],
                duration=pre_duration,
                system=system,
            )

            gz_pre = pp.make_trapezoid(
                channel=ch2,
                area=-1.0
                * phase_areas1[pe_idx],
                duration=pre_duration,
                system=system,
            )

            seq.add_block(
                gx_pre,
                gy_pre,
                gz_pre,
            )

            seq.add_block(
                pp.make_delay(tau1)
            )

            if is_dummyshot:
                seq.add_block(gx)
            else:
                seq.add_block(
                    gx,
                    adc,
                )

                adc_phase.append(
                    rfspoil_phase
                )

            gy_pre.amplitude = (
                -gy_pre.amplitude
            )

            gz_pre.amplitude = (
                -gz_pre.amplitude
            )

            seq.add_block(
                gx_spoil,
                gy_pre,
                gz_pre,
            )

            seq.add_block(
                pp.make_delay(delay_TR)
            )

    if check_timing:
        ok, error_report = (
            seq.check_timing()
        )

        if ok:
            log.info(
                "Timing check passed successfully"
            )
        else:
            log.info(
                "Timing check failed. "
                "Error listing follows:"
            )

            for error in error_report:
                print(error)

    try:
        np.save(
            adc_phase_file,
            adc_phase,
        )
    except Exception:
        log.error(
            "Could not write file with ADC phase"
        )
        return False

    try:
        seq.write(output_file)
        log.debug(
            "Seq file stored"
        )
    except Exception:
        log.error(
            "Could not write sequence file"
        )
        return False

    return True