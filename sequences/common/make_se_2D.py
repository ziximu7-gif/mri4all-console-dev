import math
import numpy as np

import pypulseq as pp  # type: ignore
import external.seq.adjustments_acq.config as cfg

from sequences.common import view_traj
import common.logger as logger
from common.geometry import (
    cm_to_m,
    mm_to_m,
    orientation_channels,
)

log = logger.get_logger()


def pypulseq_se2D(
    inputs=None, check_timing=True, output_file="", output_folder="") -> bool:
    if not output_file:
        log.error("No output file specified")
        return False

    # ======
    # DEFAULTS FROM CONFIG FILE              TODO: MOVE DEFAULTS TO UI
    # ======
    LARMOR_FREQ = cfg.LARMOR_FREQ
    RF_MAX = cfg.RF_MAX
    RF_PI2_FRACTION = cfg.RF_PI2_FRACTION

    # fov = 140e-3  # Define FOV and resolution
    # Nx = 70
    alpha1 = 90  # flip angle
    alpha1_duration = 100e-6  # pulse duration
    alpha2 = 180  # refocusing flip angle
    alpha2_duration = 100e-6  # pulse duration
    #num_averages = 1
    prephaser_duration = 3e-3  # TODO: Need to define this behind the scenes and optimze

    TR = inputs["TR"] / 1000
    TE = inputs["TE"] / 1000
    num_averages = inputs["NSA"]
    Orientation = inputs["Orientation"]
    fov = cm_to_m(
        inputs["FOV"]
    )
    slice_thickness_mm = inputs.get(
        "Slice_Thickness"
    )

    slice_selective = (
        slice_thickness_mm is not None
    )

    if slice_selective:
        slice_thickness = mm_to_m(
            float(slice_thickness_mm)
        )

        if slice_thickness <= 0:
            raise ValueError(
                "Slice thickness must be positive"
            )
    else:
        slice_thickness = None

    Nx = inputs["Base_Resolution"]
    BW = inputs["BW"]
    visualize = inputs["view_traj"]
    # Trajectory = inputs['Trajectory']     TODO
    # PE_Ordering = inputs['PE_Ordering']   TODO
    # PF = inputs['PF']                     TODO

    Ny = Nx
    #BW = 32e3
    adc_dwell = 1 / BW
    adc_duration = Nx * adc_dwell  # 6.4e-3

    ch0, ch1, ch2 = (
        orientation_channels(
            Orientation
        )
    )

    # ======
    # INITIATE SEQUENCE
    # ======

    seq = pp.Sequence()

    # ======
    # SET SYSTEM CONFIG TODO --> ?
    # ======

    # system = pp.Opts(
    #     max_grad=12,
    #     grad_unit="mT/m",
    #     max_slew=25,
    #     slew_unit="T/m/s",
    #     rf_ringdown_time=20e-6,
    #     rf_dead_time=100e-6,
    #     rf_raster_time=1e-6,
    #     adc_dead_time=20e-6,
    # )

    system = pp.Opts(
        max_grad=100,
        grad_unit="mT/m",
        max_slew=4000,
        slew_unit="T/m/s",
        rf_ringdown_time=100e-6,
        rf_dead_time=100e-6,
        rf_raster_time=1e-6,
        adc_dead_time=10e-6,
    )

    # ======
    # CREATE EVENTS
    # ======
    # Create excitation and refocusing RF pulses.
    # Localizer can use slice-selective excitation,
    # while the legacy path remains non-selective.
    if slice_selective:
        rf1, g_slice, g_slice_rephase = (
            pp.make_sinc_pulse(
                flip_angle=(
                    alpha1
                    * math.pi
                    / 180
                ),
                duration=2e-3,
                slice_thickness=(
                    slice_thickness
                ),
                apodization=0.5,
                time_bw_product=4,
                return_gz=True,
                system=system,
                use="excitation",
            )
        )

        # make_sinc_pulse() creates its
        # slice gradients on z internally.
        # Localizer slice selection must use
        # the orientation's third channel.
        g_slice.channel = ch2
        g_slice_rephase.channel = ch2

    else:
        rf1 = pp.make_block_pulse(
            flip_angle=(
                alpha1
                * math.pi
                / 180
            ),
            duration=alpha1_duration,
            delay=100e-6,
            system=system,
            use="excitation",
        )

        g_slice = None
        g_slice_rephase = None

    rf2 = pp.make_block_pulse(
        flip_angle=alpha2 * math.pi / 180,
        duration=alpha2_duration,
        delay=100e-6,
        phase_offset=math.pi / 2,
        system=system,
        use="refocusing",
    )

    # Define other gradients and ADC events
    delta_k = 1 / fov
    gx = pp.make_trapezoid(
        channel=ch0, flat_area=Nx * delta_k, flat_time=adc_duration, system=system
    )
    adc = pp.make_adc(
        num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system
    )
    gx_pre = pp.make_trapezoid(
        channel=ch0, area=gx.area / 2, duration=prephaser_duration, system=system
    )

    phase_areas = -(np.arange(Ny) - Ny / 2) * delta_k

    # Gradient spoiling -TODO: Need to see if this is really required based on data
    gx_spoil = pp.make_trapezoid(channel=ch0, area=2 * Nx * delta_k, system=system)

    # ======
    # CALCULATE DELAYS
    # ======
    if slice_selective:
        raster = seq.grad_raster_time

        def ceil_to_raster(value):
            return (
                math.ceil(
                    value / raster
                )
                * raster
            )

        excitation_block_duration = (
            pp.calc_duration(
                rf1,
                g_slice,
            )
        )

        excitation_rf_center = (
            rf1.delay
            + pp.calc_rf_center(rf1)[0]
        )

        prephase_block_duration = (
            pp.calc_duration(
                gx_pre,
                g_slice_rephase,
            )
        )

        refocusing_block_duration = (
            pp.calc_duration(rf2)
        )

        refocusing_rf_center = (
            rf2.delay
            + pp.calc_rf_center(rf2)[0]
        )

        # make_adc() only defines `duration` when `dwell`
        # is passed; this sequence passes `duration`, so use
        # the equivalent sampling window num_samples * dwell.
        echo_center_in_readout = (
            adc.delay
            + 0.5
            * adc.num_samples
            * adc.dwell
        )

        tau1_raw = (
            TE / 2.0
            - (
                excitation_block_duration
                - excitation_rf_center
            )
            - prephase_block_duration
            - refocusing_rf_center
        )

        tau2_raw = (
            TE / 2.0
            - (
                refocusing_block_duration
                - refocusing_rf_center
            )
            - echo_center_in_readout
        )

        if tau1_raw < 0 or tau2_raw < 0:
            raise ValueError(
                "TE is too short for "
                "slice-selective Localizer"
            )

        tau1 = ceil_to_raster(
            tau1_raw
        )

        tau2 = ceil_to_raster(
            tau2_raw
        )

        post_block_duration = max(
            pp.calc_duration(gx_spoil),
            pp.calc_duration(gx_pre),
        )

        readout_block_duration = (
            pp.calc_duration(
                gx,
                adc,
            )
        )

        used_tr = (
            excitation_block_duration
            + prephase_block_duration
            + tau1
            + refocusing_block_duration
            + tau2
            + readout_block_duration
            + post_block_duration
        )

        delay_TR_raw = TR - used_tr

        if delay_TR_raw < 0:
            raise ValueError(
                "TR is too short for "
                "slice-selective Localizer"
            )

        delay_TR = ceil_to_raster(
            delay_TR_raw
        )

    else:
        tau1 = (
            math.ceil(
                (
                    TE / 2
                    - 0.5 * (pp.calc_duration(rf1) + pp.calc_duration(rf2))
                    - pp.calc_duration(gx_pre)
                )
                / seq.grad_raster_time
            )
        ) * seq.grad_raster_time

        tau2 = (
            math.ceil(
                (TE / 2 - 0.5 * (pp.calc_duration(rf2)) - pp.calc_duration(gx_pre))
                / seq.grad_raster_time
            )
        ) * seq.grad_raster_time

        delay_TR = (
            math.ceil(
                (
                    TR
                    - TE
                    - pp.calc_duration(gx_pre)
                    - np.max(pp.calc_duration(gx_spoil, gx_pre))
                )
                / seq.grad_raster_time
            )
        ) * seq.grad_raster_time
        assert np.all(tau1 >= 0)
        assert np.all(tau2 >= 0)
        assert np.all(delay_TR >= pp.calc_duration(gx_spoil))

    # ======
    # CONSTRUCT SEQUENCE
    # ======
    # Loop over phase encodes and define sequence blocks
    for avg in range(num_averages):
        for i in range(Ny):
            # rf1.phase_offset = rf_phase / 180 * np.pi  # TODO: Include later
            # adc.phase_offset = rf_phase / 180 * np.pi
            # rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
            # rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
            if slice_selective:
                seq.add_block(
                    rf1,
                    g_slice,
                )
            else:
                seq.add_block(
                    rf1
                )
            gy_pre = pp.make_trapezoid(
                channel=ch1,
                area=phase_areas[i],
                duration=pp.calc_duration(gx_pre),
                system=system,
            )
            if slice_selective:
                seq.add_block(
                    gx_pre,
                    gy_pre,
                    g_slice_rephase,
                )
            else:
                seq.add_block(
                    gx_pre,
                    gy_pre,
                )
            seq.add_block(pp.make_delay(tau1))
            seq.add_block(rf2)
            seq.add_block(pp.make_delay(tau2))
            seq.add_block(gx, adc)
            gy_pre.amplitude = -gy_pre.amplitude
            seq.add_block(gx_spoil, gy_pre)  # TODO: Figure if we need spoiling
            seq.add_block(pp.make_delay(delay_TR))

    # Check whether the timing of the sequence is correct
    if check_timing:
        ok, error_report = seq.check_timing()
        if ok:
            log.info("Timing check passed successfully")
        else:
            log.info("Timing check failed. Error listing follows:")
            [print(e) for e in error_report]

    # Visualize Trajectory and other things
    if visualize:
        [k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc] = seq.calculate_kspace(spoil_val=2 * Nx * delta_k)
        log.info("Completed calculating Trajectory")
        log.info("Generating plots...")
        view_traj.view_traj_2d(k_traj_adc, k_traj, output_folder)

    # Save sequence
    log.debug(output_file)
    try:
        seq.write(output_file)
        log.debug("Seq file stored")
    except:
        log.error("Could not write sequence file")
        return False

    return True


# implement 2D radial Trajectory
def pypulseq_se2D_radial(inputs=None, check_timing=True, output_file="") -> bool:
    if not output_file:
        log.error("No output file specified")
        return False

    # ======
    # DEFAULTS FROM CONFIG FILE              TODO: MOVE DEFAULTS TO UI
    # ======
    LARMOR_FREQ = cfg.LARMOR_FREQ
    RF_MAX = cfg.RF_MAX
    RF_PI2_FRACTION = cfg.RF_PI2_FRACTION

    fov = 140e-3  # Define FOV and resolution
    Nx = 70
    Ny = Nx
    Nspokes = math.ceil(Nx * math.pi / 2)
    alpha1 = 90  # flip angle
    alpha1_duration = 100e-6  # pulse duration
    alpha2 = 180  # refocusing flip angle
    alpha2_duration = 100e-6  # pulse duration
    num_averages = 1
    BW = 20e3
    adc_dwell = 1 / BW
    adc_duration = Nx * adc_dwell  # 6.4e-3
    prephaser_duration = 3e-3  # TODO: Need to define this behind the scenes and optimze

    TR = inputs["TR"] / 1000
    TE = inputs["TE"] / 1000
    spoke_inc = "golden_angle"  # TODO: get from UI: GA or linear increment over 180

    # ======
    # INITIATE SEQUENCE
    # ======

    seq = pp.Sequence()

    # ======
    # SET SYSTEM CONFIG TODO --> ?
    # ======

    # system = pp.Opts(
    #     max_grad=12,
    #     grad_unit="mT/m",
    #     max_slew=25,
    #     slew_unit="T/m/s",
    #     rf_ringdown_time=20e-6,
    #     rf_dead_time=100e-6,
    #     rf_raster_time=1e-6,
    #     adc_dead_time=20e-6,
    # )

    system = pp.Opts(
        max_grad=400,
        grad_unit="mT/m",
        max_slew=4000,
        slew_unit="T/m/s",
        rf_ringdown_time=100e-6,
        rf_dead_time=100e-6,
        rf_raster_time=1e-6,
        adc_dead_time=10e-6,
    )

    # ======
    # CREATE EVENTS
    # ======
    # Create non-selective RF pulses for excitation and refocusing
    rf1 = pp.make_block_pulse(
        flip_angle=alpha1 * math.pi / 180,
        duration=alpha1_duration,
        delay=100e-6,
        system=system,
    )
    rf2 = pp.make_block_pulse(
        flip_angle=alpha2 * math.pi / 180,
        duration=alpha2_duration,
        delay=100e-6,
        phase_offset=math.pi / 2,
        system=system,
    )

    # Define other gradients and ADC events
    delta_k = 1 / fov  # frequency-oversampling is not implemented
    gx = pp.make_trapezoid(
        channel=ch0, flat_area=Nx * delta_k, flat_time=adc_duration, system=system
    )
    gy = pp.make_trapezoid(
        channel=ch1, flat_area=Nx * delta_k, flat_time=adc_duration, system=system
    )
    adc = pp.make_adc(
        num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system
    )
    gx_pre = pp.make_trapezoid(
        channel=ch0, area=gx.area / 2, duration=prephaser_duration, system=system
    )
    gy_pre = pp.make_trapezoid(
        channel=ch1, area=gy.area / 2, duration=prephaser_duration, system=system
    )

    amp_pre_max = gx_pre.amplitude
    amp_enc_max = gx.amplitude

    # Gradient spoiling -TODO: Need to see if this is really required based on data
    gx_spoil = pp.make_trapezoid(channel=ch0, area=2 * Nx * delta_k, system=system)
    gy_spoil = pp.make_trapezoid(channel=ch1, area=2 * Nx * delta_k, system=system)

    # ======
    # CALCULATE DELAYS
    # ======
    tau1 = (
        math.ceil(
            (
                TE / 2
                - 0.5 * (pp.calc_duration(rf1) + pp.calc_duration(rf2))
                - pp.calc_duration(gx_pre)
            )
            / seq.grad_raster_time
        )
    ) * seq.grad_raster_time

    tau2 = (
        math.ceil(
            (TE / 2 - 0.5 * (pp.calc_duration(rf2)) - pp.calc_duration(gx_pre))
            / seq.grad_raster_time
        )
    ) * seq.grad_raster_time

    delay_TR = (
        math.ceil(
            (
                TR
                - TE
                - pp.calc_duration(gx_pre)
                - np.max(pp.calc_duration(gx_spoil, gx_pre))
            )
            / seq.grad_raster_time
        )
    ) * seq.grad_raster_time
    assert np.all(tau1 >= 0)
    assert np.all(tau2 >= 0)
    assert np.all(delay_TR >= pp.calc_duration(gx_spoil))

    # ======
    # CONSTRUCT SEQUENCE
    # ======
    # Loop over phase encodes and define sequence blocks
    for avg in range(num_averages):
        for i in range(Nspokes):
            # rf1.phase_offset = rf_phase / 180 * np.pi  # TODO: Include later
            # adc.phase_offset = rf_phase / 180 * np.pi
            # rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
            # rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
            seq.add_block(rf1)
            if spoke_inc == "linear_increment":
                phi = i * (math.pi / Nspokes)
            elif spoke_inc == "golden_angle":
                phi = i * (111.246117975 / 180 * math.pi)
            gx_pre.amplitude = amp_pre_max * math.sin(phi)
            gy_pre.amplitude = amp_pre_max * math.cos(phi)
            seq.add_block(gx_pre, gy_pre)
            seq.add_block(pp.make_delay(tau1))
            seq.add_block(rf2)
            seq.add_block(pp.make_delay(tau2))
            gx.amplitude = amp_enc_max * math.sin(phi)
            gy.amplitude = amp_enc_max * math.cos(phi)
            seq.add_block(gx, gy, adc)
            seq.add_block(gx_spoil, gy_spoil)  # TODO: Figure if we need spoiling
            seq.add_block(pp.make_delay(delay_TR))
        seq.plot(time_range=[0, 3 * TR])

    # Check whether the timing of the sequence is correct
    if check_timing:
        ok, error_report = seq.check_timing()
        if ok:
            log.info("Timing check passed successfully")
        else:
            log.info("Timing check failed. Error listing follows:")
            [print(e) for e in error_report]

    log.debug(output_file)
    try:
        seq.write(output_file)
        log.debug("Seq file stored")
    except:
        log.error("Could not write sequence file")
        return False

    return True
