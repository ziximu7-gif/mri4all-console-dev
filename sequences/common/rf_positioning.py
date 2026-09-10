import numpy as np
import pypulseq as pp


def apply_centered_rf_frequency_modulation(
    rf,
    frequency_hz,
):
    """
    Encode an RF frequency offset directly
    into the complex RF waveform.

    The added phase is zero at the effective
    RF center, so slice/slab positioning does
    not introduce an unnecessary constant
    excitation phase.
    """

    frequency_hz = float(
        frequency_hz
    )

    if not np.isfinite(
        frequency_hz
    ):
        raise ValueError(
            "RF frequency offset "
            "must be finite"
        )

    rf_center_s = float(
        pp.calc_rf_center(
            rf
        )[0]
    )

    rf_time_s = np.asarray(
        rf.t,
        dtype=float,
    )

    rf_signal = np.asarray(
        rf.signal,
        dtype=np.complex128,
    )

    if (
        rf_time_s.shape
        != rf_signal.shape
    ):
        raise ValueError(
            "RF time and signal "
            "shapes must match"
        )

    phase = (
        2j
        * np.pi
        * frequency_hz
        * (
            rf_time_s
            - rf_center_s
        )
    )

    rf.signal = (
        rf_signal
        * np.exp(
            phase
        )
    )

    # Keep Pulseq event-level frequency
    # offset at zero because the current
    # FLOCRA interpreter treats it as a
    # global LO frequency.
    rf.freq_offset = 0.0

    return rf