"""Localizer slice-selective excitation contract tests.

These tests generate real Pulseq ``.seq`` files through
``sequences.common.make_se_2D.pypulseq_se2D`` and re-read them to
verify, on the written artefact:

* the legacy (no ``Slice_Thickness``) call keeps a NON-selective
  first block (RF only, no gradients);
* the slice-selective call puts the slice-select gradient on the
  orientation's third channel (``ch2``) for BOTH the excitation and
  the slice rephaser;
* timing still passes ``seq.check_timing()``;
* the raw sample-count contract used by
  ``run_reconstruction_localizer2d`` is preserved
  (NSA * Base_Resolution ADC events, each with Base_Resolution samples).

Orientation channel order is ``(read, phase, slice)``:

    Axial    -> ("x", "y", "z") -> slice Gz
    Coronal  -> ("x", "z", "y") -> slice Gy
    Sagittal -> ("y", "z", "x") -> slice Gx
"""

import sys

sys.path.insert(0, ".")
sys.path.append("../")

import numpy as np
import pypulseq as pp  # type: ignore

from common.geometry import orientation_channels
from sequences.common import make_se_2D


# Small matrix keeps the generated sequences fast while
# still exercising every code path.
BASE_RESOLUTION = 8
NSA = 1


def _base_inputs(orientation, fov_cm=20, te_ms=20, tr_ms=3000):
    """Minimal Localizer-shaped inputs for make_se_2D.pypulseq_se2D."""
    return {
        "TE": te_ms,
        "TR": tr_ms,
        "NSA": NSA,
        "FOV": fov_cm,
        "Orientation": orientation,
        "Base_Resolution": BASE_RESOLUTION,
        "BW": 32000,
        "Trajectory": "Cartesian",
        "PE_Ordering": "Center_out",
        "PF": 1,
        "view_traj": False,
    }


def _generate(tmp_path, inputs, name):
    """Write one .seq file and return it re-read as a Sequence object."""
    output_file = tmp_path / f"{name}.seq"

    ok = make_se_2D.pypulseq_se2D(
        inputs=inputs,
        check_timing=True,
        output_file=str(output_file),
        output_folder=str(tmp_path),
    )

    assert ok is True
    assert output_file.is_file()

    seq = pp.Sequence()
    seq.read(str(output_file))

    return seq


def _assert_timing_ok(seq):
    ok, errors = seq.check_timing()
    assert ok, errors


def _count_blocks_and_samples(seq):
    """Walk the re-read sequence and collect ADC blocks."""
    num_blocks = len(seq.dict_block_events)

    adc_blocks = []

    for block_index in range(1, num_blocks + 1):
        block = seq.get_block(block_index)

        if hasattr(block, "adc"):
            adc_blocks.append(block.adc)

    return num_blocks, adc_blocks


def _assert_raw_sample_contract(seq):
    """NSA * Base_Resolution ADC events, each Base_Resolution samples."""
    _, adc_blocks = _count_blocks_and_samples(seq)

    assert len(adc_blocks) == NSA * BASE_RESOLUTION

    for adc in adc_blocks:
        assert int(adc.num_samples) == BASE_RESOLUTION


# =====================================================================
# Test 1: legacy call stays non-selective
# =====================================================================


def test_legacy_localizer_excitation_is_nonselective(tmp_path):
    inputs = _base_inputs("Axial")
    assert "Slice_Thickness" not in inputs

    seq = _generate(tmp_path, inputs, "legacy_axial")

    first_block = seq.get_block(1)

    # RF present.
    assert hasattr(first_block, "rf")

    # No gradients of any axis: proves non-selective excitation.
    assert not hasattr(first_block, "gx")
    assert not hasattr(first_block, "gy")
    assert not hasattr(first_block, "gz")

    _assert_timing_ok(seq)
    _assert_raw_sample_contract(seq)


# =====================================================================
# Tests 2-4: slice selective excitation on ch2 per orientation
# =====================================================================


def _assert_slice_selective_channels(seq, expected_slice_channel):
    """First block = excitation RF + slice gradient on ch2.
    Second block = slice rephaser on the same ch2."""
    first_block = seq.get_block(1)

    assert hasattr(first_block, "rf")

    grad_axes = ["gx", "gy", "gz"]

    present = [axis for axis in grad_axes if hasattr(first_block, axis)]

    assert present == [f"g{expected_slice_channel}"]
    assert present == [grad_axes[["x", "y", "z"].index(expected_slice_channel)]]

    second_block = seq.get_block(2)

    present_second = [
        axis for axis in grad_axes if hasattr(second_block, axis)
    ]

    assert f"g{expected_slice_channel}" in present_second


def test_slice_selective_axial_uses_gz(tmp_path):
    expected = orientation_channels("Axial")[2]
    assert expected == "z"

    inputs = _base_inputs("Axial")
    inputs["Slice_Thickness"] = 10.0

    seq = _generate(tmp_path, inputs, "slice_axial")

    _assert_slice_selective_channels(seq, expected)
    _assert_timing_ok(seq)
    _assert_raw_sample_contract(seq)


def test_slice_selective_coronal_uses_gy(tmp_path):
    expected = orientation_channels("Coronal")[2]
    assert expected == "y"

    inputs = _base_inputs("Coronal")
    inputs["Slice_Thickness"] = 10.0

    seq = _generate(tmp_path, inputs, "slice_coronal")

    _assert_slice_selective_channels(seq, expected)
    _assert_timing_ok(seq)
    _assert_raw_sample_contract(seq)


def test_slice_selective_sagittal_uses_gx(tmp_path):
    expected = orientation_channels("Sagittal")[2]
    assert expected == "x"

    inputs = _base_inputs("Sagittal")
    inputs["Slice_Thickness"] = 10.0

    seq = _generate(tmp_path, inputs, "slice_sagittal")

    _assert_slice_selective_channels(seq, expected)
    _assert_timing_ok(seq)
    _assert_raw_sample_contract(seq)


# =====================================================================
# Guard rails: read/phase encoding must be unchanged
# =====================================================================


def test_readout_gradient_stays_on_read_channel(tmp_path):
    inputs = _base_inputs("Sagittal")
    inputs["Slice_Thickness"] = 10.0

    seq = _generate(tmp_path, inputs, "slice_sagittal_readout")

    read_channel = orientation_channels("Sagittal")[0]
    assert read_channel == "y"

    # The ADC block is the only block carrying an adc event.
    num_blocks = len(seq.dict_block_events)

    for block_index in range(1, num_blocks + 1):
        block = seq.get_block(block_index)

        if hasattr(block, "adc"):
            assert hasattr(block, f"g{read_channel}")
            # The slice channel must be free during readout.
            assert not hasattr(
                block,
                f"g{orientation_channels('Sagittal')[2]}",
            )
            break
    else:
        raise AssertionError("No ADC block found")


# =====================================================================
# Backward compatibility: old Localizer parameter dictionaries
# =====================================================================


def test_localizer_old_parameters_default_slice_thickness():
    """A parameter dict WITHOUT Slice_Thickness must still be accepted.

    This locks the ``parameters.get("Slice_Thickness", 10.0)`` fallback so
    that a future regression back to ``parameters["Slice_Thickness"]``
    fails loudly instead of breaking previously saved Localizer tasks.
    """
    from sequences.localizer import SequenceSE_2D

    sequence = SequenceSE_2D()

    parameters = sequence.get_default_parameters()
    parameters.pop("Slice_Thickness")

    assert sequence.set_parameters(
        parameters,
        None,
    )

    assert (
        sequence.param_Slice_Thickness
        == 10.0
    )


# =====================================================================
# Spin-echo timing physics (RF centers, not just raster legality)
#
# check_timing() only proves that the Pulseq events are legal on the
# hardware raster/dead-time grid. It does NOT prove that TE is the
# requested value. These tests lock the acquisition physics:
#
#     90 RF center --TE/2--> 180 RF center --TE/2--> ADC center
# =====================================================================


def _duration_from_zero(seq, block_index):
    """calc_duration summed over blocks 1..block_index.

    Returned normalised to a Python float so it is additive across
    blocks; re-read sequences report the last block's duration as a
    length-1 ndarray.
    """
    total = 0.0

    for index in range(1, block_index + 1):
        value = pp.calc_duration(seq.get_block(index))

        if isinstance(value, np.ndarray):
            value = float(np.max(value))

        total += float(value)

    return total


def _rf_center_absolute(seq, rf_block_index):
    """Absolute time of the RF effective center for a given block."""
    block = seq.get_block(rf_block_index)

    return (
        _duration_from_zero(seq, rf_block_index - 1)
        + block.rf.delay
        + pp.calc_rf_center(block.rf)[0]
    )


def _adc_center_absolute(seq, adc_block_index):
    """Absolute time of the ADC sampling-window center for a block."""
    block = seq.get_block(adc_block_index)

    return (
        _duration_from_zero(seq, adc_block_index - 1)
        + block.adc.delay
        + 0.5 * block.adc.num_samples * block.adc.dwell
    )


def _find_rf_blocks(seq):
    """Return (excitation_block, refocusing_block) indices.

    Identification is by ORDER, not by ``rf.use``: this vendored
    pypulseq drops the RF ``use`` flag when a sequence is re-read
    (``rf_from_lib_data`` appends padding and ends up indexing past
    the stored ``use`` value). Within every TR the excitation always
    precedes the refocusing pulse, so the first RF block of the first
    TR is the excitation and the second is the refocusing pulse.
    """
    rf_blocks = [
        block_index
        for block_index in range(1, len(seq.dict_block_events) + 1)
        if hasattr(seq.get_block(block_index), "rf")
    ]

    assert len(rf_blocks) >= 2, "Expected at least two RF blocks"

    return rf_blocks[0], rf_blocks[1]


def _find_adc_block(seq):
    for block_index in range(1, len(seq.dict_block_events) + 1):
        if hasattr(seq.get_block(block_index), "adc"):
            return block_index

    raise AssertionError("No ADC block found")


def _echo_timing_offsets(seq):
    """Measure the two TE/2 intervals of the first spin echo."""
    excitation_block, refocusing_block = _find_rf_blocks(seq)
    adc_block = _find_adc_block(seq)

    assert excitation_block is not None
    assert refocusing_block is not None
    assert excitation_block < refocusing_block < adc_block

    excitation_center = _rf_center_absolute(seq, excitation_block)
    refocusing_center = _rf_center_absolute(seq, refocusing_block)
    adc_center = _adc_center_absolute(seq, adc_block)

    return (
        refocusing_center - excitation_center,
        adc_center - refocusing_center,
        adc_center - excitation_center,
    )


def _assert_spin_echo_te(seq, te_ms):
    """The echo must be centred on the requested TE.

    The slice-selective timing rounds each half up to the gradient
    raster, so the analytic residual is bounded by one raster per
    interval; the small extra headroom absorbs .seq read-back
    quantisation.
    """
    first_half, second_half, total = _echo_timing_offsets(seq)

    te = te_ms / 1000.0
    tolerance = 2 * seq.grad_raster_time

    np.testing.assert_allclose(
        first_half,
        te / 2.0,
        atol=tolerance,
        err_msg="90 -> 180 interval is not TE/2",
    )
    np.testing.assert_allclose(
        second_half,
        te / 2.0,
        atol=tolerance,
        err_msg="180 -> ADC interval is not TE/2",
    )
    np.testing.assert_allclose(
        total,
        te,
        atol=2 * tolerance,
        err_msg="90 -> ADC interval is not TE",
    )


def test_slice_selective_axial_echo_timing_matches_te(tmp_path):
    inputs = _base_inputs("Axial", te_ms=20)
    inputs["Slice_Thickness"] = 10.0

    seq = _generate(tmp_path, inputs, "slice_axial_timing")

    _assert_spin_echo_te(seq, 20)


def test_slice_selective_coronal_echo_timing_matches_te(tmp_path):
    inputs = _base_inputs("Coronal", te_ms=20)
    inputs["Slice_Thickness"] = 10.0

    seq = _generate(tmp_path, inputs, "slice_coronal_timing")

    _assert_spin_echo_te(seq, 20)


def test_slice_selective_sagittal_echo_timing_matches_te(tmp_path):
    inputs = _base_inputs("Sagittal", te_ms=20)
    inputs["Slice_Thickness"] = 10.0

    seq = _generate(tmp_path, inputs, "slice_sagittal_timing")

    _assert_spin_echo_te(seq, 20)


def _assert_blocks_tile_tr(seq, tr_ms):
    """The blocks of ONE TR must tile TR exactly.

    The second excitation RF marks the start of the second TR, so the
    blocks of the first TR are everything before it. Two independent
    things are locked here:

    * the summed block durations equal TR; this catches any mismatch
      between the ``used_tr`` accounting and the blocks that are
      actually assembled;
    * the final delay-only block holds a strictly positive delay.

    Note: production code never clamps ``delay_TR``. A negative
    ``delay_TR_raw`` raises ``ValueError`` instead, so there is no
    silent floor behaviour to guard against here.
    """
    second_tr_rf_blocks = [
        block_index
        for block_index in range(1, len(seq.dict_block_events) + 1)
        if hasattr(seq.get_block(block_index), "rf")
    ]

    assert len(second_tr_rf_blocks) >= 3, "Expected at least two TRs of RF"

    # Third RF block start == first block of the second TR.
    first_tr_blocks = range(1, second_tr_rf_blocks[2])

    total = 0.0
    last_block = None

    for block_index in first_tr_blocks:
        block = seq.get_block(block_index)

        value = pp.calc_duration(block)
        if isinstance(value, np.ndarray):
            value = float(np.max(value))
        total += float(value)

        last_block = block

    # The final block of a TR is the delay_TR fill.
    assert hasattr(last_block, "delay")
    assert not any(
        hasattr(last_block, attr)
        for attr in ("rf", "gx", "gy", "gz", "adc")
    )
    assert float(last_block.delay.delay) > 0

    np.testing.assert_allclose(
        total,
        tr_ms / 1000.0,
        atol=2 * seq.grad_raster_time,
        err_msg="one TR of blocks does not tile TR",
    )


def test_slice_selective_te_40_ms_delay_tr_stays_positive(tmp_path):
    """A longer TE must still tile TR with a positive ``delay_TR``.

    TE=40 ms shifts the ``used_tr`` / ``delay_TR_raw`` balance away
    from the default TE=20 ms, so the used-TR accounting is checked
    beyond the default parameters. Both TEs take the SAME code path:
    a negative ``delay_TR_raw`` would raise the ``TR is too short``
    ValueError rather than being clamped.
    """
    inputs = _base_inputs("Axial", te_ms=40)
    inputs["Slice_Thickness"] = 10.0

    seq = _generate(tmp_path, inputs, "slice_axial_te40")

    _assert_spin_echo_te(seq, 40)
    _assert_blocks_tile_tr(seq, 3000)


def test_slice_selective_te_20_ms_delay_tr_stays_positive(tmp_path):
    """Default TE must tile TR with a positive ``delay_TR``.

    ``delay_TR_raw`` is expected to be positive and far from zero for
    the default parameters (used_tr is tens of ms against a 3 s TR).
    """
    inputs = _base_inputs("Axial", te_ms=20)
    inputs["Slice_Thickness"] = 10.0

    seq = _generate(tmp_path, inputs, "slice_axial_te20_tr")

    _assert_blocks_tile_tr(seq, 3000)
