import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pypulseq as pp


def load_sequence(sequence_source):
    """
    Load a Pulseq sequence.

    sequence_source can be either:

    1. Path to a .seq file
    2. An already-created pypulseq Sequence object
    """

    # Already a Pulseq Sequence object
    if hasattr(sequence_source, "get_block"):
        return sequence_source

    # Otherwise treat it as a .seq file path
    seq_file = Path(sequence_source)

    if not seq_file.is_file():
        raise FileNotFoundError(
            f"Pulseq sequence file not found: {seq_file}"
        )

    seq = pp.Sequence()
    seq.read(str(seq_file))

    return seq


def save_figure(fig, output_file):
    """
    Save one Matplotlib figure using MRI4ALL's
    existing pickled .plot format.
    """

    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_file,
        "wb",
    ) as file:
        pickle.dump(
            fig,
            file,
        )

    return str(output_file)


def visualize_sequence(
    sequence_source,
    output_folder,
    prefix="sequence",
    time_range=(0, 0.1),
    time_disp="ms",
    plot_type="Gradient",
    label="",
):
    """
    Visualize one Pulseq sequence using the native
    PyPulseq Sequence.plot() implementation.

    Parameters
    ----------
    sequence_source:
        .seq file path or pypulseq Sequence object.

    output_folder:
        Folder where MRI4ALL .plot files will be stored.

    prefix:
        Prefix used for generated files.

        Example:
            prefix="localizer_axial"

        generates:
            localizer_axial_rf_adc.plot
            localizer_axial_gradients.plot

    time_range:
        Pulseq display range in seconds.

        Example:
            (0, 0.1)

        means:
            0 - 100 ms

    time_disp:
        Pulseq time display unit.

        Usually:
            "ms"

    plot_type:
        Native Pulseq plot type.

        Usually:
            "Gradient"

    label:
        Optional Pulseq label to display.

    Returns
    -------
    dict

        {
            "rf_adc": "...plot",
            "gradients": "...plot"
        }
    """

    seq = load_sequence(
        sequence_source
    )

    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Use the native PyPulseq viewer.
    #
    # IMPORTANT:
    # This requires Sequence.plot() to support:
    #
    #     show=False
    #
    # and return:
    #
    #     fig1, fig2
    # -----------------------------------------------------

    fig1, fig2 = seq.plot(
        label=label,
        save=False,
        time_range=time_range,
        time_disp=time_disp,
        plot_type=plot_type,
        show=False,
    )

    # -----------------------------------------------------
    # PyPulseq fig1:
    #
    # ADC
    # RF magnitude
    # RF / ADC phase
    # -----------------------------------------------------

    rf_adc_file = (
        output_folder
        / f"{prefix}_rf_adc.plot"
    )

    save_figure(
        fig1,
        rf_adc_file,
    )

    # -----------------------------------------------------
    # PyPulseq fig2:
    #
    # Gx
    # Gy
    # Gz
    # -----------------------------------------------------

    gradients_file = (
        output_folder
        / f"{prefix}_gradients.plot"
    )

    save_figure(
        fig2,
        gradients_file,
    )
    plt.close(fig1)
    plt.close(fig2)

    return {
        "rf_adc": str(rf_adc_file),
        "gradients": str(gradients_file),
    }


def visualize_sequences(
    sequence_sources,
    output_folder,
    prefixes=None,
    time_range=(0, 0.1),
    time_disp="ms",
    plot_type="Gradient",
):
    """
    Visualize multiple Pulseq sequences.

    Useful for sequences such as the 3-plane Localizer,
    which produces multiple .seq files.

    Parameters
    ----------
    sequence_sources:
        List of .seq paths or Sequence objects.

    prefixes:
        Optional file prefixes.

        Example:
            [
                "axial",
                "coronal",
                "sagittal",
            ]

    Returns
    -------
    list[dict]
    """

    sequence_sources = list(
        sequence_sources
    )

    if prefixes is None:
        prefixes = [
            f"sequence_{index}"
            for index
            in range(
                len(sequence_sources)
            )
        ]

    if len(prefixes) != len(
        sequence_sources
    ):
        raise ValueError(
            "prefixes must have the same length "
            "as sequence_sources"
        )

    results = []

    for source, prefix in zip(
        sequence_sources,
        prefixes,
    ):

        result = visualize_sequence(
            sequence_source=source,
            output_folder=output_folder,
            prefix=prefix,
            time_range=time_range,
            time_disp=time_disp,
            plot_type=plot_type,
        )

        results.append(
            result
        )

    return results