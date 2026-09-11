from copy import copy

import numpy as np

from pypulseq.add_gradients import (
    add_gradients,
)


_GRADIENT_AXES = (
    "x",
    "y",
    "z",
)

_GRADIENT_AXIS_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
}


def scale_gradient_event(
    gradient,
    scale,
):
    """
    Scale a gradient event without relying on
    newer PyPulseq scale_grad().
    """

    scaled = copy(gradient)

    if gradient.type == "trap":
        scaled.amplitude = (
            gradient.amplitude
            * scale
        )

        scaled.area = (
            gradient.area
            * scale
        )

        scaled.flat_area = (
            gradient.flat_area
            * scale
        )

        scaled.first = (
            gradient.first
            * scale
        )

        scaled.last = (
            gradient.last
            * scale
        )

    elif gradient.type == "grad":
        scaled.waveform = (
            np.asarray(
                gradient.waveform,
                dtype=float,
            )
            * scale
        )

        scaled.first = (
            gradient.first
            * scale
        )

        scaled.last = (
            gradient.last
            * scale
        )

    else:
        raise ValueError(
            "Unsupported gradient type: "
            + str(gradient.type)
        )

    return scaled


def transform_gradient_events(
    gradients,
    logical_to_scanner,
    system,
):
    """
    Transform logical X/Y/Z gradient events
    into physical scanner X/Y/Z gradients.

    G_scanner = M @ G_logical
    """

    contributions = {
        "x": [],
        "y": [],
        "z": [],
    }

    for gradient in gradients:
        if gradient is None:
            continue

        if gradient.type not in (
            "trap",
            "grad",
        ):
            raise ValueError(
                "Only gradient events can "
                "be transformed."
            )

        logical_axis_index = (
            _GRADIENT_AXIS_INDEX[
                gradient.channel
            ]
        )

        for (
            scanner_axis_index,
            scanner_axis,
        ) in enumerate(
            _GRADIENT_AXES
        ):
            coefficient = float(
                logical_to_scanner[
                    scanner_axis_index,
                    logical_axis_index,
                ]
            )

            if abs(coefficient) < 1e-12:
                continue

            transformed = (
                scale_gradient_event(
                    gradient=gradient,
                    scale=coefficient,
                )
            )

            transformed.channel = (
                scanner_axis
            )

            contributions[
                scanner_axis
            ].append(
                transformed
            )

    scanner_gradients = []

    for scanner_axis in (
        _GRADIENT_AXES
    ):
        axis_gradients = (
            contributions[
                scanner_axis
            ]
        )

        if not axis_gradients:
            continue

        if len(axis_gradients) == 1:
            scanner_gradients.append(
                axis_gradients[0]
            )

        else:
            scanner_gradients.append(
                add_gradients(
                    grads=axis_gradients,
                    system=system,
                )
            )

    return scanner_gradients


def add_gradient_block(
    seq,
    gradients,
    logical_to_scanner,
    system,
    extra_events=None,
):
    """
    Add a gradient block.

    If logical_to_scanner is provided,
    logical gradients are first transformed
    into scanner physical X/Y/Z.
    """

    if logical_to_scanner is None:
        output_gradients = [
            gradient
            for gradient in gradients
            if gradient is not None
        ]

    else:
        output_gradients = (
            transform_gradient_events(
                gradients=gradients,
                logical_to_scanner=(
                    logical_to_scanner
                ),
                system=system,
            )
        )

    events = list(
        output_gradients
    )

    if extra_events is not None:
        events.extend(
            extra_events
        )

    seq.add_block(
        *events
    )

def logical_gradient_safety_scale(
    logical_to_scanner,
    safety_margin=0.95,
):
    """
    Return a conservative logical gradient limit scale
    for an oblique logical-to-scanner transform.

    If several logical gradients are active at the same
    time:

        G_scanner = M @ G_logical

    then the worst physical-axis amplification is bounded
    by the largest absolute row sum of M.
    """

    if logical_to_scanner is None:
        return 1.0, 1.0

    matrix = np.asarray(
        logical_to_scanner,
        dtype=float,
    )

    if matrix.shape != (3, 3):
        raise ValueError(
            "logical_to_scanner must be 3x3."
        )

    if not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError(
            "logical_to_scanner must be finite."
        )

    mixing_bound = float(
        np.max(
            np.sum(
                np.abs(matrix),
                axis=1,
            )
        )
    )

    if mixing_bound <= 1.0 + 1e-9:
        return mixing_bound, 1.0

    limit_scale = (
        float(safety_margin)
        / mixing_bound
    )

    return (
        mixing_bound,
        limit_scale,
    )