import numpy as np

from recon.recon_utils.cartesian_translation import (
    apply_cartesian_center_translation,
)


def _image_to_kspace(image):
    return np.fft.fftshift(
        np.fft.ifftn(
            np.fft.fftshift(
                image
            )
        )
    )


def _kspace_to_image(kspace):
    return np.fft.fftshift(
        np.fft.fftn(
            np.fft.fftshift(
                kspace
            )
        )
    )


def test_cartesian_center_translation():
    shape = (
        16,
        8,
        8,
    )

    center_indices = (
        8,
        4,
        4,
    )

    fov_logical_m = np.array(
        [
            0.08,
            0.08,
            0.08,
        ]
    )

    image = np.zeros(
        shape,
        dtype=np.complex128,
    )

    # Current object location relative
    # to the reconstruction center:
    #
    # read  +1 voxel = +10 mm
    # phase -1 voxel = -10 mm
    # third +2 voxels = +20 mm
    image[
        9,
        3,
        6,
    ] = 1.0

    kspace = _image_to_kspace(
        image
    )

    translated = (
        apply_cartesian_center_translation(
            kspace=kspace,
            center_logical_m=[
                0.01,
                -0.01,
                0.02,
            ],
            fov_logical_m=(
                fov_logical_m
            ),
            oversampling_read=2,
            center_indices=(
                center_indices
            ),
        )
    )

    reconstructed = (
        _kspace_to_image(
            translated
        )
    )

    maximum_index = (
        np.unravel_index(
            np.argmax(
                np.abs(
                    reconstructed
                )
            ),
            shape,
        )
    )

    assert maximum_index == (
        8,
        4,
        4,
    )