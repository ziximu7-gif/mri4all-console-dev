import numpy as np
import os
from os import path
import time
import matplotlib.pyplot as plt

import common.logger as logger
from common.constants import *
from common.types import ScanTask
import services.recon.utils as utils

from recon.kspaceFiltering.kspace_filtering import *
from recon.B0Correction import B0Corrector
import recon.DICOM.DICOM_utils as DICOM
from recon.ismrmrd.numpy_to_ismrmrd import create_ismrmrd
from recon.image_filters import denoise
from recon.recon_utils.cartesian3d import (
    reconstruct_cartesian_3d_complex,
)

log = logger.get_logger()


def run_reconstruction(folder: str, task: ScanTask) -> bool:
    """
    Perform the reconstruction of the scan contained in the given folder. Scan information such
    as sequence, protocol name, patient information and system information can be found in the
    task object.
    """
    # log.info(f"Folder where the task is = {folder}")
    # log.info(f"JSON information = {task}")

    if task.processing.recon_mode == "bypass":
        log.info("Bypassing reconstruction")
        return True

    if task.processing.recon_mode == "fake_dicoms":
        log.info("Generating fake DICOMs")
        utils.generate_fake_dicoms(folder, task)
        time.sleep(1)
        return True

    if task.processing.recon_mode == "basic3d":
        log.info("Running Basic 3D reconstruction")
        run_reconstruction_basic3d(folder, task)
        return True

    if task.processing.recon_mode == "localizer2d":
        log.info("Running 3-plane Localizer reconstruction")
        run_reconstruction_localizer2d(folder, task)
        return True

    if task.processing.trajectory == "cartesian":
        log.info("Running Cartesian reconstruction")
        run_reconstruction_cartesian(folder, task)
        return True

    log.error(f"Unknown trajectory type: {task.processing.trajectory}")
    return False


def run_reconstruction_basic3d(folder: str, task: ScanTask) -> bool:
    if task.processing.dim != 3:
        log.error(
            "Unable to perform reconstruction. This algorithm only support 3 dimensions"
        )
        return False

    order = np.load(
        folder + "/" + mri4all_taskdata.RAWDATA + "/" + mri4all_scanfiles.PE_ORDER
    )
    adc_phases = np.load(
        folder + "/" + mri4all_taskdata.RAWDATA + "/" + mri4all_scanfiles.ADC_PHASE
    )
    kData = np.load(
        folder + "/" + mri4all_taskdata.RAWDATA + "/" + mri4all_scanfiles.RAWDATA
    )
    dims = task.processing.dim_size.split(",")
    # dim = slices:pe:read

    images, kspaces = reconstruct_cartesian_3d_complex(
        raw=kData,
        order=order,
        adc_phases=adc_phases,
        dims=dims,
        echo_count=1,
        oversampling_read=task.processing.oversampling_read,
    )

    fft = images[0]
    kSpace = kspaces[0]

    log.info(f"Readout size = {kData.shape}")
    log.info(f"Matrix size = {kSpace.shape}")
    log.info(f"FFT shape = {fft.shape}")
    log.info(f"FFT abs min = {np.min(np.abs(fft))}")
    log.info(f"FFT abs max = {np.max(np.abs(fft))}")
    log.info(f"FFT abs mean = {np.mean(np.abs(fft))}")
    log.info(f"FFT contains NaN = {np.isnan(fft).any()}")

    for i in range(fft.shape[2]):
        log.info(
            f"Slice {i}: "
            f"min={np.min(np.abs(fft[:, :, i]))}, "
            f"max={np.max(np.abs(fft[:, :, i]))}"
        )
    DICOM.write_dicom(fft, task, folder + "/" + mri4all_taskdata.DICOM, result_index=0)

    # kSpace = np.angle(kSpace)
    kSpace = 100 * (kSpace - kSpace.min()) / (kSpace.max() - kSpace.min())
    DICOM.write_dicom(
        kSpace,
        task,
        folder + "/" + mri4all_taskdata.DICOM,
        series_offset=1,
        name="k-Space",
        primary_result=False,
        autoload_viewer=2,
        result_index=2,
    )

    DICOM.write_dicom(
        np.angle(fft),
        task,
        folder + "/" + mri4all_taskdata.DICOM,
        series_offset=2,
        name="Phase",
        primary_result=False,
        result_index=3,
        autoload_viewer=3,
    )

    return True

def run_reconstruction_localizer2d(
    folder: str,
    task: ScanTask,
) -> bool:

    log.info(
        "Starting 3-plane 2D localizer reconstruction"
    )

    # -------------------------------------------------
    # Read matrix size
    # -------------------------------------------------

    if task.processing.dim_size:
        dims = task.processing.dim_size.split(",")

        ny = int(dims[0])
        nx = int(dims[1])
        nsa = int(task.parameters.get("NSA", 1))

    else:
        raise RuntimeError(
            "Localizer reconstruction: "
            "dim_size is not defined"
        )

    log.info(
        f"Localizer matrix size = {ny} x {nx}"
    )
    log.info(
        f"Localizer NSA = {nsa}"
    )

    # -------------------------------------------------
    # Raw-data folder
    # -------------------------------------------------

    raw_folder = os.path.join(
        folder,
        mri4all_taskdata.RAWDATA,
    )

    # -------------------------------------------------
    # Definition of the three localizer groups
    # -------------------------------------------------

    groups = [
        {
            "orientation": "Axial",
            "filename": "localizer_axial.npy",
            "viewer": 1,
            "series_offset": 0,
        },
        {
            "orientation": "Coronal",
            "filename": "localizer_coronal.npy",
            "viewer": 2,
            "series_offset": 1,
        },
        {
            "orientation": "Sagittal",
            "filename": "localizer_sagittal.npy",
            "viewer": 3,
            "series_offset": 2,
        },
    ]

    # -------------------------------------------------
    # Reconstruct every localizer group
    # -------------------------------------------------

    for index, group in enumerate(groups):

        orientation = group["orientation"]

        raw_file = os.path.join(
            raw_folder,
            group["filename"],
        )

        log.info(
            f"Reconstructing {orientation} localizer"
        )

        log.info(
            f"Loading raw data: {raw_file}"
        )

        # ---------------------------------------------
        # Check file
        # ---------------------------------------------

        if not os.path.isfile(raw_file):
            raise RuntimeError(
                f"{orientation} localizer raw data "
                f"not found: {raw_file}"
            )

        # ---------------------------------------------
        # Load ADC / k-space
        # ---------------------------------------------

        raw = np.load(raw_file)

        log.info(
            f"{orientation} raw samples = "
            f"{raw.size}"
        )

        expected_samples = nsa * nx * ny

        if raw.size != expected_samples:
            raise RuntimeError(
                f"{orientation} localizer expected "
                f"{expected_samples} samples, "
                f"but found {raw.size}"
            )

        # ---------------------------------------------
        # Convert 1D ADC stream into 2D k-space
        # ---------------------------------------------

        kspace_averages = raw.reshape(
            (nsa, ny, nx)
        )

        kspace = np.mean(
            kspace_averages,
            axis=0,
        )

        log.info(
            f"{orientation} k-space shape = "
            f"{kspace.shape}"
        )

        # ---------------------------------------------
        # 2D FFT
        # ---------------------------------------------

        image = np.fft.fftshift(
            np.fft.fft2(
                np.fft.fftshift(
                    kspace
                )
            )
        )

        log.info(
            f"{orientation} image shape = "
            f"{image.shape}"
        )

        log.info(
            f"{orientation} image magnitude: "
            f"min={np.min(np.abs(image)):.6f}, "
            f"max={np.max(np.abs(image)):.6f}, "
            f"mean={np.mean(np.abs(image)):.6f}"
        )

        # ---------------------------------------------
        # DICOM writer expects a 3D ndarray:
        #
        # height x width x slices
        #
        # Our localizer is only one 2D image,
        # therefore add a one-slice dimension.
        # ---------------------------------------------

        image_3d = image[:, :, np.newaxis]

        # ---------------------------------------------
        # Write one DICOM series
        # ---------------------------------------------

        DICOM.write_dicom(
            image_3d,
            task,
            folder
            + "/"
            + mri4all_taskdata.DICOM,
            series_offset=group["series_offset"],
            name=f"{orientation} Localizer",
            description=(
                f"{orientation} 2D Spin-Echo "
                "localizer image"
            ),
            primary_result=(
                orientation == "Axial"
            ),
            autoload_viewer=group["viewer"],
            result_index=index,
        )

        log.info(
            f"Finished {orientation} localizer"
        )

    log.info(
        "Finished 3-plane Localizer reconstruction"
    )

    return True

def run_reconstruction_cartesian(folder: str, task: ScanTask):
    """
    Runs the reconstruction pipeline for Cartesian sampling
    """
    fnames = os.listdir(folder)
    if not fnames:
        log.error(f"Folder {folder} is empty.")
        return

    # Load the k-space data
    kData = np.load(
        folder + "/" + mri4all_taskdata.RAWDATA + "/" + mri4all_scanfiles.RAWDATA
    )
    kTraj = np.genfromtxt(
        folder + "/" + mri4all_taskdata.RAWDATA + "/" + mri4all_scanfiles.TRAJ,
        delimiter=",",
    )  # pe_table a lot by 2 # check rotation

    if kTraj.shape[0] > 2:
        kTraj = np.rot90(kTraj)
    # grad_delay_correction(kData, kTraj, delayT, param)

    filterType = "fermi"
    kData = kFilter(kData, filterType, center_correction=True)
    log.info(f"kSpace {filterType} filtering finished.")

    # Preform B0 correction and reconstruct the image
    fname_B0_map = list(filter(lambda x: mri4all_scanfiles.BDATA in x, fnames))
    Y = kData
    kt = kTraj
    df = np.load(path.join(folder, fname_B0_map[0])) if fname_B0_map else None
    Lx = 1
    nonCart = None
    params = None
    b0_corrector = B0Corrector(Y, kt, df, Lx, nonCart, params)
    iData = b0_corrector()
    log.info(f"B0 correction finished.")

    # Denoise the image
    try:
        strength = task.processing.denoising_strength
        iData = denoise.remove_gaussian_noise_complex(
            iData, method="gaussian_filter", strength=strength
        )
        log.info(f"Finished image denoising with strength={strength}.")
    except ValueError:
        log.error(f"Image denoising failed.")

    # Create the DICOM file
    DICOM.write_dicom(iData, task, folder + "/" + mri4all_taskdata.DICOM)
    log.info(f"DICOM writting finished.")

    # Create the ISMRMRD file
    # TODO: Enable ISMRMRD creation after bug fix
    create_ismrmrd(folder, kData, task)
    log.info(f"ISMRMRD format writting finished.")
