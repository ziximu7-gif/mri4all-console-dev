import os
from pathlib import Path
import datetime
from PyQt5 import uic
import matplotlib.pyplot as plt
import pickle

import external.seq.adjustments_acq.config as cfg
from external.seq.adjustments_acq.scripts import run_pulseq
from sequences import PulseqSequence
from sequences.common import make_gre_3D
from common.constants import *
import common.logger as logger
from common.types import ResultItem
import common.helper as helper

log = logger.get_logger()

from common.ipc import Communicator

ipc_comm = Communicator(Communicator.ACQ)


class SequenceGRE_3D(PulseqSequence, registry_key=Path(__file__).stem):
    # Sequence parameters
    param_TE: int = 50
    param_TR: int = 250
    param_NSA: int = 1
    param_orientation: str = "Axial"
    param_FOV: int = 200
    param_baseresolution: int = 64
    param_slices: int = 8
    param_BW: int = 32000
    param_trajectory: str = "Cartesian"
    param_ordering: str = "linear_up"
    param_dummy_shots: int = 20

    @classmethod
    def get_readable_name(self) -> str:
        return "3D Gradient Echo"

    @classmethod
    def get_description(self) -> str:
        return "Volumetric 3D GRE acquisition with Cartesian sampling"

    def setup_ui(self, widget) -> bool:
        seq_path = os.path.dirname(os.path.abspath(__file__))
        uic.loadUi(f"{seq_path}/{self.get_name()}/interface.ui", widget)

        widget.TR_SpinBox.valueChanged.connect(self.update_info)
        widget.Baseresolution_SpinBox.valueChanged.connect(self.update_info)
        widget.Slices_SpinBox.valueChanged.connect(self.update_info)
        return True

    def update_info(self):
        duration_sec = int(
            self.main_widget.TR_SpinBox.value()
            * (
                self.main_widget.Baseresolution_SpinBox.value()
                * self.main_widget.Slices_SpinBox.value()
                + self.param_dummy_shots
            )
            / 1000
        )
        duration = str(datetime.timedelta(seconds=duration_sec))

        res_slice = (
            self.main_widget.FOV_SpinBox.value()
            / self.main_widget.Slices_SpinBox.value()
            * 10
        )
        res_inplane = (
            self.main_widget.FOV_SpinBox.value()
            / self.main_widget.Baseresolution_SpinBox.value()
            * 10
        )

        self.show_ui_info_text(
            f"TA: {duration} sec       Voxel Size: {res_inplane:.2f} x {res_inplane:.2f} x {res_slice:.2f} mm"
        )

    def get_parameters(self) -> dict:
        return {
            "TE": self.param_TE,
            "TR": self.param_TR,
            "NSA": self.param_NSA,
            "orientation": self.param_orientation,
            "FOV": self.param_FOV,
            "baseresolution": self.param_baseresolution,
            "slices": self.param_slices,
            "BW": self.param_BW,
            "trajectory": self.param_trajectory,
            "ordering": self.param_ordering,
            "FA": self.param_FA,
        }

    @classmethod
    def get_default_parameters(
        self,
    ) -> dict:
        return {
            "TE": 0,
            "TR": 1000,
            "NSA": 1,
            "orientation": "Axial",
            "FOV": 15,
            "baseresolution": 32,
            "slices": 8,
            "BW": 32000,
            "trajectory": "Cartesian",
            "ordering": "linear_up",
            "FA": 20,
        }

    def set_parameters(self, parameters, scan_task) -> bool:
        self.problem_list = []
        try:
            self.param_TE = parameters["TE"]
            self.param_TR = parameters["TR"]
            self.param_NSA = parameters["NSA"]
            self.param_orientation = parameters["orientation"]
            self.param_FOV = parameters["FOV"]
            self.param_baseresolution = parameters["baseresolution"]
            self.param_slices = parameters["slices"]
            self.param_BW = parameters["BW"]
            self.param_trajectory = parameters["trajectory"]
            self.param_ordering = parameters["ordering"]
            self.param_FA = parameters["FA"]
        except:
            self.problem_list.append("Invalid parameters provided")
            return False
        return self.validate_parameters(scan_task)

    def write_parameters_to_ui(self, widget) -> bool:
        widget.TE_SpinBox.setValue(self.param_TE)
        widget.TR_SpinBox.setValue(self.param_TR)
        widget.FA_SpinBox.setValue(self.param_FA)
        widget.NSA_SpinBox.setValue(self.param_NSA)
        widget.Orientation_ComboBox.setCurrentText(self.param_orientation)
        widget.FOV_SpinBox.setValue(self.param_FOV)
        widget.Baseresolution_SpinBox.setValue(self.param_baseresolution)
        widget.Slices_SpinBox.setValue(self.param_slices)
        widget.BW_SpinBox.setValue(self.param_BW)
        widget.Trajectory_ComboBox.setCurrentText(self.param_trajectory)
        widget.Ordering_ComboBox.setCurrentText(self.param_ordering)
        return True

    def read_parameters_from_ui(self, widget, scan_task) -> bool:
        self.problem_list = []
        self.param_TE = widget.TE_SpinBox.value()
        self.param_TR = widget.TR_SpinBox.value()
        self.param_NSA = widget.NSA_SpinBox.value()
        self.param_orientation = widget.Orientation_ComboBox.currentText()
        self.param_FOV = widget.FOV_SpinBox.value()
        self.param_baseresolution = widget.Baseresolution_SpinBox.value()
        self.param_slices = widget.Slices_SpinBox.value()
        self.param_BW = widget.BW_SpinBox.value()
        self.param_trajectory = widget.Trajectory_ComboBox.currentText()
        self.param_ordering = widget.Ordering_ComboBox.currentText()
        self.param_FA = widget.FA_SpinBox.value()
        self.validate_parameters(scan_task)
        return self.is_valid()

    def validate_parameters(self, scan_task) -> bool:
        if self.param_TE > self.param_TR:
            self.problem_list.append("TE cannot be longer than TR")
        return self.is_valid()

    def calculate_sequence(self, scan_task) -> bool:
        log.info("Calculating sequence " + self.get_name())
        ipc_comm.send_status(f"Calculating sequence...")

        scan_task.processing.recon_mode = "basic3d"
        scan_task.processing.dim = 3
        scan_task.processing.dim_size = f"{self.param_slices},{self.param_baseresolution},{2*self.param_baseresolution}"
        scan_task.processing.oversampling_read = 2
        self.seq_file_path = self.get_working_folder() + "/seq/acq0.seq"

        if not self.generate_pulseq():
            log.error("Unable to calculate sequence " + self.get_name())
            return False

        log.info("Done calculating sequence " + self.get_name())
        return True

    def run_sequence(self, scan_task) -> bool:
        log.info("Running sequence " + self.get_name())
        ipc_comm.send_status(f"Preparing scan...")

        expected_duration_sec = int(
            self.param_TR
            * (self.param_baseresolution * self.param_slices + self.param_dummy_shots)
            / 1000
        )

        plot_instructions = True

        rxd, rx_t = run_pulseq(
            seq_file=self.seq_file_path,
            rf_center=cfg.LARMOR_FREQ,
            tx_t=1,
            grad_t=10,
            tx_warmup=100,
            shim_x=cfg.SHIM_X,
            shim_y=cfg.SHIM_Y,
            shim_z=cfg.SHIM_Z,
            grad_cal=False,
            save_np=True,
            save_mat=False,
            save_msgs=False,
            gui_test=False,
            case_path=self.get_working_folder(),
            raw_filename="raw",
            expected_duration_sec=expected_duration_sec,
            plot_instructions=plot_instructions,
        )
        scan_task.adjustment.rf.larmor_frequency = cfg.LARMOR_FREQ

        if plot_instructions:
            file = open(self.get_working_folder() + "/other/seq.plot", "wb")
            fig = plt.gcf()
            pickle.dump(fig, file)
            file.close()

            result = ResultItem()
            result.name = "seq_plot"
            result.description = "Timing diagram of sequence"
            result.type = "plot"
            result.file_path = "other/seq.plot"
            result.autoload_viewer = 4
            scan_task.results.append(result)

        log.info("Done running sequence " + self.get_name())
        return True

    def generate_pulseq(self) -> bool:
        return make_gre_3D.pypulseq_gre3D(
            inputs={
                "TE": self.param_TE,
                "TR": self.param_TR,
                "NSA": self.param_NSA,
                "orientation": self.param_orientation,
                "FOV": self.param_FOV,
                "baseresolution": self.param_baseresolution,
                "slices": self.param_slices,
                "BW": self.param_BW,
                "ordering": self.param_ordering,
                "FA": self.param_FA,
                "dummy_shots": self.param_dummy_shots,
            },
            check_timing=True,
            output_file=self.seq_file_path,
            working_folder=self.get_working_folder(),
        )