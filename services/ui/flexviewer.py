from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *  # type: ignore
import qtawesome as qta

import common.runtime as rt
import common.logger as logger
from services.ui.viewerwidget import ViewerWidget
import services.ui.ui_runtime as ui_runtime


class FlexViewer(QDialog):
    def __init__(self):
        super(FlexViewer, self).__init__()
        uic.loadUi(f"{rt.get_console_path()}/services/ui/forms/flexviewer.ui", self)

        # self.setWindowFlags(Qt.WindowStaysOnTopHint)
        # self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        # self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)

        screen_width, screen_height = ui_runtime.get_screen_size()
        self.resize(int(screen_width * 0.9), int(screen_height * 0.85))
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

        viewerLayout = QHBoxLayout(
            self.flexViewerFrame
        )

        viewerLayout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        self.splitter = QSplitter(
            Qt.Horizontal
        )

        self.left_viewer = ViewerWidget()
        self.left_viewer.setProperty(
            "id",
            "flex_left",
        )

        self.right_viewer = ViewerWidget()
        self.right_viewer.setProperty(
            "id",
            "flex_right",
        )

        self.splitter.addWidget(
            self.left_viewer
        )

        self.splitter.addWidget(
            self.right_viewer
        )

        self.splitter.setStretchFactor(
            0,
            1,
        )

        self.splitter.setStretchFactor(
            1,
            1,
        )

        viewerLayout.addWidget(
            self.splitter
        )

        self.flexViewerFrame.setLayout(
            viewerLayout
        )

        # Backward compatibility: self.viewer points to
        # the left viewer.
        self.viewer = self.left_viewer

        self.right_viewer.hide()

        self.left_viewer.plot_xlim_changed.connect(
            self.right_viewer.set_plot_xlim
        )

        self.right_viewer.plot_xlim_changed.connect(
            self.left_viewer.set_plot_xlim
        )

        self.resizeButton.setText(" Maximize")
        self.resizeButton.clicked.connect(self.clickResize)
        self.resizeButton.setIcon(qta.icon("fa5s.expand"))
        self.resizeButton.setIconSize(QSize(24, 24))

        self.closeButton.setText(" Close")
        self.closeButton.clicked.connect(self.close)
        self.closeButton.setIcon(qta.icon("fa5s.check"))
        self.closeButton.setIconSize(QSize(24, 24))

        self.maximized = False

    def closeEvent(self, event):
        ui_runtime.examination_widget.flexViewerButton.setChecked(False)
        event.accept()

    def clickResize(self):
        if self.maximized:
            screen_width, screen_height = ui_runtime.get_screen_size()
            self.resize(int(screen_width * 0.9), int(screen_height * 0.85))
            qr = self.frameGeometry()
            cp = QDesktopWidget().availableGeometry().center()
            qr.moveCenter(cp)
            self.move(qr.topLeft())
            self.resizeButton.setText(" Maximize")
            self.resizeButton.setIcon(qta.icon("fa5s.expand"))
            self.maximized = False
        else:
            screen_width, screen_height = ui_runtime.get_screen_size()
            self.resize(int(screen_width * 1), int(screen_height * 1))
            qr = self.frameGeometry()
            cp = QDesktopWidget().availableGeometry().center()
            qr.moveCenter(cp)
            self.move(qr.topLeft())
            self.resizeButton.setText(" Restore")
            self.resizeButton.setIcon(qta.icon("fa5s.compress"))
            self.maximized = True
        self.closeButton.setFocus()

    def show_single_view(self):
        self.right_viewer.hide()
        self.splitter.setSizes(
            [1, 0]
        )

    def show_sequence_pair(
        self,
        rf_adc_path,
        gradients_path,
        scan_task,
    ):
        self.right_viewer.show()

        self.left_viewer.view_data(
            rf_adc_path,
            "plot",
            scan_task,
        )

        self.right_viewer.view_data(
            gradients_path,
            "plot",
            scan_task,
        )

        self.splitter.setSizes(
            [1, 1]
        )

        left_fig = (
            self.left_viewer
            .get_plot_figure()
        )

        if (
            left_fig is not None
            and left_fig.get_axes()
        ):
            xmin, xmax = (
                left_fig
                .get_axes()[0]
                .get_xlim()
            )

            self.right_viewer.set_plot_xlim(
                xmin,
                xmax,
            )
