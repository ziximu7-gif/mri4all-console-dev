import numpy as np

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
)

import pyqtgraph.opengl as gl


class Planning3DWidget(QWidget):
    """
    Read-only 3D scanner-space planning viewer.

    Coordinate convention:
        X = scanner X
        Y = scanner Y
        Z = scanner Z

    Scene coordinates are meters, matching common.geometry.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        layout.setSpacing(0)

        self.view = gl.GLViewWidget()

        self.view.setBackgroundColor(
            (0, 0, 0, 255)
        )

        layout.addWidget(
            self.view
        )

        # Camera looks toward scanner isocenter.
        #
        # Scene units are meters. Current Localizer FOV is
        # around 0.20 m, so 0.45 m gives a useful initial view.
        self.view.setCameraPosition(
            distance=0.45,
            elevation=20,
            azimuth=45,
        )

        self.axis_items = []

        self._create_scanner_axes()

    def _create_scanner_axes(
        self,
    ):
        """
        Draw positive scanner X/Y/Z axes from isocenter.

        This is deliberately the first minimal 3D scene.
        Planning geometry will be added in later steps.
        """

        axis_length_m = 0.12

        axis_specs = (
            (
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [axis_length_m, 0.0, 0.0],
                    ],
                    dtype=float,
                ),
                (1.0, 0.2, 0.2, 1.0),
            ),
            (
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, axis_length_m, 0.0],
                    ],
                    dtype=float,
                ),
                (0.2, 1.0, 0.2, 1.0),
            ),
            (
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, axis_length_m],
                    ],
                    dtype=float,
                ),
                (0.2, 0.5, 1.0, 1.0),
            ),
        )

        for points, color in axis_specs:

            item = gl.GLLinePlotItem(
                pos=points,
                color=color,
                width=3,
                antialias=True,
                mode="lines",
            )

            self.view.addItem(
                item
            )

            self.axis_items.append(
                item
            )
