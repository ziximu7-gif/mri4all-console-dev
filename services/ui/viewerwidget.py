import glob
try:
    import sip  # type: ignore
except ImportError:
    from PyQt5 import sip  # type: ignore
import pickle
from pathlib import Path
from typing import Literal, Optional
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *  # type: ignore

import pyqtgraph as pg  # type: ignore
import pydicom
import numpy as np
from PyQt5 import QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import matplotlib.pyplot as plt
import common.logger as logger
from common.types import ResultTypes, ScanTask, TimeSeriesResult
from services.ui.spatialbox import PlanningState

log = logger.get_logger()


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        plt.style.use("dark_background")
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class StaticTextItem(pg.TextItem):
    """
    Stays where you put it and ignores viewport translation/zoom.
    """

    def updateTransform(self, force=False):
        if not self.isVisible():
            return

        p = self.parentItem()
        if p is None:
            pt = QtGui.QTransform()
        else:
            pt = p.sceneTransform()

        if not force and pt == self._lastTransform:
            return
        self.setTransform(pt.inverted()[0])
        self._lastTransform = pt
        self.updateTextPos()


class ViewerWidget(QWidget):

    planning_changed = pyqtSignal()
    # layout: QBoxLayout
    widget: Optional[QWidget] = None
    viewed_scan_task: Optional[ScanTask] = None
    viewer_mode: ResultTypes = "empty"

    def __init__(self):
        super(ViewerWidget, self).__init__()

        self.setLayout(
            QVBoxLayout(self)
        )

        self.layout().setContentsMargins(
            0,
            0,
            0,
            0,
        )

        self.layout().setSpacing(0)

        # -------------------------------------------------
        # Scan planning
        # -------------------------------------------------

        self.planning_state = None
        self.planning_orientation = None

        self.fov_roi = None
        self.shim_roi = None
        self.fov_label = None
        self.shim_label = None

        # Prevent recursive updates when one viewer
        # updates another viewer.
        self.updating_planning_rois = False

        self.set_empty_viewer()

        # def __del__(self):
        #     self.clear_view()

    def clear_view(self):
        if self.widget:
            widget_to_delete = self.widget
            self.layout().removeWidget(self.widget)
            sip.delete(widget_to_delete)
            self.widget = None
            self.viewed_scan_task = None
        self.fov_roi = None
        self.shim_roi = None
        self.fov_label = None
        self.shim_label = None

        self.viewer_mode = "empty"

    def set_empty_viewer(self):
        self.widget = QWidget()
        self.widget.setStyleSheet("background-color: #000;")
        self.layout().addWidget(self.widget)

    def view_data(self, file_path: str, viewer_mode: ResultTypes, task) -> bool:
        """
        Used to load results into the viewer for the inline widgets
        """
        self.clear_view()
        self.viewer_mode = viewer_mode
        if viewer_mode == "dicom":
            self.load_dicoms(file_path, task)
            return True
        elif viewer_mode == "plot":
            self.load_pickled_plot(file_path, task)
            return True
        else:
            self.set_empty_viewer()
            self.viewer_mode = "empty"
            return False

    def set_planning_context(
        self,
        orientation,
        planning_state,
    ):
        """
        Tell this viewer which localizer plane it represents
        and which shared PlanningState it should use.
        """

        self.planning_orientation = orientation
        self.planning_state = planning_state

    def _plane_axes(self):
        """
        Return horizontal and vertical normalized axes.

        Prototype convention:

            Axial:
                horizontal = X
                vertical   = Y

            Coronal:
                horizontal = X
                vertical   = Z

            Sagittal:
                horizontal = Y
                vertical   = Z
        """

        if self.planning_orientation == "Axial":
            return "x", "y"

        if self.planning_orientation == "Coronal":
            return "x", "z"

        if self.planning_orientation == "Sagittal":
            return "y", "z"

        return None, None

    def _plane_rotation_axis(self):
        """
        Return the Box3D rotation component corresponding
        to the current localizer plane.

        Axial XY:
            in-plane rotation is around Z

        Coronal XZ:
            in-plane rotation is around Y

        Sagittal YZ:
            in-plane rotation is around X
        """

        if self.planning_orientation == "Axial":
            return "z"

        if self.planning_orientation == "Coronal":
            return "y"

        if self.planning_orientation == "Sagittal":
            return "x"

        return None

    def _box_angle(self, box):

        rotation_axis = (
            self._plane_rotation_axis()
        )

        if rotation_axis is None:
            return 0.0

        return float(
            getattr(
                box,
                f"rotation_{rotation_axis}",
            )
        )

    def _set_box_angle(
        self,
        box,
        angle,
    ):

        rotation_axis = (
            self._plane_rotation_axis()
        )

        if rotation_axis is None:
            return

        setattr(
            box,
            f"rotation_{rotation_axis}",
            float(angle),
        )

        box.clamp()

    def _planning_image_size(self):

        if not isinstance(
            self.widget,
            pg.ImageView,
        ):
            return None

        image = self.widget.image

        if image is None:
            return None

        if len(image.shape) < 2:
            return None

        height = image.shape[-2]
        width = image.shape[-1]

        return float(width), float(height)

    def _box_to_rect(
        self,
        box,
    ):
        """
        Project Box3D onto this viewer plane.

        Returns:
            x, y, width, height
        in image pixel coordinates.
        """

        image_size = (
            self._planning_image_size()
        )

        if image_size is None:
            return None

        image_width, image_height = (
            image_size
        )

        horizontal_axis, vertical_axis = (
            self._plane_axes()
        )

        if horizontal_axis is None:
            return None

        horizontal_center = getattr(
            box,
            f"center_{horizontal_axis}",
        )

        vertical_center = getattr(
            box,
            f"center_{vertical_axis}",
        )

        horizontal_size = getattr(
            box,
            f"size_{horizontal_axis}",
        )

        vertical_size = getattr(
            box,
            f"size_{vertical_axis}",
        )

        roi_width = (
            horizontal_size
            * image_width
        )

        roi_height = (
            vertical_size
            * image_height
        )

        roi_x = (
            horizontal_center
            * image_width
            - roi_width / 2.0
        )

        roi_y = (
            vertical_center
            * image_height
            - roi_height / 2.0
        )

        return (
            roi_x,
            roi_y,
            roi_width,
            roi_height,
        )
    def _roi_to_box(
        self,
        roi,
        box,
    ):
        """
        Update shared Box3D from the current 2D ROI.

        Supports:
            translation
            resizing
            rotation
        """

        image_size = (
            self._planning_image_size()
        )

        if image_size is None:
            return

        image_width, image_height = (
            image_size
        )

        if (
            image_width <= 0
            or image_height <= 0
        ):
            return

        horizontal_axis, vertical_axis = (
            self._plane_axes()
        )

        if horizontal_axis is None:
            return

        size = roi.size()

        roi_width = float(
            size.x()
        )

        roi_height = float(
            size.y()
        )

        # -------------------------------------------------
        # IMPORTANT:
        #
        # roi.pos() + size / 2 is NOT correct after
        # rotation.
        #
        # The ROI center must be mapped from ROI-local
        # coordinates into the parent image coordinates.
        # -------------------------------------------------

        local_center = QPointF(
            roi_width / 2.0,
            roi_height / 2.0,
        )

        parent_center = (
            roi.mapToParent(
                local_center
            )
        )

        horizontal_center = (
            float(parent_center.x())
            / image_width
        )

        vertical_center = (
            float(parent_center.y())
            / image_height
        )

        horizontal_size = (
            roi_width
            / image_width
        )

        vertical_size = (
            roi_height
            / image_height
        )

        setattr(
            box,
            f"center_{horizontal_axis}",
            horizontal_center,
        )

        setattr(
            box,
            f"center_{vertical_axis}",
            vertical_center,
        )

        setattr(
            box,
            f"size_{horizontal_axis}",
            horizontal_size,
        )

        setattr(
            box,
            f"size_{vertical_axis}",
            vertical_size,
        )

        # Save current in-plane rotation
        self._set_box_angle(
            box,
            roi.angle(),
        )

        box.clamp()

    def _apply_box_to_roi(
        self,
        roi,
        box,
    ):
        """
        Project Box3D into this viewer and apply:

            center
            size
            rotation

        to an existing PyQtGraph ROI.
        """

        image_size = (
            self._planning_image_size()
        )

        if image_size is None:
            return

        image_width, image_height = (
            image_size
        )

        horizontal_axis, vertical_axis = (
            self._plane_axes()
        )

        if horizontal_axis is None:
            return

        center_horizontal = getattr(
            box,
            f"center_{horizontal_axis}",
        )

        center_vertical = getattr(
            box,
            f"center_{vertical_axis}",
        )

        size_horizontal = getattr(
            box,
            f"size_{horizontal_axis}",
        )

        size_vertical = getattr(
            box,
            f"size_{vertical_axis}",
        )

        roi_width = (
            size_horizontal
            * image_width
        )

        roi_height = (
            size_vertical
            * image_height
        )

        desired_center_x = (
            center_horizontal
            * image_width
        )

        desired_center_y = (
            center_vertical
            * image_height
        )

        angle = self._box_angle(
            box
        )

        # Apply size first
        roi.setSize(
            [
                roi_width,
                roi_height,
            ],
            update=False,
        )

        # Reset origin temporarily
        roi.setPos(
            [
                0.0,
                0.0,
            ],
            update=False,
        )

        # Apply rotation
        roi.setAngle(
            angle,
            update=False,
        )

        # Find where the local center currently appears
        local_center = QPointF(
            roi_width / 2.0,
            roi_height / 2.0,
        )

        current_center = (
            roi.mapToParent(
                local_center
            )
        )

        # Translate so the ROI center is exactly
        # the desired Box3D center.
        roi.setPos(
            [
                desired_center_x
                - float(
                    current_center.x()
                ),

                desired_center_y
                - float(
                    current_center.y()
                ),
            ],
            update=False,
        )

        roi.stateChanged(
            finish=False
        )



    def create_planning_rois(self):
        """
        Create the FOV and Shim rectangles
        on the current DICOM viewer.
        """

        if self.planning_state is None:
            return

        if self.planning_orientation is None:
            return

        if not isinstance(
            self.widget,
            pg.ImageView,
        ):
            return

        # Avoid creating twice.
        if self.fov_roi is not None:
            return

        # -------------------------------------------------
        # FOV box
        # -------------------------------------------------

        fov_rect = self._box_to_rect(
            self.planning_state.fov_box
        )

        if fov_rect is None:
            return

        (
            fov_x,
            fov_y,
            fov_width,
            fov_height,
        ) = fov_rect
        self.fov_roi = pg.RectROI(
            [
                fov_x,
                fov_y,
            ],
            [
                fov_width,
                fov_height,
            ],
            angle=self._box_angle(
                self.planning_state.fov_box
            ),
            pen=pg.mkPen(
                "y",
                width=3,
            ),
            movable=True,
            rotatable=True,
            resizable=True,
        )

        self.widget.getView().addItem(
            self.fov_roi
        )
        self.fov_roi.addRotateHandle(
            [1.0, 0.0],
            [0.5, 0.5],
        )
        self.fov_roi.addScaleHandle(
            [1.0, 1.0],
            [0.0, 0.0],
        )

        self.fov_roi.addScaleHandle(
            [0.0, 0.0],
            [1.0, 1.0],
        )
        self.fov_label = pg.TextItem(
            text="FOV",
            color="y",
            anchor=(0, 1),
        )

        self.widget.getView().addItem(
            self.fov_label
        )

        # -------------------------------------------------
        # Shim box
        # -------------------------------------------------

        shim_rect = self._box_to_rect(
            self.planning_state.shim_box
        )

        (
            shim_x,
            shim_y,
            shim_width,
            shim_height,
        ) = shim_rect

        self.shim_roi = pg.RectROI(
            [
                shim_x,
                shim_y,
            ],
            [
                shim_width,
                shim_height,
            ],
            angle=self._box_angle(
                self.planning_state.shim_box
            ),
            pen=pg.mkPen(
                "c",
                width=2,
            ),
            movable=True,
            rotatable=True,
            resizable=True,
        )

        self.widget.getView().addItem(
            self.shim_roi
        )
        self.shim_roi.addRotateHandle(
            [1.0, 0.0],
            [0.5, 0.5],
        )
        self.shim_roi.addScaleHandle(
            [1.0, 1.0],
            [0.0, 0.0],
        )

        self.shim_roi.addScaleHandle(
            [0.0, 0.0],
            [1.0, 1.0],
        )
        self.shim_label = pg.TextItem(
            text="Shim",
            color="c",
            anchor=(0, 1),
        )

        self.widget.getView().addItem(
            self.shim_label
        )

        # -------------------------------------------------
        # Connect interaction signals
        # -------------------------------------------------

        self.fov_roi.sigRegionChanged.connect(
            self._fov_roi_changed
        )

        self.shim_roi.sigRegionChanged.connect(
            self._shim_roi_changed
        )
        self._apply_box_to_roi(
            self.fov_roi,
            self.planning_state.fov_box,
        )

        self._apply_box_to_roi(
            self.shim_roi,
            self.planning_state.shim_box,
        )

        self._update_planning_labels()

    def _update_planning_labels(self):
        """
        Keep FOV / Shim labels attached to each ROI
        and display the current in-plane angle.
        """

        if (
            self.fov_roi is not None
            and
            self.fov_label is not None
        ):

            fov_pos = (
                self.fov_roi.pos()
            )

            fov_angle = (
                self.fov_roi.angle()
            )

            self.fov_label.setText(
                f"FOV  {fov_angle:.1f} deg"
            )

            self.fov_label.setPos(
                float(fov_pos.x()),
                float(fov_pos.y()),
            )

        if (
            self.shim_roi is not None
            and
            self.shim_label is not None
        ):

            shim_pos = (
                self.shim_roi.pos()
            )

            shim_angle = (
                self.shim_roi.angle()
            )

            self.shim_label.setText(
                f"Shim  {shim_angle:.1f} deg"
            )

            self.shim_label.setPos(
                float(shim_pos.x()),
                float(shim_pos.y()),
            )

    def _fov_roi_changed(self):

        if self.updating_planning_rois:
            return

        if self.planning_state is None:
            return

        if self.fov_roi is None:
            return

        self._roi_to_box(
            self.fov_roi,
            self.planning_state.fov_box,
        )

        self._update_planning_labels()

        self.planning_changed.emit()


    def _shim_roi_changed(self):

        if self.updating_planning_rois:
            return

        if self.planning_state is None:
            return

        if self.shim_roi is None:
            return

        self._roi_to_box(
            self.shim_roi,
            self.planning_state.shim_box,
        )

        self._update_planning_labels()

        self.planning_changed.emit()

    def refresh_planning_rois(self):
        """
        Refresh this viewer from the shared PlanningState.

        Called when another localizer viewer changes
        FOV or Shim geometry.
        """

        if self.planning_state is None:
            return

        if self.planning_orientation is None:
            return

        if self.fov_roi is None:
            return

        if self.shim_roi is None:
            return

        self.updating_planning_rois = True

        try:

            self._apply_box_to_roi(
                self.fov_roi,
                self.planning_state.fov_box,
            )

            self._apply_box_to_roi(
                self.shim_roi,
                self.planning_state.shim_box,
            )

        finally:

            self.updating_planning_rois = False

        # Put it HERE.
        self._update_planning_labels()

    def clear_planning_context(self):
        self.planning_orientation = None
        self.planning_state = None

        self.fov_roi = None
        self.shim_roi = None

    def load_dicoms(self, input_path, task: Optional[ScanTask] = None):
        if not input_path:
            self.set_empty_viewer()
            return

        lstFilesDCM = None
        if isinstance(input_path, list):
            lstFilesDCM = input_path
        else:
            path = Path(input_path)
            if not path.is_file():
                lstFilesDCM = [str(name) for name in glob.glob(input_path + "*.dcm")]
            else:
                lstFilesDCM = [input_path]
        lstFilesDCM.sort()
        if len(lstFilesDCM) < 1:
            self.set_empty_viewer()
            return

        ds = pydicom.dcmread(lstFilesDCM[0])
        ConstPixelDims = (len(lstFilesDCM), int(ds.Rows), int(ds.Columns))
        ArrayDicom = np.zeros(ConstPixelDims, dtype=ds.pixel_array.dtype)
        for filenameDCM in lstFilesDCM:
            ds = pydicom.dcmread(filenameDCM)
            ArrayDicom[lstFilesDCM.index(filenameDCM), :, :] = ds.pixel_array

        pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

        self.widget = pg.ImageView()
        self.widget.setImage(ArrayDicom)
        self.widget.timeLine.setPen(color=(200, 200, 200), width=8)
        self.widget.timeLine.setHoverPen(color=(255, 255, 255), width=8)

        # viewer_widget.ui.histogram.hide()
        self.widget.ui.roiBtn.hide()
        self.widget.ui.menuBtn.hide()
        self.widget.autoRange()

        if task:
            text = StaticTextItem(
                html=f"""<span style='font-size: 16px; color: #999;'>
                    {task.patient.last_name}, {task.patient.first_name}<br/>
                    {task.patient.mrn}<br/>
                    {task.protocol_name}<br/>
                    Scan {task.scan_number}<br/>
                    </span><br/>""",
                anchor=(0, 0),
            )
            text.setPos(0, 0)  # todo: this only works with 0,0 position
            self.widget.addItem(text)

        self.layout().addWidget(self.widget)
        if (
            self.planning_state is not None
            and
            self.planning_orientation is not None
        ):
            self.create_planning_rois()

    def load_pickled_plot(self, input_path, task: Optional[ScanTask] = None):
        if not input_path:
            self.set_empty_viewer()
            return

        pickled_file_path = Path(input_path)

        if not pickled_file_path.is_file():
            log.error("File not found: " + str(pickled_file_path))
            return

        # TODO: Add error handling!
        self.widget = QWidget()
        self.widget.setLayout(QVBoxLayout(self.widget))
        self.widget.layout().setContentsMargins(0, 0, 0, 0)
        self.widget.layout().setSpacing(0)

        plt.style.use("dark_background")
        with open(pickled_file_path, "rb") as pickle_file:
            fig = pickle.load(pickle_file)

        figCanvas = FigureCanvasQTAgg(fig)
        fig.tight_layout()

        toolbar = NavigationToolbar2QT(figCanvas, self)
        toolbar.setStyleSheet(
            "QToolBar::separator { background-color: #0C1123; } QFrame, QFrame:hover { border: 0px solid #000; }  QToolBar { background-color: #000; } QToolButton { background-color: #262C44; } QToolButton:checked { background-color: #FFF; }  QToolButton:disabled { background-color: #000; } QToolButton:hover { background-color: #E0A526; }"
        )
        unwanted_buttons = ["Back", "Forward"]
        for x in toolbar.actions():
            if x.text() in unwanted_buttons:
                toolbar.removeAction(x)

        self.widget.layout().addWidget(figCanvas)
        self.widget.layout().addWidget(toolbar)
        self.layout().addWidget(self.widget)

        if task:
            figCanvas.setToolTip(
                f"{task.scan_number}:  {task.protocol_name}"
            )

        # -------------------------------------------------
        # Interactive span selector for plot results
        # -------------------------------------------------

        # List storing the axis most recently clicked
        curr_ax = []

        axis = fig.get_axes()

        self.textvar = None

        # Detect the currently selected matplotlib axis
        def on_click(event):
            if event.inaxes:
                curr_ax[:] = [
                    event.inaxes
                ]

        # Apply the selected x-range to all plot axes
        def onselect(xmin, xmax):

            # Ignore an accidental click without dragging
            if xmin == xmax:

                for ax, span in zip(
                    axis,
                    list_of_spans,
                ):
                    span.set_visible(
                        False
                    )

                if self.textvar:
                    self.textvar.remove()
                    self.textvar = None

                fig.canvas.draw_idle()

                return

            # Display the same selected range
            # on all sequence plot axes
            for ax, span in zip(
                axis,
                list_of_spans,
            ):
                span.set_visible(
                    True
                )

                span.extents = (
                    xmin,
                    xmax,
                )

            txt = (
                f"start = {xmin:.2f}, "
                f"end = {xmax:.2f}, "
                f"delta = {xmax - xmin:.2f}"
            )

            if self.textvar:
                self.textvar.remove()

            self.textvar = plt.figtext(
                0.5,
                0.01,
                txt,
                wrap=True,
                horizontalalignment="center",
                fontsize=10,
            )

            fig.canvas.draw_idle()

        # Create one horizontal SpanSelector
        # for every matplotlib axis in the figure
        list_of_spans = [
            SpanSelector(
                ax,
                onselect,
                "horizontal",
                useblit=True,
                props=dict(
                    alpha=0.5,
                    facecolor="#262C44",
                ),
                interactive=True,
                drag_from_anywhere=True,
            )
            for ax in axis
        ]

        figCanvas.mpl_connect(
            "button_press_event",
            on_click,
        )

    def layoutUpdate(self):
        if self.viewer_mode == "plot":
            self.widget.layout().itemAt(0).widget().figure.tight_layout()
            self.widget.layout().itemAt(0).widget().figure.canvas.draw()


# def load_plot(self, result: Optional[TimeSeriesResult] = None):
#     sc = MplCanvas(self)
#     if result is None:
#         result = TimeSeriesResult(data=np.random.normal(size=10).tolist())

#     result.show(sc.axes)
#     self.layout().addWidget(sc)
#     self.widget = sc
