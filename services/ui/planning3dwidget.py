import numpy as np

from PyQt5.QtCore import (
    Qt,
    pyqtSignal,
)

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QToolButton,
    QShortcut,
    QComboBox,
)

from PyQt5.QtGui import (
    QVector3D,
    QVector4D,
    QKeySequence,
)

from pyqtgraph import Transform3D
import pyqtgraph.opengl as gl

from OpenGL.GL import (
    GL_DEPTH_TEST,
    GL_BLEND,
    GL_ALPHA_TEST,
    GL_CULL_FACE,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
)

from common.geometry import (
    planning_box_to_scan_geometry,
    scan_geometry_box_corners_scanner_m,
    scan_geometry_box_edges_scanner_m,
    planning_euler_to_matrix,
    planning_matrix_to_euler,
)


PLANNING_OVERLAY_GL_OPTIONS = {
    GL_DEPTH_TEST: False,
    GL_BLEND: True,
    GL_ALPHA_TEST: False,
    GL_CULL_FACE: False,
    "glBlendFunc": (
        GL_SRC_ALPHA,
        GL_ONE_MINUS_SRC_ALPHA,
    ),
}


class PlanningGLViewWidget(
    gl.GLViewWidget
):
    """
    GLViewWidget with one additional planning gesture.

    Standard pyqtgraph camera interaction is preserved:

        left drag          -> orbit camera
        Ctrl + left drag   -> pan camera
        middle drag        -> pan camera
        wheel              -> zoom

    Planning interaction:

        right drag         -> FOV operation

    This widget deliberately knows nothing about
    PlanningState or the MOVE/ROT/axis selection. It only
    reports how far the pointer travelled horizontally.
    """

    planning_dragged = pyqtSignal(
        float
    )

    planning_handle_dragged = pyqtSignal(
        str,
        float,
        float,
        bool,
    )

    planning_handle_hovered = pyqtSignal(
        object
    )

    def __init__(
        self,
        parent=None,
    ):
        super().__init__(
            parent=parent
        )

        self._planning_drag_active = False
        self._planning_drag_last_pos = None

        self._planning_handle_items = {}
        self._active_planning_handle = None
        self._planning_handle_last_pos = None
        self._planning_handle_symmetric_resize = False
        self._hovered_planning_handle = None

        self.setMouseTracking(
            True
        )

    @staticmethod
    def _event_local_pos(
        event,
    ):
        if hasattr(
            event,
            "position",
        ):
            return event.position()

        return event.localPos()

    def register_planning_handle(
        self,
        item,
        handle_id,
    ):
        """
        Register one separately pickable GL item.

        The GL view only knows the opaque handle ID.
        It does not know PlanningState semantics.
        """

        self._planning_handle_items[
            item
        ] = str(
            handle_id
        )

    def _pick_planning_handle(
        self,
        position,
    ):
        """
        Pick the closest registered planning GL item around
        the pointer using GLViewWidget's native OpenGL picking.
        """

        radius = 8

        region = (
            int(
                position.x()
            ) - radius,
            int(
                position.y()
            ) - radius,
            2 * radius,
            2 * radius,
        )

        for item in self.itemsAt(
            region
        ):
            handle_id = (
                self._planning_handle_items
                .get(
                    item
                )
            )

            if handle_id is not None:
                return handle_id

        return None

    def mousePressEvent(
        self,
        event,
    ):
        position = (
            self._event_local_pos(
                event
            )
        )

        # Also reset pyqtgraph's camera-drag reference point,
        # preventing a jump when starting a new camera drag.
        self.mousePos = position

        if (
            event.button()
            == Qt.LeftButton
        ):
            handle_id = (
                self._pick_planning_handle(
                    position
                )
            )

            if handle_id is not None:

                self._active_planning_handle = (
                    handle_id
                )

                self._planning_handle_last_pos = (
                    position
                )

                # Interaction mode is fixed for the entire drag.
                # Alt must already be held when the handle is picked.
                self._planning_handle_symmetric_resize = bool(
                    event.modifiers()
                    & Qt.AltModifier
                )

                event.accept()
                return

        if (
            event.button()
            == Qt.RightButton
        ):
            self._planning_drag_active = True
            self._planning_drag_last_pos = position

            self.setFocus()

            event.accept()
            return

        super().mousePressEvent(
            event
        )

    def mouseMoveEvent(
        self,
        event,
    ):
        if (
            event.buttons()
            == Qt.NoButton
        ):
            position = (
                self._event_local_pos(
                    event
                )
            )

            hovered = (
                self._pick_planning_handle(
                    position
                )
            )

            if (
                hovered
                != self._hovered_planning_handle
            ):
                self._hovered_planning_handle = (
                    hovered
                )

                self.planning_handle_hovered.emit(
                    hovered
                )

        if (
            self._active_planning_handle
            is not None
            and (
                event.buttons()
                & Qt.LeftButton
            )
        ):
            position = (
                self._event_local_pos(
                    event
                )
            )

            if (
                self._planning_handle_last_pos
                is None
            ):
                self._planning_handle_last_pos = (
                    position
                )

                event.accept()
                return

            difference = (
                position
                - self._planning_handle_last_pos
            )

            self._planning_handle_last_pos = (
                position
            )

            self.planning_handle_dragged.emit(
                self._active_planning_handle,
                float(
                    difference.x()
                ),
                float(
                    difference.y()
                ),
                bool(
                    self._planning_handle_symmetric_resize
                ),
            )

            event.accept()
            return

        if (
            self._planning_drag_active
            and (
                event.buttons()
                & Qt.RightButton
            )
        ):
            position = (
                self._event_local_pos(
                    event
                )
            )

            if (
                self._planning_drag_last_pos
                is None
            ):
                self._planning_drag_last_pos = (
                    position
                )

                event.accept()
                return

            difference = (
                position
                - self._planning_drag_last_pos
            )

            self._planning_drag_last_pos = (
                position
            )

            # v0 contract:
            #
            # Horizontal screen drag is a scalar control
            # for the selected scanner axis.
            #
            # right -> positive
            # left  -> negative
            delta_pixels = float(
                difference.x()
            )

            if (
                abs(delta_pixels)
                > 1e-6
            ):
                self.planning_dragged.emit(
                    delta_pixels
                )

            event.accept()
            return

        super().mouseMoveEvent(
            event
        )

    def mouseReleaseEvent(
        self,
        event,
    ):
        if (
            event.button()
            == Qt.LeftButton
            and self._active_planning_handle
            is not None
        ):
            self._active_planning_handle = None
            self._planning_handle_last_pos = None
            self._planning_handle_symmetric_resize = False

            event.accept()
            return

        if (
            event.button()
            == Qt.RightButton
            and self._planning_drag_active
        ):
            # mousePos is re-seeded in mousePressEvent, so no
            # extra bookkeeping is needed here.
            self._planning_drag_active = False
            self._planning_drag_last_pos = None

            event.accept()
            return

        super().mouseReleaseEvent(
            event
        )

    def clear_planning_handle_state(
        self,
    ):
        self._active_planning_handle = None
        self._planning_handle_last_pos = None
        self._planning_handle_symmetric_resize = False
        self._hovered_planning_handle = None

        self.planning_handle_hovered.emit(
            None
        )

    def leaveEvent(
        self,
        event,
    ):
        if (
            self._hovered_planning_handle
            is not None
        ):
            self._hovered_planning_handle = None

            self.planning_handle_hovered.emit(
                None
            )

        super().leaveEvent(
            event
        )


class Planning3DWidget(QWidget):
    """
    3D scanner-space planning viewer.

    Coordinate convention:
        X = scanner X
        Y = scanner Y
        Z = scanner Z

    Scene coordinates are meters, matching common.geometry.

    The widget does not own planning geometry.
    It displays and edits the shared PlanningState supplied
    by ExaminationWindow.
    """

    # Emitted after this widget changed the shared
    # PlanningState, so every other planning view can
    # refresh from the SAME object.
    planning_changed = pyqtSignal()

    HOME_CAMERA_DISTANCE_M = 0.45
    HOME_CAMERA_ELEVATION_DEG = 20.0
    HOME_CAMERA_AZIMUTH_DEG = 45.0

    # FOV operation steps.
    #
    # Box3D center coordinates are normalized scanner
    # X/Y/Z, so a 0.01 step is 1% of the reference FOV
    # (about 2 mm for a 20 cm Localizer FOV).
    FOV_TRANSLATION_STEP_NORM = 0.01
    FOV_ROTATION_STEP_DEG = 5.0

    # Mouse drag sensitivity for the right-drag FOV gesture.
    #
    # 10 px of drag == one MOVE toolbar step, and
    # 25 px of drag == one ROT toolbar step.
    FOV_DRAG_TRANSLATION_NORM_PER_PIXEL = 0.001
    FOV_DRAG_ROTATION_DEG_PER_PIXEL = 0.2

    # Orthogonal preset views.
    #
    # Scanner channel convention:
    #
    #   Axial     = XY plane, normal Z
    #   Coronal   = XZ plane, normal Y
    #   Sagittal  = YZ plane, normal X
    #
    # The angles below are derived from that geometry so
    # that the on-screen axes match the 2D Localizer views:
    #
    #   AX   screen right = +X   screen up = +Y
    #   COR  screen right = +X   screen up = +Z
    #   SAG  screen right = +Y   screen up = +Z
    AXIAL_CAMERA_ELEVATION_DEG = 90.0
    AXIAL_CAMERA_AZIMUTH_DEG = -90.0

    CORONAL_CAMERA_ELEVATION_DEG = 0.0
    CORONAL_CAMERA_AZIMUTH_DEG = -90.0

    SAGITTAL_CAMERA_ELEVATION_DEG = 0.0
    SAGITTAL_CAMERA_AZIMUTH_DEG = 0.0

    # FOV box-local face handles.
    #
    # axis index:
    #   0 = box-local X
    #   1 = box-local Y
    #   2 = box-local Z
    #
    # sign = +/- face along that box-local axis.
    FOV_FACE_HANDLE_SPECS = (
        (
            "face_x_neg",
            0,
            -1.0,
        ),
        (
            "face_x_pos",
            0,
            1.0,
        ),
        (
            "face_y_neg",
            1,
            -1.0,
        ),
        (
            "face_y_pos",
            1,
            1.0,
        ),
        (
            "face_z_neg",
            2,
            -1.0,
        ),
        (
            "face_z_pos",
            2,
            1.0,
        ),
    )

    FOV_MOVE_HANDLE_SPECS = (
        (
            "move_x",
            np.array(
                [1.0, 0.0, 0.0],
                dtype=float,
            ),
        ),
        (
            "move_y",
            np.array(
                [0.0, 1.0, 0.0],
                dtype=float,
            ),
        ),
        (
            "move_z",
            np.array(
                [0.0, 0.0, 1.0],
                dtype=float,
            ),
        ),
    )

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

        self.view = (
            PlanningGLViewWidget()
        )

        self.view.planning_dragged.connect(
            self._drag_fov
        )

        self.view.planning_handle_dragged.connect(
            self._drag_planning_handle
        )

        self.view.planning_handle_hovered.connect(
            self._hover_fov_handle
        )

        self.view.setToolTip(
            "Left drag empty space: orbit camera\n"
            "Left drag FOV face handle: resize FOV\n"
            "Alt + Left drag handle: symmetric resize\n"
            "Middle/Ctrl+Left: pan camera\n"
            "Wheel: zoom\n"
            "Right drag: legacy FOV MOVE / ROT"
        )

        self.view.setBackgroundColor(
            (0, 0, 0, 255)
        )

        # -------------------------------------------------
        # Camera controls
        # -------------------------------------------------

        camera_layout = QHBoxLayout()

        camera_layout.setContentsMargins(
            4,
            2,
            4,
            2,
        )

        camera_layout.setSpacing(
            4
        )

        camera_layout.addStretch(
            1
        )

        self.axial_button = (
            QToolButton()
        )

        self.axial_button.setText(
            "AX"
        )

        self.axial_button.setToolTip(
            "Axial camera view"
        )

        self.axial_button.clicked.connect(
            self.set_camera_axial
        )

        camera_layout.addWidget(
            self.axial_button
        )

        self.coronal_button = (
            QToolButton()
        )

        self.coronal_button.setText(
            "COR"
        )

        self.coronal_button.setToolTip(
            "Coronal camera view"
        )

        self.coronal_button.clicked.connect(
            self.set_camera_coronal
        )

        camera_layout.addWidget(
            self.coronal_button
        )

        self.sagittal_button = (
            QToolButton()
        )

        self.sagittal_button.setText(
            "SAG"
        )

        self.sagittal_button.setToolTip(
            "Sagittal camera view"
        )

        self.sagittal_button.clicked.connect(
            self.set_camera_sagittal
        )

        camera_layout.addWidget(
            self.sagittal_button
        )

        self.iso_button = (
            QToolButton()
        )

        self.iso_button.setText(
            "ISO"
        )

        self.iso_button.setToolTip(
            "Isometric camera view"
        )

        self.iso_button.clicked.connect(
            self.set_camera_iso
        )

        camera_layout.addWidget(
            self.iso_button
        )

        self.home_button = (
            QToolButton()
        )

        self.home_button.setText(
            "HOME"
        )

        self.home_button.setToolTip(
            "Reset 3D camera (Home)"
        )

        self.home_button.clicked.connect(
            self.reset_camera
        )

        camera_layout.addWidget(
            self.home_button
        )

        layout.addLayout(
            camera_layout
        )

        # -------------------------------------------------
        # FOV planning controls
        # -------------------------------------------------

        planning_layout = QHBoxLayout()

        planning_layout.setContentsMargins(
            4,
            2,
            4,
            2,
        )

        planning_layout.setSpacing(
            4
        )

        planning_layout.addStretch(
            1
        )

        self.fov_action_combo = (
            QComboBox()
        )

        self.fov_action_combo.addItems(
            [
                "MOVE",
                "ROT",
            ]
        )

        self.fov_action_combo.setToolTip(
            "FOV operation: use +/- or right-drag in 3D"
        )

        self.fov_action_combo.setMinimumWidth(
            76
        )

        self.fov_action_combo.view().setMinimumWidth(
            76
        )

        planning_layout.addWidget(
            self.fov_action_combo
        )

        self.fov_axis_combo = (
            QComboBox()
        )

        self.fov_axis_combo.addItems(
            [
                "X",
                "Y",
                "Z",
            ]
        )

        self.fov_axis_combo.setToolTip(
            "Scanner axis for MOVE / ROT"
        )

        self.fov_axis_combo.setMinimumWidth(
            54
        )

        self.fov_axis_combo.view().setMinimumWidth(
            54
        )

        planning_layout.addWidget(
            self.fov_axis_combo
        )

        self.fov_minus_button = (
            QToolButton()
        )

        self.fov_minus_button.setText(
            "-"
        )

        self.fov_minus_button.setToolTip(
            "Move or rotate FOV in negative direction"
        )

        self.fov_minus_button.clicked.connect(
            lambda: self._nudge_fov(
                -1.0
            )
        )

        planning_layout.addWidget(
            self.fov_minus_button
        )

        self.fov_plus_button = (
            QToolButton()
        )

        self.fov_plus_button.setText(
            "+"
        )

        self.fov_plus_button.setToolTip(
            "Move or rotate FOV in positive direction"
        )

        self.fov_plus_button.clicked.connect(
            lambda: self._nudge_fov(
                1.0
            )
        )

        planning_layout.addWidget(
            self.fov_plus_button
        )

        layout.addLayout(
            planning_layout
        )

        layout.addWidget(
            self.view,
            1,
        )

        self.home_shortcut = (
            QShortcut(
                QKeySequence(
                    Qt.Key_Home
                ),
                self,
            )
        )

        self.home_shortcut.setContext(
            Qt.WidgetWithChildrenShortcut
        )

        self.home_shortcut.activated.connect(
            self.reset_camera
        )

        # -------------------------------------------------
        # Shared planning context
        # -------------------------------------------------

        self.planning_state = None

        # Physical size of the centered scanner-space
        # reference volume used by normalized Box3D.
        self.reference_fov_m = None

        # -------------------------------------------------
        # GL scene items
        # -------------------------------------------------

        self.axis_items = []

        # Fallback colored planning planes shown before real
        # Localizer pixel data is available.
        self.localizer_plane_items = []

        # Real Localizer textures placed in scanner XYZ.
        self.localizer_image_items = []

        # One GLLinePlotItem contains all 12 FOV edges.
        #
        # mode="lines" interprets each consecutive pair
        # of points as one independent line segment.
        # Low-opacity x-ray representation.
        #
        # Depth testing is deliberately disabled so the complete
        # planned FOV remains spatially understandable even when
        # part of it lies behind a Localizer image plane.
        self.fov_wireframe_ghost_item = (
            gl.GLLinePlotItem(
                pos=np.zeros(
                    (24, 3),
                    dtype=float,
                ),
                color=(
                    1.0,
                    1.0,
                    0.0,
                    0.22,
                ),
                width=2,
                antialias=True,
                mode="lines",
                glOptions=(
                    PLANNING_OVERLAY_GL_OPTIONS
                ),
            )
        )

        self.fov_wireframe_ghost_item.hide()

        self.fov_wireframe_ghost_item.setDepthValue(
            0
        )

        self.view.addItem(
            self.fov_wireframe_ghost_item
        )

        # Depth-tested representation.
        #
        # This pass draws only the FOV portions that are visible in
        # front of the Localizer image planes.
        self.fov_wireframe_item = (
            gl.GLLinePlotItem(
                pos=np.zeros(
                    (24, 3),
                    dtype=float,
                ),
                color=(
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                ),
                width=3,
                antialias=True,
                mode="lines",
                glOptions="translucent",
            )
        )

        self.fov_wireframe_item.hide()

        self.fov_wireframe_item.setDepthValue(
            1
        )

        self.view.addItem(
            self.fov_wireframe_item
        )

        # Yellow corner handles marking the eight FOV
        # corners in scanner XYZ.
        #
        # pxMode=True keeps them a constant ~12 px on
        # screen, independent of camera zoom.
        self.fov_corner_handle_items = {}

        self.fov_corner_positions_scanner_m = None

        for corner_index in range(8):

            handle_id = (
                f"corner_{corner_index}"
            )

            item = (
                gl.GLScatterPlotItem(
                    pos=np.zeros(
                        (1, 3),
                        dtype=float,
                    ),
                    color=(
                        1.0,
                        1.0,
                        0.0,
                        1.0,
                    ),
                    size=12.0,
                    pxMode=True,
                    glOptions=(
                        PLANNING_OVERLAY_GL_OPTIONS
                    ),
                )
            )

            item.setDepthValue(
                3
            )

            item.hide()

            self.view.addItem(
                item
            )

            self.view.register_planning_handle(
                item,
                handle_id,
            )

            self.fov_corner_handle_items[
                handle_id
            ] = item

        # One separately pickable handle per FOV face center.
        self.fov_face_handle_items = {}

        self.fov_face_handle_positions_scanner_m = {}

        for (
            handle_id,
            axis_index,
            sign,
        ) in self.FOV_FACE_HANDLE_SPECS:

            item = (
                gl.GLScatterPlotItem(
                    pos=np.zeros(
                        (1, 3),
                        dtype=float,
                    ),
                    color=(
                        1.0,
                        0.75,
                        0.0,
                        1.0,
                    ),
                    size=14.0,
                    pxMode=True,
                    glOptions=(
                        PLANNING_OVERLAY_GL_OPTIONS
                    ),
                )
            )

            item.setDepthValue(
                3
            )

            item.hide()

            self.view.addItem(
                item
            )

            self.view.register_planning_handle(
                item,
                handle_id,
            )

            self.fov_face_handle_items[
                handle_id
            ] = item

        self.fov_move_axis_items = {}
        self.fov_move_handle_items = {}

        for (
            handle_id,
            scanner_axis,
        ) in self.FOV_MOVE_HANDLE_SPECS:

            axis_item = (
                gl.GLLinePlotItem(
                    pos=np.zeros(
                        (2, 3),
                        dtype=float,
                    ),
                    color=(
                        1.0,
                        1.0,
                        1.0,
                        0.75,
                    ),
                    width=3,
                    antialias=True,
                    mode="lines",
                    glOptions=(
                        PLANNING_OVERLAY_GL_OPTIONS
                    ),
                )
            )

            axis_item.setDepthValue(
                2
            )

            axis_item.hide()

            self.view.addItem(
                axis_item
            )

            handle_item = (
                gl.GLScatterPlotItem(
                    pos=np.zeros(
                        (1, 3),
                        dtype=float,
                    ),
                    color=(
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ),
                    size=14.0,
                    pxMode=True,
                    glOptions=(
                        PLANNING_OVERLAY_GL_OPTIONS
                    ),
                )
            )

            handle_item.setDepthValue(
                3
            )

            handle_item.hide()

            self.view.addItem(
                handle_item
            )

            self.view.register_planning_handle(
                handle_item,
                handle_id,
            )

            self.fov_move_axis_items[
                handle_id
            ] = axis_item

            self.fov_move_handle_items[
                handle_id
            ] = handle_item

        # Start from the same camera state used by HOME.
        self.reset_camera()

        self._create_scanner_axes()

    # =====================================================
    # Camera
    # =====================================================

    def _set_camera_view(
        self,
        elevation_deg,
        azimuth_deg,
    ):
        """
        Set a scanner-space camera orientation.

        Preset views recenter on scanner isocenter but preserve
        the user's current zoom/distance and field of view.

        This changes only the camera and never PlanningState.
        """

        self.view.setCameraPosition(
            pos=QVector3D(
                0.0,
                0.0,
                0.0,
            ),
            elevation=float(
                elevation_deg
            ),
            azimuth=float(
                azimuth_deg
            ),
        )

    def set_camera_axial(
        self,
    ):
        """
        Look along the Localizer Axial normal.

        Screen:
            right = scanner +X
            up    = scanner +Y
        """

        self._set_camera_view(
            self.AXIAL_CAMERA_ELEVATION_DEG,
            self.AXIAL_CAMERA_AZIMUTH_DEG,
        )

    def set_camera_coronal(
        self,
    ):
        """
        Look perpendicular to the Coronal X/Z plane.

        Screen:
            right = scanner +X
            up    = scanner +Z
        """

        self._set_camera_view(
            self.CORONAL_CAMERA_ELEVATION_DEG,
            self.CORONAL_CAMERA_AZIMUTH_DEG,
        )

    def set_camera_sagittal(
        self,
    ):
        """
        Look perpendicular to the Sagittal Y/Z plane.

        Screen:
            right = scanner +Y
            up    = scanner +Z
        """

        self._set_camera_view(
            self.SAGITTAL_CAMERA_ELEVATION_DEG,
            self.SAGITTAL_CAMERA_AZIMUTH_DEG,
        )

    def set_camera_iso(
        self,
    ):
        """
        Restore the standard oblique orientation while
        preserving the current zoom.
        """

        self._set_camera_view(
            self.HOME_CAMERA_ELEVATION_DEG,
            self.HOME_CAMERA_AZIMUTH_DEG,
        )

    def reset_camera(
        self,
    ):
        """
        Restore the standard scanner-space 3D camera.

        This changes only the camera. It never changes
        PlanningState or any scan-planning geometry.
        """

        # GLViewWidget.reset() restores:
        #
        #   center = scanner origin
        #   FOV = 60 degrees
        #   viewport = default
        #
        # It also resets distance/orientation to pyqtgraph
        # defaults, which are overridden immediately below
        # by our planning-view home camera.
        self.view.reset()

        self.view.setBackgroundColor(
            (
                0,
                0,
                0,
                255,
            )
        )

        self.view.setCameraPosition(
            distance=(
                self.HOME_CAMERA_DISTANCE_M
            ),
            elevation=(
                self.HOME_CAMERA_ELEVATION_DEG
            ),
            azimuth=(
                self.HOME_CAMERA_AZIMUTH_DEG
            ),
        )

    # =====================================================
    # 3D FOV interaction
    # =====================================================

    def _translate_fov_scanner_axis_delta(
        self,
        axis,
        delta_norm,
    ):
        """
        Translate the planned FOV by an arbitrary normalized
        amount along one scanner/global axis.
        """

        if self.planning_state is None:
            return

        axis = str(
            axis
        ).upper()

        if axis not in (
            "X",
            "Y",
            "Z",
        ):
            return

        delta_norm = float(
            delta_norm
        )

        if not np.isfinite(
            delta_norm
        ):
            return

        box = (
            self.planning_state
            .fov_box
        )

        attribute = (
            "center_"
            + axis.lower()
        )

        setattr(
            box,
            attribute,
            float(
                getattr(
                    box,
                    attribute,
                )
            )
            + delta_norm,
        )

        box.clamp()

    def _translate_fov_scanner_axis(
        self,
        axis,
        direction,
    ):
        """
        Apply one discrete toolbar translation step.
        """

        self._translate_fov_scanner_axis_delta(
            axis,
            float(direction)
            * self.FOV_TRANSLATION_STEP_NORM,
        )

    def _rotate_fov_scanner_axis_delta(
        self,
        axis,
        angle_deg,
    ):
        """
        Rotate the planned FOV by an arbitrary angle around
        one scanner/global axis.

        Scanner-space incremental rotation is pre-multiplied:

            R_new = delta_R @ R_current

        This preserves the repository Euler convention while
        giving the UI a true scanner-axis rotation.
        """

        if self.planning_state is None:
            return

        axis = str(
            axis
        ).upper()

        if axis not in (
            "X",
            "Y",
            "Z",
        ):
            return

        angle_deg = float(
            angle_deg
        )

        if not np.isfinite(
            angle_deg
        ):
            return

        box = (
            self.planning_state
            .fov_box
        )

        current_rotation = (
            planning_euler_to_matrix(
                float(
                    box.rotation_x
                ),
                float(
                    box.rotation_y
                ),
                float(
                    box.rotation_z
                ),
            )
        )

        if axis == "X":

            delta_rotation = (
                planning_euler_to_matrix(
                    angle_deg,
                    0.0,
                    0.0,
                )
            )

        elif axis == "Y":

            delta_rotation = (
                planning_euler_to_matrix(
                    0.0,
                    angle_deg,
                    0.0,
                )
            )

        else:

            delta_rotation = (
                planning_euler_to_matrix(
                    0.0,
                    0.0,
                    angle_deg,
                )
            )

        new_rotation = (
            delta_rotation
            @ current_rotation
        )

        (
            rotation_x,
            rotation_y,
            rotation_z,
        ) = planning_matrix_to_euler(
            new_rotation
        )

        box.rotation_x = float(
            rotation_x
        )

        box.rotation_y = float(
            rotation_y
        )

        box.rotation_z = float(
            rotation_z
        )

        box.clamp()

    def _rotate_fov_scanner_axis(
        self,
        axis,
        direction,
    ):
        """
        Apply one discrete toolbar rotation step.
        """

        self._rotate_fov_scanner_axis_delta(
            axis,
            float(direction)
            * self.FOV_ROTATION_STEP_DEG,
        )

    def _drag_fov(
        self,
        delta_pixels,
    ):
        """
        Apply an axis-constrained mouse drag to the planned FOV.

        Horizontal right-drag is positive along the selected
        scanner axis; left-drag is negative.
        """

        if self.planning_state is None:
            return

        delta_pixels = float(
            delta_pixels
        )

        if (
            not np.isfinite(
                delta_pixels
            )
            or abs(delta_pixels) < 1e-6
        ):
            return

        action = (
            self.fov_action_combo
            .currentText()
        )

        axis = (
            self.fov_axis_combo
            .currentText()
        )

        if action == "MOVE":

            delta_norm = (
                delta_pixels
                * self.FOV_DRAG_TRANSLATION_NORM_PER_PIXEL
            )

            self._translate_fov_scanner_axis_delta(
                axis,
                delta_norm,
            )

        elif action == "ROT":

            angle_deg = (
                delta_pixels
                * self.FOV_DRAG_ROTATION_DEG_PER_PIXEL
            )

            self._rotate_fov_scanner_axis_delta(
                axis,
                angle_deg,
            )

        else:
            return

        self.refresh_planning_geometry()

        self.planning_changed.emit()

    def _face_handle_spec(
        self,
        handle_id,
    ):
        for (
            current_id,
            axis_index,
            sign,
        ) in self.FOV_FACE_HANDLE_SPECS:

            if current_id == handle_id:
                return (
                    axis_index,
                    sign,
                )

        return None

    def _set_handle_visual(
        self,
        item,
        hovered,
    ):
        if hovered:
            item.setData(
                color=(
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ),
                size=18.0,
                pxMode=True,
            )
        else:
            item.setData(
                color=(
                    1.0,
                    0.75,
                    0.0,
                    1.0,
                ),
                size=14.0,
                pxMode=True,
            )

    def _hover_fov_handle(
        self,
        handle_id,
    ):
        """
        Highlight the currently pickable FOV face handle.
        """

        for (
            current_id,
            item,
        ) in (
            self.fov_face_handle_items
            .items()
        ):
            self._set_handle_visual(
                item,
                current_id == handle_id,
            )

        for (
            current_id,
            item,
        ) in (
            self.fov_move_handle_items
            .items()
        ):
            self._set_handle_visual(
                item,
                current_id == handle_id,
            )

        for (
            current_id,
            item,
        ) in (
            self.fov_corner_handle_items
            .items()
        ):
            self._set_handle_visual(
                item,
                current_id == handle_id,
            )

    def _drag_planning_handle(
        self,
        handle_id,
        delta_x_pixels,
        delta_y_pixels,
        symmetric_resize,
    ):
        """
        Dispatch one direct-manipulation handle drag.

        Handle IDs describe UI semantics only; PlanningState
        remains the single source of truth.
        """

        handle_id = str(
            handle_id
        )

        if handle_id.startswith(
            "face_"
        ):
            self._drag_fov_face_handle(
                handle_id,
                delta_x_pixels,
                delta_y_pixels,
                symmetric_resize,
            )
            return

        if handle_id.startswith(
            "move_"
        ):
            self._drag_fov_move_handle(
                handle_id,
                delta_x_pixels,
                delta_y_pixels,
            )
            return

        if handle_id.startswith(
            "corner_"
        ):
            self._drag_fov_corner_handle(
                handle_id,
                delta_x_pixels,
                delta_y_pixels,
                symmetric_resize,
            )

    def _project_scanner_point_to_widget(
        self,
        point_scanner_m,
    ):
        """
        Project one scanner-space 3D point onto widget
        coordinates in pixels.

        Returns:
            np.ndarray([x, y])

        or None if the point cannot be projected.
        """

        point = np.asarray(
            point_scanner_m,
            dtype=float,
        )

        if (
            point.shape != (3,)
            or not np.all(
                np.isfinite(
                    point
                )
            )
        ):
            return None

        world_point = QVector4D(
            float(
                point[0]
            ),
            float(
                point[1]
            ),
            float(
                point[2]
            ),
            1.0,
        )

        clip_point = (
            self.view.projectionMatrix()
            * self.view.viewMatrix()
            * world_point
        )

        w = float(
            clip_point.w()
        )

        if (
            not np.isfinite(
                w
            )
            or w <= 1e-9
        ):
            return None

        ndc_x = (
            float(
                clip_point.x()
            )
            / w
        )

        ndc_y = (
            float(
                clip_point.y()
            )
            / w
        )

        return np.array(
            [
                (
                    ndc_x + 1.0
                )
                * 0.5
                * float(
                    self.view.width()
                ),
                (
                    1.0 - ndc_y
                )
                * 0.5
                * float(
                    self.view.height()
                ),
            ],
            dtype=float,
        )

    def _drag_fov_face_handle(
        self,
        handle_id,
        delta_x_pixels,
        delta_y_pixels,
        symmetric_resize,
    ):
        """
        Resize one FOV box-local axis by dragging a face handle.

        The opposite face remains fixed.

        Mouse motion is projected onto the selected face's
        box-local outward normal as seen by the current camera.
        """

        if (
            self.planning_state is None
            or self.reference_fov_m is None
        ):
            return

        spec = (
            self._face_handle_spec(
                handle_id
            )
        )

        if spec is None:
            return

        (
            axis_index,
            sign,
        ) = spec

        box = (
            self.planning_state
            .fov_box
        )

        reference_fov_m = np.full(
            3,
            float(
                self.reference_fov_m
            ),
            dtype=float,
        )

        scan_geometry = (
            planning_box_to_scan_geometry(
                fov_box=(
                    box.as_dict()
                ),
                reference_fov_mm=(
                    reference_fov_m
                    * 1000.0
                ),
            )
        )

        rotation = np.asarray(
            scan_geometry.rotation_local_to_scanner,
            dtype=float,
        )

        local_axis = (
            rotation[
                :,
                axis_index,
            ]
        )

        outward_axis = (
            float(sign)
            * local_axis
        )

        face_position = (
            np.asarray(
                scan_geometry.center_scanner_m,
                dtype=float,
            )
            + outward_axis
            * 0.5
            * float(
                scan_geometry.fov_local_m[
                    axis_index
                ]
            )
        )

        # Project the selected local-axis direction through the
        # full camera + perspective transform.
        #
        # This measures the actual screen-space direction and
        # pixels-per-meter of this handle at its current depth.

        probe_length_m = max(
            1e-4,
            0.02
            * float(
                scan_geometry.fov_local_m[
                    axis_index
                ]
            ),
        )

        screen_face = (
            self._project_scanner_point_to_widget(
                face_position
            )
        )

        screen_probe = (
            self._project_scanner_point_to_widget(
                face_position
                + outward_axis
                * probe_length_m
            )
        )

        if (
            screen_face is None
            or screen_probe is None
        ):
            return

        screen_axis_vector = (
            screen_probe
            - screen_face
        )

        screen_axis_pixels = float(
            np.linalg.norm(
                screen_axis_vector
            )
        )

        # If we are looking almost exactly along this axis,
        # a 2D mouse drag cannot control it reliably.
        if screen_axis_pixels < 1e-3:
            return

        screen_axis = (
            screen_axis_vector
            / screen_axis_pixels
        )

        mouse_delta = np.array(
            [
                float(
                    delta_x_pixels
                ),
                float(
                    delta_y_pixels
                ),
            ],
            dtype=float,
        )

        delta_pixels = float(
            np.dot(
                mouse_delta,
                screen_axis,
            )
        )

        pixels_per_meter = (
            screen_axis_pixels
            / probe_length_m
        )

        if (
            not np.isfinite(
                pixels_per_meter
            )
            or pixels_per_meter < 1e-6
        ):
            return

        outward_delta_m = (
            delta_pixels
            / pixels_per_meter
        )

        size_attribute = (
            (
                "size_x",
                "size_y",
                "size_z",
            )[
                axis_index
            ]
        )

        if symmetric_resize:

            # Selected face moves by d and the opposite face moves
            # by -d, therefore total box size changes by 2*d.
            size_delta_norm = (
                2.0
                * outward_delta_m
                / reference_fov_m[
                    axis_index
                ]
            )

        else:

            # Selected face moves while the opposite face stays
            # fixed.
            size_delta_norm = (
                outward_delta_m
                / reference_fov_m[
                    axis_index
                ]
            )

        current_size = float(
            getattr(
                box,
                size_attribute,
            )
        )

        new_size = float(
            np.clip(
                current_size
                + size_delta_norm,
                0.02,
                1.0,
            )
        )

        actual_size_delta_norm = (
            new_size
            - current_size
        )

        if abs(
            actual_size_delta_norm
        ) < 1e-12:
            return

        actual_delta_m = (
            actual_size_delta_norm
            * reference_fov_m[
                axis_index
            ]
        )

        candidate_center_norm = np.array(
            [
                float(
                    box.center_x
                ),
                float(
                    box.center_y
                ),
                float(
                    box.center_z
                ),
            ],
            dtype=float,
        )

        if not symmetric_resize:

            center_delta_norm = (
                0.5
                * actual_delta_m
                * outward_axis
                / reference_fov_m
            )

            candidate_center_norm += (
                center_delta_norm
            )

        candidate_sizes = np.array(
            [
                float(
                    box.size_x
                ),
                float(
                    box.size_y
                ),
                float(
                    box.size_z
                ),
            ],
            dtype=float,
        )

        candidate_sizes[
            axis_index
        ] = new_size

        minimum_center = (
            0.5
            * candidate_sizes
        )

        maximum_center = (
            1.0
            - 0.5
            * candidate_sizes
        )

        tolerance = 1e-12

        if (
            np.any(
                candidate_center_norm
                < minimum_center
                - tolerance
            )
            or np.any(
                candidate_center_norm
                > maximum_center
                + tolerance
            )
        ):
            return

        setattr(
            box,
            size_attribute,
            new_size,
        )

        box.center_x = float(
            candidate_center_norm[0]
        )

        box.center_y = float(
            candidate_center_norm[1]
        )

        box.center_z = float(
            candidate_center_norm[2]
        )

        self.refresh_planning_geometry()

        self.planning_changed.emit()

    def _move_handle_axis(
        self,
        handle_id,
    ):
        for (
            current_id,
            scanner_axis,
        ) in self.FOV_MOVE_HANDLE_SPECS:

            if current_id == handle_id:
                return np.asarray(
                    scanner_axis,
                    dtype=float,
                )

        return None

    def _drag_fov_move_handle(
        self,
        handle_id,
        delta_x_pixels,
        delta_y_pixels,
    ):
        """
        Translate the complete planned FOV along one scanner
        X/Y/Z axis using the projected screen direction of
        that axis.
        """

        if (
            self.planning_state is None
            or self.reference_fov_m is None
        ):
            return

        scanner_axis = (
            self._move_handle_axis(
                handle_id
            )
        )

        if scanner_axis is None:
            return

        box = (
            self.planning_state
            .fov_box
        )

        reference_fov_m = np.full(
            3,
            float(
                self.reference_fov_m
            ),
            dtype=float,
        )

        scan_geometry = (
            planning_box_to_scan_geometry(
                fov_box=(
                    box.as_dict()
                ),
                reference_fov_mm=(
                    reference_fov_m
                    * 1000.0
                ),
            )
        )

        center = np.asarray(
            scan_geometry.center_scanner_m,
            dtype=float,
        )

        probe_length_m = 0.02

        screen_center = (
            self._project_scanner_point_to_widget(
                center
            )
        )

        screen_probe = (
            self._project_scanner_point_to_widget(
                center
                + scanner_axis
                * probe_length_m
            )
        )

        if (
            screen_center is None
            or screen_probe is None
        ):
            return

        screen_axis_vector = (
            screen_probe
            - screen_center
        )

        screen_axis_pixels = float(
            np.linalg.norm(
                screen_axis_vector
            )
        )

        # Camera is looking almost exactly along this scanner
        # axis, so 2D dragging cannot determine a stable amount.
        if screen_axis_pixels < 1e-3:
            return

        screen_axis = (
            screen_axis_vector
            / screen_axis_pixels
        )

        mouse_delta = np.array(
            [
                float(
                    delta_x_pixels
                ),
                float(
                    delta_y_pixels
                ),
            ],
            dtype=float,
        )

        constrained_pixels = float(
            np.dot(
                mouse_delta,
                screen_axis,
            )
        )

        pixels_per_meter = (
            screen_axis_pixels
            / probe_length_m
        )

        if (
            not np.isfinite(
                pixels_per_meter
            )
            or pixels_per_meter < 1e-6
        ):
            return

        delta_m = (
            constrained_pixels
            / pixels_per_meter
        )

        center_delta_norm = (
            delta_m
            * scanner_axis
            / reference_fov_m
        )

        box.center_x += float(
            center_delta_norm[0]
        )

        box.center_y += float(
            center_delta_norm[1]
        )

        box.center_z += float(
            center_delta_norm[2]
        )

        box.clamp()

        self.refresh_planning_geometry()

        self.planning_changed.emit()

    @staticmethod
    def _corner_handle_index(
        handle_id,
    ):
        prefix = "corner_"

        if not str(
            handle_id
        ).startswith(
            prefix
        ):
            return None

        try:
            corner_index = int(
                str(
                    handle_id
                )[
                    len(prefix):
                ]
            )
        except ValueError:
            return None

        if not (
            0
            <= corner_index
            < 8
        ):
            return None

        return corner_index

    def _drag_fov_corner_handle(
        self,
        handle_id,
        delta_x_pixels,
        delta_y_pixels,
        symmetric_resize,
    ):
        """
        Uniformly resize the FOV using one corner.

        Normal:
            opposite corner stays fixed.

        Alt:
            FOV center stays fixed.
        """

        if (
            self.planning_state is None
            or self.reference_fov_m is None
        ):
            return

        corner_index = (
            self._corner_handle_index(
                handle_id
            )
        )

        if corner_index is None:
            return

        box = (
            self.planning_state
            .fov_box
        )

        reference_fov_m = np.full(
            3,
            float(
                self.reference_fov_m
            ),
            dtype=float,
        )

        scan_geometry = (
            planning_box_to_scan_geometry(
                fov_box=(
                    box.as_dict()
                ),
                reference_fov_mm=(
                    reference_fov_m
                    * 1000.0
                ),
            )
        )

        corners = (
            scan_geometry_box_corners_scanner_m(
                scan_geometry
            )
        )

        center = np.asarray(
            scan_geometry.center_scanner_m,
            dtype=float,
        )

        corner = np.asarray(
            corners[
                corner_index
            ],
            dtype=float,
        )

        # The geometrically opposite corner does not require
        # knowledge of the canonical corner index table.
        opposite_corner = (
            2.0 * center
            - corner
        )

        diagonal = (
            corner
            - opposite_corner
        )

        diagonal_length_m = float(
            np.linalg.norm(
                diagonal
            )
        )

        if diagonal_length_m < 1e-9:
            return

        diagonal_axis = (
            diagonal
            / diagonal_length_m
        )

        screen_corner = (
            self._project_scanner_point_to_widget(
                corner
            )
        )

        screen_opposite = (
            self._project_scanner_point_to_widget(
                opposite_corner
            )
        )

        if (
            screen_corner is None
            or screen_opposite is None
        ):
            return

        screen_diagonal = (
            screen_corner
            - screen_opposite
        )

        screen_diagonal_length = float(
            np.linalg.norm(
                screen_diagonal
            )
        )

        if screen_diagonal_length < 1e-3:
            return

        screen_axis = (
            screen_diagonal
            / screen_diagonal_length
        )

        mouse_delta = np.array(
            [
                float(
                    delta_x_pixels
                ),
                float(
                    delta_y_pixels
                ),
            ],
            dtype=float,
        )

        delta_pixels = float(
            np.dot(
                mouse_delta,
                screen_axis,
            )
        )

        pixels_per_meter = (
            screen_diagonal_length
            / diagonal_length_m
        )

        if (
            not np.isfinite(
                pixels_per_meter
            )
            or pixels_per_meter < 1e-6
        ):
            return

        corner_delta_m = (
            delta_pixels
            / pixels_per_meter
        )

        current_sizes_norm = np.array(
            [
                float(
                    box.size_x
                ),
                float(
                    box.size_y
                ),
                float(
                    box.size_z
                ),
            ],
            dtype=float,
        )

        if symmetric_resize:

            # Center fixed:
            # selected corner moves d,
            # opposite corner moves -d.
            target_diagonal_length_m = (
                diagonal_length_m
                + 2.0
                * corner_delta_m
            )

        else:

            # Opposite corner fixed:
            # selected corner alone moves d.
            target_diagonal_length_m = (
                diagonal_length_m
                + corner_delta_m
            )

        raw_scale = (
            target_diagonal_length_m
            / diagonal_length_m
        )

        minimum_scale = float(
            np.max(
                0.02
                / current_sizes_norm
            )
        )

        maximum_scale = float(
            np.min(
                1.0
                / current_sizes_norm
            )
        )

        scale = float(
            np.clip(
                raw_scale,
                minimum_scale,
                maximum_scale,
            )
        )

        if abs(
            scale - 1.0
        ) < 1e-12:
            return

        candidate_sizes = (
            current_sizes_norm
            * scale
        )

        if symmetric_resize:

            candidate_center_scanner_m = (
                center.copy()
            )

        else:

            # opposite is fixed:
            #
            # new selected =
            #     opposite + scale * old diagonal
            #
            # center is the midpoint.
            candidate_center_scanner_m = (
                opposite_corner
                + 0.5
                * scale
                * diagonal
            )

        candidate_center_norm = (
            0.5
            + candidate_center_scanner_m
            / reference_fov_m
        )

        minimum_center = (
            0.5
            * candidate_sizes
        )

        maximum_center = (
            1.0
            - 0.5
            * candidate_sizes
        )

        tolerance = 1e-12

        if (
            np.any(
                candidate_center_norm
                < minimum_center
                - tolerance
            )
            or np.any(
                candidate_center_norm
                > maximum_center
                + tolerance
            )
        ):
            return

        box.size_x = float(
            candidate_sizes[0]
        )

        box.size_y = float(
            candidate_sizes[1]
        )

        box.size_z = float(
            candidate_sizes[2]
        )

        box.center_x = float(
            candidate_center_norm[0]
        )

        box.center_y = float(
            candidate_center_norm[1]
        )

        box.center_z = float(
            candidate_center_norm[2]
        )

        self.refresh_planning_geometry()

        self.planning_changed.emit()

    def _nudge_fov(
        self,
        direction,
    ):
        """
        Apply one toolbar FOV operation and notify all
        planning views through the shared PlanningState.
        """

        if self.planning_state is None:
            return

        action = (
            self.fov_action_combo
            .currentText()
        )

        axis = (
            self.fov_axis_combo
            .currentText()
        )

        if action == "MOVE":

            self._translate_fov_scanner_axis(
                axis,
                direction,
            )

        elif action == "ROT":

            self._rotate_fov_scanner_axis(
                axis,
                direction,
            )

        else:
            return

        # Update this viewer immediately.
        self.refresh_planning_geometry()

        # ExaminationWindow will refresh all three
        # Localizer planning views from the SAME
        # PlanningState object.
        self.planning_changed.emit()

    # =====================================================
    # Static scanner reference
    # =====================================================

    def _create_scanner_axes(
        self,
    ):
        """
        Draw positive scanner X/Y/Z axes from isocenter.
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

    # =====================================================
    # Planning context
    # =====================================================

    def set_planning_context(
        self,
        planning_state,
        reference_fov_m,
    ):
        """
        Attach the shared PlanningState and physical
        Localizer reference FOV.

        The PlanningState object is shared; this widget does
        not copy it.
        """

        reference_fov_m = float(
            reference_fov_m
        )

        if (
            not np.isfinite(
                reference_fov_m
            )
            or reference_fov_m <= 0.0
        ):
            raise ValueError(
                "Planning 3D reference FOV "
                "must be finite and positive"
            )

        self.planning_state = (
            planning_state
        )

        self.reference_fov_m = (
            reference_fov_m
        )

        # A newly loaded Localizer must never retain textures
        # belonging to the previous planning source.
        self._clear_localizer_image_items()

        # Until the real images arrive, keep the existing
        # colored center planes as a fallback.
        self._create_localizer_planes()

        self.refresh_planning_geometry()

    def clear_planning_context(
        self,
    ):
        self.planning_state = None
        self.reference_fov_m = None

        self._clear_localizer_planes()
        self._clear_localizer_image_items()

        self.fov_wireframe_item.hide()
        self.fov_wireframe_ghost_item.hide()

        for item in (
            self.fov_corner_handle_items
            .values()
        ):
            item.hide()

        self.fov_corner_positions_scanner_m = None

        for item in (
            self.fov_face_handle_items
            .values()
        ):
            item.hide()

        self.fov_face_handle_positions_scanner_m = {}

        for item in (
            self.fov_move_axis_items
            .values()
        ):
            item.hide()

        for item in (
            self.fov_move_handle_items
            .values()
        ):
            item.hide()

        self.view.clear_planning_handle_state()

    # =====================================================
    # Localizer planes
    # =====================================================

    def _clear_localizer_image_items(
        self,
    ):
        """
        Remove scanner-space Localizer image textures.
        """

        for item in (
            self.localizer_image_items
        ):
            try:
                self.view.removeItem(
                    item
                )
            except Exception:
                pass

        self.localizer_image_items = []

    @staticmethod
    def _localizer_image_to_rgba(
        image_array,
        alpha=150,
    ):
        """
        Convert a 2D Localizer magnitude image into the
        uint8 RGBA layout expected by GLImageItem.

        Input image convention:
            shape = (rows, columns)

        GLImageItem convention:
            shape = (x, y, RGBA)

        Therefore the grayscale array is deliberately
        transposed so that:
            GL x == image column
            GL y == image row
        """

        image = np.asarray(
            image_array,
            dtype=float,
        )

        if image.ndim != 2:
            raise ValueError(
                "Localizer texture must be a 2D image"
            )

        finite = np.isfinite(
            image
        )

        gray = np.zeros(
            image.shape,
            dtype=np.uint8,
        )

        if np.any(finite):

            minimum = float(
                np.min(
                    image[finite]
                )
            )

            maximum = float(
                np.max(
                    image[finite]
                )
            )

            if maximum > minimum:

                scaled = (
                    image - minimum
                ) / (
                    maximum - minimum
                )

                scaled[
                    ~finite
                ] = 0.0

                gray = np.clip(
                    np.rint(
                        scaled * 255.0
                    ),
                    0.0,
                    255.0,
                ).astype(
                    np.uint8
                )

        # GLImageItem uses its first data dimension as
        # local X and its second dimension as local Y.
        gray_xy = (
            gray.T
        )

        rgba = np.empty(
            (
                gray_xy.shape[0],
                gray_xy.shape[1],
                4,
            ),
            dtype=np.uint8,
        )

        rgba[
            ...,
            0
        ] = gray_xy

        rgba[
            ...,
            1
        ] = gray_xy

        rgba[
            ...,
            2
        ] = gray_xy

        rgba[
            ...,
            3
        ] = np.uint8(
            alpha
        )

        return np.ascontiguousarray(
            rgba
        )

    @staticmethod
    def _localizer_image_transform(
        image_plane,
    ):
        """
        Build the exact transform from GLImageItem local
        pixel coordinates into scanner X/Y/Z.

        A GL point [x, y, 0] is mapped to the same scanner
        point as ImagePlaneGeometry.image_xy_to_scanner_m(
        [x, y]
        ).

        The explicit horizontal / vertical / normal basis
        is preserved. No cross product is used, which is
        important for the repository's Coronal convention.
        """

        horizontal = np.asarray(
            image_plane.horizontal_scanner,
            dtype=float,
        )

        vertical = np.asarray(
            image_plane.vertical_scanner,
            dtype=float,
        )

        normal = np.asarray(
            image_plane.normal_scanner,
            dtype=float,
        )

        center = np.asarray(
            image_plane.center_scanner_m,
            dtype=float,
        )

        pixel_x_m = (
            float(
                image_plane.width_m
            )
            / float(
                image_plane.columns
            )
        )

        pixel_y_m = (
            float(
                image_plane.height_m
            )
            / float(
                image_plane.rows
            )
        )

        # GLImageItem's local origin is image [0, 0].
        #
        # ImagePlaneGeometry defines the image center as
        # scanner-space center, so [0,0] is half an FOV
        # toward -horizontal and -vertical.
        origin = (
            center
            - 0.5
            * float(
                image_plane.width_m
            )
            * horizontal
            - 0.5
            * float(
                image_plane.height_m
            )
            * vertical
        )

        transform = Transform3D()

        # Matrix columns are the scanner-space images of
        # local GL X/Y/Z basis vectors.
        transform.setColumn(
            0,
            QVector4D(
                float(
                    horizontal[0]
                    * pixel_x_m
                ),
                float(
                    horizontal[1]
                    * pixel_x_m
                ),
                float(
                    horizontal[2]
                    * pixel_x_m
                ),
                0.0,
            ),
        )

        transform.setColumn(
            1,
            QVector4D(
                float(
                    vertical[0]
                    * pixel_y_m
                ),
                float(
                    vertical[1]
                    * pixel_y_m
                ),
                float(
                    vertical[2]
                    * pixel_y_m
                ),
                0.0,
            ),
        )

        # GLImageItem itself lies at local z=0, but keeping
        # the explicit normal as the third basis column makes
        # the transform complete and preserves handedness.
        transform.setColumn(
            2,
            QVector4D(
                float(
                    normal[0]
                ),
                float(
                    normal[1]
                ),
                float(
                    normal[2]
                ),
                0.0,
            ),
        )

        transform.setColumn(
            3,
            QVector4D(
                float(
                    origin[0]
                ),
                float(
                    origin[1]
                ),
                float(
                    origin[2]
                ),
                1.0,
            ),
        )

        return transform

    def set_localizer_images(
        self,
        plane_images,
    ):
        """
        Display real Localizer images in scanner XYZ.

        plane_images contains entries returned by:
            ViewerWidget.get_localizer_plane_image()

        Each entry:
            (
                orientation,
                image_array,
                ImagePlaneGeometry,
            )

        Orientation is descriptive only. Physical placement
        comes entirely from ImagePlaneGeometry.
        """

        self._clear_localizer_image_items()

        image_count = 0

        for plane_image in plane_images:

            if plane_image is None:
                continue

            (
                orientation,
                image_array,
                image_plane,
            ) = plane_image

            image_array = np.asarray(
                image_array
            )

            expected_shape = (
                int(
                    image_plane.rows
                ),
                int(
                    image_plane.columns
                ),
            )

            if (
                image_array.shape
                != expected_shape
            ):
                raise ValueError(
                    "Localizer image / geometry shape "
                    "mismatch for "
                    + str(
                        orientation
                    )
                    + ": image "
                    + str(
                        image_array.shape
                    )
                    + ", geometry "
                    + str(
                        expected_shape
                    )
                )

            rgba = (
                self._localizer_image_to_rgba(
                    image_array
                )
            )

            image_item = (
                gl.GLImageItem(
                    rgba,
                    smooth=True,
                    glOptions="translucent",
                )
            )

            image_item.setTransform(
                self._localizer_image_transform(
                    image_plane
                )
            )

            # Draw image planes before the yellow planning
            # wireframe.
            image_item.setDepthValue(
                -10
            )

            self.view.addItem(
                image_item
            )

            self.localizer_image_items.append(
                image_item
            )

            image_count += 1

        if image_count > 0:
            # Real MRI images replace the temporary colored
            # center-plane meshes.
            self._clear_localizer_planes()

        elif (
            self.reference_fov_m
            is not None
        ):
            # Keep the old geometry-only view as fallback.
            self._create_localizer_planes()

    def _clear_localizer_planes(
        self,
    ):
        for item in (
            self.localizer_plane_items
        ):
            try:
                self.view.removeItem(
                    item
                )
            except Exception:
                pass

        self.localizer_plane_items = []

    def _create_localizer_planes(
        self,
    ):
        """
        Draw the three centered Localizer center planes.

        Current Localizer geometry:
            Axial    = XY, z=0
            Coronal  = XZ, y=0
            Sagittal = YZ, x=0

        These are center planes only. The acquired 10 mm
        slice thickness is deliberately not rendered here.
        """

        self._clear_localizer_planes()

        if (
            self.reference_fov_m
            is None
        ):
            return

        half = (
            0.5
            * float(
                self.reference_fov_m
            )
        )

        axial = np.array(
            [
                [-half, -half, 0.0],
                [ half, -half, 0.0],
                [ half,  half, 0.0],
                [-half,  half, 0.0],
            ],
            dtype=float,
        )

        coronal = np.array(
            [
                [-half, 0.0, -half],
                [ half, 0.0, -half],
                [ half, 0.0,  half],
                [-half, 0.0,  half],
            ],
            dtype=float,
        )

        sagittal = np.array(
            [
                [0.0, -half, -half],
                [0.0,  half, -half],
                [0.0,  half,  half],
                [0.0, -half,  half],
            ],
            dtype=float,
        )

        plane_specs = (
            (
                axial,
                (
                    0.2,
                    0.5,
                    1.0,
                    0.10,
                ),
                (
                    0.2,
                    0.5,
                    1.0,
                    0.55,
                ),
            ),
            (
                coronal,
                (
                    0.2,
                    1.0,
                    0.5,
                    0.10,
                ),
                (
                    0.2,
                    1.0,
                    0.5,
                    0.55,
                ),
            ),
            (
                sagittal,
                (
                    1.0,
                    0.3,
                    0.7,
                    0.10,
                ),
                (
                    1.0,
                    0.3,
                    0.7,
                    0.55,
                ),
            ),
        )

        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ],
            dtype=int,
        )

        for (
            vertices,
            face_color,
            edge_color,
        ) in plane_specs:

            mesh = gl.GLMeshItem(
                vertexes=vertices,
                faces=faces,
                color=face_color,
                smooth=False,
                drawEdges=False,
                glOptions="translucent",
            )

            outline_points = (
                np.vstack(
                    [
                        vertices,
                        vertices[0],
                    ]
                )
            )

            outline = (
                gl.GLLinePlotItem(
                    pos=outline_points,
                    color=edge_color,
                    width=1,
                    antialias=True,
                    mode="line_strip",
                )
            )

            self.view.addItem(
                mesh
            )

            self.view.addItem(
                outline
            )

            self.localizer_plane_items.extend(
                [
                    mesh,
                    outline,
                ]
            )

    # =====================================================
    # Dynamic FOV
    # =====================================================

    def refresh_planning_geometry(
        self,
    ):
        """
        Rebuild the displayed 3D FOV wireframe from the
        shared PlanningState.

        No geometry is inferred from any 2D ROI.
        """

        if (
            self.planning_state
            is None
        ):
            self.fov_wireframe_item.hide()
            self.fov_wireframe_ghost_item.hide()

            for item in (
                self.fov_corner_handle_items
                .values()
            ):
                item.hide()

            for item in (
                self.fov_face_handle_items
                .values()
            ):
                item.hide()

            for item in (
                self.fov_move_axis_items
                .values()
            ):
                item.hide()

            for item in (
                self.fov_move_handle_items
                .values()
            ):
                item.hide()

            return

        if (
            self.reference_fov_m
            is None
        ):
            self.fov_wireframe_item.hide()
            self.fov_wireframe_ghost_item.hide()

            for item in (
                self.fov_corner_handle_items
                .values()
            ):
                item.hide()

            for item in (
                self.fov_face_handle_items
                .values()
            ):
                item.hide()

            for item in (
                self.fov_move_axis_items
                .values()
            ):
                item.hide()

            for item in (
                self.fov_move_handle_items
                .values()
            ):
                item.hide()

            return

        reference_fov_mm = np.full(
            3,
            float(
                self.reference_fov_m
            )
            * 1000.0,
            dtype=float,
        )

        scan_geometry = (
            planning_box_to_scan_geometry(
                fov_box=(
                    self.planning_state
                    .fov_box
                    .as_dict()
                ),
                reference_fov_mm=(
                    reference_fov_mm
                ),
            )
        )

        edges = (
            scan_geometry_box_edges_scanner_m(
                scan_geometry
            )
        )

        # GLLinePlotItem mode="lines":
        #
        #   p0,p1 = edge 0
        #   p2,p3 = edge 1
        #   ...
        #
        # Shape becomes:
        #   (24, 3)
        edge_points = (
            edges.reshape(
                -1,
                3,
            )
        )

        self.fov_wireframe_ghost_item.setData(
            pos=edge_points,
            color=(
                1.0,
                1.0,
                0.0,
                0.22,
            ),
            width=2,
            antialias=True,
            mode="lines",
        )

        self.fov_wireframe_ghost_item.show()

        self.fov_wireframe_item.setData(
            pos=edge_points,
            color=(
                1.0,
                1.0,
                0.0,
                1.0,
            ),
            width=3,
            antialias=True,
            mode="lines",
        )

        self.fov_wireframe_item.show()

        # Same canonical geometry, second representation:
        # the eight handles are the FOV corners used later
        # for direct 3D interaction.
        corners = (
            scan_geometry_box_corners_scanner_m(
                scan_geometry
            )
        )

        self.fov_corner_positions_scanner_m = (
            np.asarray(
                corners,
                dtype=float,
            ).copy()
        )

        for corner_index in range(8):

            handle_id = (
                f"corner_{corner_index}"
            )

            corner_position = np.asarray(
                corners[
                    corner_index
                ],
                dtype=float,
            )

            self.fov_corner_handle_items[
                handle_id
            ].setData(
                pos=np.asarray(
                    [
                        corner_position
                    ],
                    dtype=float,
                ),
            )

            self.fov_corner_handle_items[
                handle_id
            ].show()

        center = np.asarray(
            scan_geometry.center_scanner_m,
            dtype=float,
        )

        sizes = np.asarray(
            scan_geometry.fov_local_m,
            dtype=float,
        )

        rotation = np.asarray(
            scan_geometry.rotation_local_to_scanner,
            dtype=float,
        )

        for (
            handle_id,
            axis_index,
            sign,
        ) in self.FOV_FACE_HANDLE_SPECS:

            local_axis_scanner = (
                rotation[
                    :,
                    axis_index,
                ]
            )

            face_position = (
                center
                + float(sign)
                * 0.5
                * sizes[
                    axis_index
                ]
                * local_axis_scanner
            )

            self.fov_face_handle_positions_scanner_m[
                handle_id
            ] = (
                face_position.copy()
            )

            self.fov_face_handle_items[
                handle_id
            ].setData(
                pos=np.asarray(
                    [
                        face_position
                    ],
                    dtype=float,
                ),
            )

            self.fov_face_handle_items[
                handle_id
            ].show()

        move_handle_length_m = max(
            0.025,
            0.20
            * float(
                np.max(
                    scan_geometry.fov_local_m
                )
            ),
        )

        for (
            handle_id,
            scanner_axis,
        ) in self.FOV_MOVE_HANDLE_SPECS:

            scanner_axis = np.asarray(
                scanner_axis,
                dtype=float,
            )

            endpoint = (
                center
                + scanner_axis
                * move_handle_length_m
            )

            self.fov_move_axis_items[
                handle_id
            ].setData(
                pos=np.asarray(
                    [
                        center,
                        endpoint,
                    ],
                    dtype=float,
                ),
            )

            self.fov_move_handle_items[
                handle_id
            ].setData(
                pos=np.asarray(
                    [
                        endpoint
                    ],
                    dtype=float,
                ),
            )

            self.fov_move_axis_items[
                handle_id
            ].show()

            self.fov_move_handle_items[
                handle_id
            ].show()
