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

from common.geometry import (
    planning_box_to_scan_geometry,
    scan_geometry_box_edges_scanner_m,
    planning_euler_to_matrix,
    planning_matrix_to_euler,
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
            "3D FOV operation"
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
            "Scanner axis"
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
            )
        )

        self.fov_wireframe_item.hide()

        self.view.addItem(
            self.fov_wireframe_item
        )

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

    def _translate_fov_scanner_axis(
        self,
        axis,
        direction,
    ):
        """
        Translate the planned FOV along one scanner axis.

        Box3D center coordinates are normalized scanner
        X/Y/Z coordinates.
        """

        if self.planning_state is None:
            return

        box = (
            self.planning_state
            .fov_box
        )

        step = (
            float(direction)
            * self.FOV_TRANSLATION_STEP_NORM
        )

        attribute = (
            "center_"
            + str(axis).lower()
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
            + step,
        )

        box.clamp()

    def _rotate_fov_scanner_axis(
        self,
        axis,
        direction,
    ):
        """
        Rotate the planned FOV around a scanner/global axis.

        The incremental scanner rotation is pre-multiplied:

            R_new = delta_R @ R_current

        This preserves the repository Euler convention while
        giving the UI a true scanner-axis rotation.
        """

        if self.planning_state is None:
            return

        box = (
            self.planning_state
            .fov_box
        )

        angle_deg = (
            float(direction)
            * self.FOV_ROTATION_STEP_DEG
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

        elif axis == "Z":

            delta_rotation = (
                planning_euler_to_matrix(
                    0.0,
                    0.0,
                    angle_deg,
                )
            )

        else:
            return

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
        alpha=185,
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
            return

        if (
            self.reference_fov_m
            is None
        ):
            self.fov_wireframe_item.hide()
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
