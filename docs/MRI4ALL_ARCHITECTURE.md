# MRI4ALL Console Architecture Guide
## Localizer Planning / Oblique 3D GRE Development

**Repository:** `ziximu7-gif/mri4all-console-dev`  
**Primary development branch:** `feature/localizer-planning`  
**Baseline used for this document:** `b899b0bf33459127a58bf6c09533493b5c1184f2`

> This document is an architecture guide, not an instruction to blindly trust the baseline commit.
> Before modifying code, always inspect the current local `HEAD`, `git status`, and relevant files.
> If the current code differs from this document, stop and identify the difference before changing behavior.

---

# 1. Architectural authority

For this project, architecture decisions remain human-owned.

A coding agent may:

- inspect the repository;
- trace call paths and data flow;
- propose a minimal implementation plan;
- implement an explicitly approved design;
- add focused tests;
- report risks, inconsistencies, and architectural debt.

A coding agent must **not** silently:

- redesign coordinate conventions;
- reinterpret the meaning of FOV, Orientation, Readout Direction, Phase Direction, or Shim;
- change persistent `ScanTask` data contracts;
- resample reconstructed images into a different coordinate system;
- alter acquisition/reconstruction workflow boundaries;
- modify vendored/local PyPulseq behavior without a targeted regression test;
- expand a task into TSE3D or unrelated sequences;
- change patient-space/DICOM orientation assumptions;
- "clean up" large areas of code while implementing a focused feature.

If a task requires one of these decisions, the agent should stop and ask for an architectural decision.

---

# 2. High-level system architecture

MRI4ALL Console is organized around a folder-backed scan task that moves through UI, acquisition, reconstruction, and completion states.

```text
                 +--------------------+
                 |        UI          |
                 | services/ui        |
                 +---------+----------+
                           |
                           | create/edit ScanTask
                           v
                 +--------------------+
                 |  ACQ queue folder  |
                 | scan.json + files  |
                 +---------+----------+
                           | PREPARED
                           v
                 +--------------------+
                 | Acquisition service|
                 | services/acq       |
                 +---------+----------+
                           |
                           | SequenceBase
                           | calculate_sequence()
                           | run_sequence()
                           v
                 +--------------------+
                 | rawdata / seq      |
                 +---------+----------+
                           |
                           v
                 +--------------------+
                 | RECON queue folder |
                 +---------+----------+
                           v
                 +--------------------+
                 | Reconstruction svc |
                 | services/recon     |
                 +---------+----------+
                           |
                           | reconstruction mode
                           | DICOM / plots / metadata
                           v
                 +--------------------+
                 | COMPLETE folder    |
                 +---------+----------+
                           v
                 +--------------------+
                 | UI result viewers  |
                 +--------------------+
```

The folder location is part of the scan state machine. A scan is not represented only by an in-memory queue object.

---

# 3. Core persistent data model

## 3.1 `ScanTask`

Primary definition:

- `common/types.py`

`ScanTask` is the persistent contract carried through the scan pipeline.

Important fields:

```text
ScanTask
├── id
├── sequence
├── protocol_name
├── scan_number
├── system
├── patient
├── exam
├── parameters
├── adjustment
├── processing
├── other
├── results
└── journal
```

Important architectural distinction:

### `parameters`

Sequence-visible user parameters.

Examples for GRE:

```text
TE
TR
NSA
orientation
FOV
baseresolution
slices
BW
trajectory
ordering
FA
readout_direction
```

### `processing`

Reconstruction dispatch/configuration.

Examples:

```text
trajectory
recon_mode
dim
dim_size
oversampling_read
```

### `other`

Prototype and cross-layer metadata.

This is currently where Localizer planning geometry and resolved acquisition geometry are carried.

### `results`

List of `ResultItem` objects.

A result contains:

```text
type
name
description
file_path
autoload_viewer
primary
```

The UI autoload mechanism depends on this metadata.

---

# 4. Task and queue lifecycle

Primary modules:

- `common/task.py`
- `common/queue.py`
- `services/ui/ui_runtime.py`
- `services/acq/main.py`
- `services/recon/main.py`

## 4.1 Task creation

The UI calls `common.task.create_task()`.

Each scan receives a folder containing at least:

```text
scan.json
seq/
rawdata/
dicom/
temp/
other/
```

State marker files such as `PREPARED`, `EDITING`, and `STOP` are used together with folder location.

## 4.2 State is largely folder-based

Typical lifecycle:

```text
DATA_QUEUE_ACQ
    ↓
DATA_ACQ
    ↓
DATA_QUEUE_RECON
    ↓
DATA_RECON
    ↓
DATA_COMPLETE
```

Failure can move the scan to:

```text
DATA_FAILURE
```

Do not assume that changing an in-memory queue entry changes the real scan state.

---

# 5. Sequence architecture

Primary module:

- `sequences/__init__.py`

Sequences subclass:

```text
SequenceBase
└── PulseqSequence
```

Sequence classes are automatically registered from Python files in `sequences/`.

Important sequence lifecycle:

```text
set_parameters()
      ↓
calculate_sequence()
      ↓
run_sequence()
```

Typical UI-related lifecycle also includes:

```text
get_default_parameters()
setup_ui()
write_parameters_to_ui()
read_parameters_from_ui()
```

## Architectural rule

The sequence wrapper, such as `sequences/gre_3D.py`, should own:

- sequence parameter contract;
- GUI parameter translation;
- planning-resolution entry point;
- processing metadata;
- working-folder integration;
- call into the lower-level Pulseq generator.

The low-level Pulseq generator should not become the owner of UI semantics.

---

# 6. Acquisition service

Primary module:

- `services/acq/main.py`

The acquisition service:

1. finds a prepared scan;
2. moves it into the active acquisition folder;
3. reads `ScanTask`;
4. instantiates the registered sequence;
5. calls `set_parameters()`;
6. calls `calculate_sequence()`;
7. calls `run_sequence()`;
8. writes updated task metadata;
9. moves the task to the reconstruction queue.

Relevant call chain:

```text
services/acq/main.py
    ↓
SequenceBase.get_sequence(...)
    ↓
sequence instance
    ↓
set_parameters()
    ↓
calculate_sequence()
    ↓
run_sequence()
```

Acquisition also snapshots scanner shim values into:

```text
scan_task.adjustment.shim
```

Do not mix acquisition-service responsibilities with reconstruction UI responsibilities.

---

# 7. Reconstruction service

Primary modules:

- `services/recon/main.py`
- `services/recon/reconstruction.py`

The reconstruction service:

1. moves the scan from reconstruction queue to active reconstruction;
2. reads `ScanTask`;
3. calls `reconstruction.run_reconstruction(...)`;
4. optionally performs B0 shim post-processing;
5. writes the updated task;
6. moves the task to `DATA_COMPLETE`.

`services/recon/reconstruction.py` dispatches based on:

```text
processing.recon_mode
processing.trajectory
```

Examples include:

```text
basic3d
b0_map
localizer2d
cartesian
```

---

# 8. Localizer planning state

Primary modules:

- `services/ui/spatialbox.py`
- `services/ui/viewerwidget.py`
- `services/ui/examination.py`

## 8.1 `PlanningState`

`PlanningState` contains two shared `Box3D` objects:

```text
PlanningState
├── fov_box
└── shim_box
```

All three Localizer viewers reference the same `PlanningState`.

Current `Box3D` representation:

```text
center_x/y/z : normalized 0..1
size_x/y/z   : normalized 0..1
rotation_x/y/z : degrees
```

The planning boxes are currently normalized relative to the Localizer reference volume rather than stored directly in millimeters.

## 8.2 Localizer viewers

Localizer planning uses scanner-base planes:

```text
Axial    -> scanner X-Y
Coronal  -> scanner X-Z
Sagittal -> scanner Y-Z
```

The UI projects the same 3D FOV/Shim geometry into these three views.

Important distinction:

> The Localizer viewers are scanner-base planning views. They are not the later FOV-native reconstructed coordinate system.

## 8.3 Editable planning overlays

`services/ui/viewerwidget.py` uses PyQtGraph `RectROI` for Localizer planning.

These are intentionally interactive:

```text
move
resize
rotate
```

Cross-view synchronization updates the shared 3D `PlanningState`.

Do not reuse these editable ROIs as the architecture for completed-scan overlays.

Post-reconstruction overlays should be read-only graphics.

---

# 9. Copying Localizer planning to a target scan

Primary module:

- `services/ui/examination.py`

Current flow:

```text
Localizer PlanningState
        ↓ Apply Planning
Localizer scan_task.other["planning"]
        ↓ Copy FOV
target scan_task.other["geometry"]
```

Current copied target geometry includes approximately:

```text
geometry
├── fov_box
├── reference_fov_mm
├── coordinate_system = "scanner_xyz_v1"
└── reference_origin = "scanner_isocenter"
```

At the baseline commit used for this document, Copy FOV does **not yet** persist a Shim-box snapshot into the target scan's `geometry`.

If post-scan reconstruction must show "the Shim box used for this scan", the preferred architecture is:

```text
target scan geometry
├── fov_box
├── shim_box         # acquisition-time planning snapshot
├── reference_fov_mm
├── coordinate_system
└── reference_origin
```

This avoids reading a later-mutated Localizer planning state after acquisition.

---

# 10. Geometry architecture

Primary modules:

- `common/geometry.py`
- `sequences/common/planning.py`

There are three different coordinate concepts. Keep them separate.

## 10.1 Scanner physical coordinates

```text
scanner X -> physical Gx
scanner Y -> physical Gy
scanner Z -> physical Gz
```

Call this basis:

```text
X, Y, Z
```

## 10.2 FOV-local coordinates

A Localizer-prescribed box defines a rotated local basis:

```text
X', Y', Z'
```

The FOV rotation is represented by:

```text
rotation_local_to_scanner
```

Meaning:

```text
scanner_vector = rotation_local_to_scanner @ local_vector
```

Euler convention:

```text
R = Rz @ Ry @ Rx
```

There is an intentional Y-axis sign convention matching the current viewer behavior.

Do not change this convention casually.

## 10.3 Logical encoding coordinates

The sequence uses:

```text
Read
Phase
Third
```

For 3D Cartesian imaging, Third is the second phase-encoding axis.

`Orientation` and `Readout Direction` decide how the logical encoding axes are arranged inside the FOV-local basis.

Then:

```text
logical_to_scanner
    =
rotation_local_to_scanner
    @
encoding_matrix
```

Meaning:

```text
scanner_vector = logical_to_scanner @ logical_vector
```

The columns are physically meaningful:

```text
logical_to_scanner[:, 0] = Read axis in scanner XYZ
logical_to_scanner[:, 1] = Phase axis in scanner XYZ
logical_to_scanner[:, 2] = Third axis in scanner XYZ
```

Inverse transformation is possible by transpose because the transform is orthogonal:

```text
logical_vector = logical_to_scanner.T @ scanner_vector
```

---

# 11. Orientation and Readout Direction

Current base orientation channel conventions:

```text
Axial    -> x, y, z
Coronal  -> x, z, y
Sagittal -> y, z, x
```

`readout_direction` currently supports:

```text
Horizontal
Vertical
```

For `Vertical`, the first two logical channels are swapped.

Architectural meaning:

```text
FOV rotation
    +
Orientation
    +
Readout Direction
    ↓
Read / Phase / Third basis
    ↓
logical_to_scanner
```

`PE Ordering` is a separate concept.

It changes traversal order through phase-encoding indices; it does not redefine the spatial encoding basis.

Keep these concepts separate in GUI labels and code.

---

# 12. Planning resolution

Primary module:

- `sequences/common/planning.py`

`resolve_task_planning(...)` reads:

```text
scan_task.other["geometry"]
```

and resolves:

```text
ScanGeometry
EncodingGeometry
```

## `ScanGeometry`

Contains:

```text
center_scanner_m
fov_local_m
rotation_local_to_scanner
```

## `EncodingGeometry`

Contains:

```text
center_logical_m
fov_logical_m
logical_to_scanner
```

The sequence wrapper should store the resolved geometry in the task for reproducibility/debugging.

Current GRE uses:

```text
scan_task.other["resolved_geometry"]
scan_task.other["resolved_encoding"]
scan_task.other["geometry_application"]
```

These are important cross-layer metadata. Do not rename them without explicit approval.

---

# 13. GRE planning and Pulseq generation

Primary modules:

- `sequences/gre_3D.py`
- `sequences/common/make_gre_3D.py`
- `sequences/common/gradient_transform.py`

High-level GRE flow:

```text
GRE GUI parameters
        ↓
SequenceGRE_3D.calculate_sequence()
        ↓
resolve_task_planning()
        ↓
resolved ScanGeometry / EncodingGeometry
        ↓
store metadata in ScanTask.other
        ↓
generate_pulseq(planned_encoding)
        ↓
make_gre_3D.pypulseq_gre3D(...)
        ↓
logical gradients
        ↓
gradient transform
        ↓
physical scanner Gx/Gy/Gz
```

---

# 14. Logical-to-physical gradient transform

Primary module:

- `sequences/common/gradient_transform.py`

Core relation:

```text
G_scanner = M @ G_logical
```

where:

```text
M = logical_to_scanner
```

The implementation:

- scales each logical gradient contribution;
- assigns contributions to scanner X/Y/Z;
- combines simultaneous contributions on each scanner axis;
- adds the resulting physical gradients to the Pulseq block.

## Gradient safety

Oblique encoding can combine multiple logical gradients onto one physical scanner axis.

A conservative safety scale is computed from the maximum absolute row sum of `logical_to_scanner`.

Do not remove this safety logic simply because axis-aligned tests pass.

---

# 15. Local PyPulseq dependency

Primary directory:

- `pypulseq/`

Treat this as a vendor-like subsystem.

A known local fix exists in:

- `pypulseq/add_gradients.py`

Triangle gradients with `flat_time == 0` must still include their falling ramp when rasterized and combined.

This is important for oblique transforms because multiple logical gradients may be combined on a physical scanner axis.

Architectural rule:

> Any change under `pypulseq/` requires a focused regression test demonstrating why the vendor-like layer must change.

Do not make broad formatting/refactoring edits in this directory as part of unrelated tasks.

---

# 16. 3D Cartesian reconstruction

Primary module:

- `recon/recon_utils/cartesian3d.py`

The reconstructed k-space array is organized as:

```text
axis 0 = Read
axis 1 = Phase
axis 2 = Third
```

More precisely, the working k-space is created as:

```text
(num_readout, num_phase, num_slices)
```

The reconstruction:

1. validates raw sample count;
2. reshapes raw ADC stream;
3. inserts lines using PE ordering;
4. corrects recorded ADC phase;
5. optionally applies center translation in k-space;
6. performs 3D FFT;
7. removes readout oversampling.

## Translation

Planned FOV center translation is applied as a Cartesian k-space phase ramp.

This preserves the native logical reconstruction basis.

Do not replace it with an arbitrary image-space translation without an explicit design decision.

---

# 17. Reconstruction coordinate system

This is a critical architectural invariant.

The reconstructed 3D volume is **FOV-native**, not automatically scanner-base XYZ.

The volume axes are:

```text
Read
Phase
Third
```

Therefore, if an oblique FOV was prescribed, the FFT result is already aligned to that prescribed logical acquisition coordinate system.

Do not describe the current implementation as:

```text
reconstruct scanner XYZ
then rotate image into FOV coordinates
```

The intended model is:

```text
prescribed FOV
    ↓
logical Read/Phase/Third encoding
    ↓
physical gradients generated through logical_to_scanner
    ↓
logical k-space
    ↓
3D FFT
    ↓
FOV-native volume
```

---

# 18. FOV-native three-view reconstruction

Primary module:

- `services/recon/reconstruction.py`

Current basic3d reconstruction creates three orthogonal FOV-native views:

```text
Read / Phase
Read / Third
Phase / Third
```

These are generated using permutations of the same FOV-native volume.

Current conceptual mapping:

```text
View 1: Read / Phase
normal: Third

View 2: Read / Third
normal: Phase

View 3: Phase / Third
normal: Read
```

These should not be called scanner Axial/Coronal/Sagittal unless the logical basis happens to be axis-aligned with scanner XYZ.

---

# 19. DICOM writer and current limitation

Primary module:

- `recon/DICOM/DICOM_utils.py`

The project has begun using resolved FOV geometry to set spatial spacing for the FOV-native result series.

However, patient-space spatial orientation is not yet fully trustworthy.

In particular, do **not** assume that:

```text
scanner XYZ == DICOM Patient LPS
```

A formal transform involving patient position and scanner/patient orientation is still required before writing authoritative:

```text
ImageOrientationPatient
ImagePositionPatient
```

Architecture rule:

> Do not claim patient-space DICOM orientation correctness until the scanner-to-patient transform is explicitly defined and tested.

---

# 20. Viewer architecture

Primary modules:

- `services/ui/examination.py`
- `services/ui/viewerwidget.py`

## Localizer viewer mode

Editable planning context:

```text
scanner-base view
+
shared PlanningState
+
interactive FOV RectROI
+
interactive Shim RectROI
```

## Completed GRE viewer mode

Preferred architecture:

```text
FOV-native reconstructed image
+
read-only acquisition overlay
```

Recommended post-scan overlay content:

```text
FOV boundary
Shim-box projection/intersection
Read/Phase/Third basis indicator
```

Do not make completed-scan overlays editable by default.

---

# 21. Recommended indicator semantics

## On Localizer

The planning box itself defines FOV-local axes:

```text
X'
Y'
Z'
```

This is the correct indicator before a specific sequence encoding choice is applied.

Why not always label Localizer indicator as Read/Phase/Third?

Because logical encoding also depends on:

```text
Orientation
Readout Direction
```

## On reconstructed GRE

Use:

```text
Read
Phase
Third
```

because these are the actual reconstructed logical axes.

This preserves the conceptual separation:

```text
Localizer:
scanner XYZ
    ↓ FOV rotation
FOV local X'Y'Z'

Target sequence:
FOV local X'Y'Z'
    ↓ Orientation + Readout Direction
Read/Phase/Third
```

---

# 22. Shim overlay architecture

For a completed scan, the Shim overlay should represent the geometry snapshot used for that scan, not a later version of the Localizer state.

Recommended persistent snapshot:

```text
scan_task.other["geometry"]["shim_box"]
```

The Shim box begins in Localizer/scanner planning coordinates.

To render it on an FOV-native result:

1. convert normalized Shim `Box3D` into scanner-space center, dimensions, and rotation;
2. build its 3D corners or its plane intersection;
3. subtract the acquisition FOV center;
4. transform scanner-relative points into logical coordinates:

```text
r_logical = logical_to_scanner.T @ r_scanner_relative
```

5. project into the current native view.

For an initial implementation, projecting the 3D box edges is acceptable.

For a more geometrically faithful implementation, intersect the 3D Shim box with the currently displayed slice plane so the outline appears/disappears as the user scrolls.

---

# 23. B0 shim architecture

Relevant modules include:

- `sequences/adj_b0_map.py`
- `services/recon/reconstruction.py`
- `services/shim/`
- `recon/B0Shim/`

Current B0 flow is conceptually:

```text
Localizer Shim box
       ↓
B0 map acquisition
       ↓
B0 map reconstruction
       ↓
ROI / physical coordinate model
       ↓
first-order B0 fit
       ↓
shim conversion/application
```

Keep this path separate from FOV-native display work unless the task explicitly concerns B0.

Do not make a Localizer display feature alter shim fitting or scanner shim application.

---

# 24. Areas of architectural risk

A coding agent should treat the following as high-risk:

## Coordinate convention changes

Files:

```text
common/geometry.py
services/ui/viewerwidget.py
sequences/common/planning.py
```

Any sign swap, axis swap, Euler order change, transpose change, or row/column reinterpretation may create a visually plausible but physically incorrect result.

## Gradient transform changes

Files:

```text
sequences/common/gradient_transform.py
sequences/common/make_gre_3D.py
pypulseq/
```

Must preserve physical max gradient and slew limits.

## Reconstruction axis permutations

Files:

```text
recon/recon_utils/cartesian3d.py
services/recon/reconstruction.py
recon/DICOM/DICOM_utils.py
```

Always document whether array axes mean:

```text
row
column
slice
```

versus:

```text
Read
Phase
Third
```

## DICOM orientation

Do not guess scanner-to-patient geometry.

## TSE3D

Do not automatically port GRE planning geometry into TSE3D. Existing orientation conventions may differ.

---

# 25. Current architecture invariants

Unless explicitly changed by the human architect, preserve these:

1. Localizer planning uses normalized scanner XYZ.
2. FOV rotation maps local FOV axes to scanner axes.
3. Euler rotation convention is `Rz @ Ry @ Rx`.
4. Logical encoding axes are Read / Phase / Third.
5. `logical_to_scanner = rotation_local_to_scanner @ encoding_matrix`.
6. `Orientation` and `Readout Direction` choose logical encoding inside the FOV-local frame.
7. PE Ordering changes sampling order, not the encoding basis.
8. Oblique gradients are physically transformed before being added to the sequence.
9. Oblique multi-axis gradient safety limits remain enforced.
10. Planned center translation is handled through Cartesian k-space translation.
11. 3D GRE reconstruction remains FOV-native by default.
12. Native result views are Read/Phase, Read/Third, and Phase/Third.
13. Localizer overlays are editable; completed-scan overlays should be read-only.
14. Patient LPS DICOM orientation is not yet a solved architectural layer.
15. B0 shim behavior must not be changed by unrelated Localizer/FOV work.
16. Changes to local `pypulseq/` require focused regression tests.
17. Do not broaden GRE planning changes into TSE3D without an explicit design discussion.

---

# 26. Recommended tests before merging geometry-related changes

At minimum, run the relevant focused tests:

```bash
python -m pytest tests/test_geometry.py -q
python -m pytest tests/test_cartesian_translation.py -q
python -m pytest tests/test_gre_gradient_transform.py -q
python -m pytest tests/test_gre_slab_selection.py -q
```

If B0/shim code is touched:

```bash
python -m pytest tests/test_b0_shim.py -q
```

Also run any new tests added for the feature.

For GUI-only geometry changes, unit tests are still preferred for the coordinate calculations even if the PyQt rendering itself is verified manually.

---

# 27. Agent change protocol

Before editing:

```text
1. git status
2. git rev-parse HEAD
3. read AGENTS.md
4. read this architecture document
5. inspect only the relevant call path
6. state proposed files to modify
7. state invariants that will remain unchanged
8. state test plan
```

During editing:

```text
- minimize touched files;
- avoid unrelated formatting;
- do not rename architecture-level keys;
- add pure geometry helpers before embedding math directly in GUI callbacks;
- keep coordinate transforms explicit;
- comment matrix direction, e.g. "logical -> scanner";
- prefer tests for transform direction/sign conventions.
```

After editing:

```text
1. run focused tests;
2. show git diff --stat;
3. summarize each changed file;
4. report any test not run;
5. report any ambiguity discovered;
6. do not hide failed tests or silently weaken assertions.
```

---

# 28. Useful file map

```text
common/types.py
    Persistent Pydantic task/result models.

common/task.py
    Scan task creation, read/write, marker state files.

common/queue.py
    Folder-backed processing queue and transitions.

common/geometry.py
    Scanner/FOV/logical geometry conventions.

services/ui/ui_runtime.py
    Exam/queue UI runtime state.

services/ui/examination.py
    High-level examination UI, planning save/copy, result autoload.

services/ui/spatialbox.py
    Box3D and shared PlanningState.

services/ui/viewerwidget.py
    DICOM viewer, planning ROI projection/interactions.

sequences/__init__.py
    Sequence registry and SequenceBase API.

sequences/common/planning.py
    Convert copied Localizer geometry into resolved geometry.

sequences/gre_3D.py
    GRE parameters, GUI bridge, planning integration, Pulseq call.

sequences/common/make_gre_3D.py
    GRE Pulseq generation.

sequences/common/gradient_transform.py
    Logical gradient -> physical scanner gradient transform.

pypulseq/
    Local/vendor-like Pulseq implementation.

services/acq/main.py
    Acquisition worker.

services/recon/main.py
    Reconstruction worker.

services/recon/reconstruction.py
    Reconstruction dispatch and result generation.

recon/recon_utils/cartesian3d.py
    3D Cartesian GRE reconstruction.

recon/recon_utils/cartesian_translation.py
    Planned center translation in k-space.

recon/DICOM/DICOM_utils.py
    DICOM writing and ResultItem creation.

services/shim/
recon/B0Shim/
    B0 fitting and shim application path.

tests/
    Geometry, gradient transform, slab selection, B0 and related regression tests.
```

---

# 29. Mental model for future work

When evaluating any Localizer/FOV feature, ask which layer it belongs to:

```text
Planning state?
    -> spatialbox / viewerwidget / examination

Physical geometry?
    -> common/geometry

Task-to-sequence geometry resolution?
    -> sequences/common/planning

Pulse sequence encoding?
    -> gre_3D / make_gre_3D / gradient_transform

Reconstruction coordinates?
    -> cartesian3d / reconstruction

Display-only overlay?
    -> viewerwidget / examination

DICOM interoperability?
    -> DICOM_utils + future scanner-to-patient transform

B0/shim fitting?
    -> B0Shim / services/shim
```

If a single small feature appears to require changes in many unrelated layers, pause and verify the design before implementing.

---

# 30. Current design direction for the next UI work

The current preferred design is:

```text
Localizer:
    scanner-base Axial / Coronal / Sagittal
    editable FOV + Shim
    FOV-local X'/Y'/Z' indicator

Target 3D GRE:
    Orientation + Readout Direction
    determine logical Read/Phase/Third

Acquisition:
    logical gradients
    -> logical_to_scanner
    -> physical scanner gradients

Reconstruction:
    FOV-native Read/Phase/Third volume

Completed-scan viewers:
    Read/Phase
    Read/Third
    Phase/Third
    +
    read-only FOV boundary
    +
    read-only Shim geometry
    +
    Read/Phase/Third indicator
```

This separation is intentional and should be preserved unless an architecture review decides otherwise.
