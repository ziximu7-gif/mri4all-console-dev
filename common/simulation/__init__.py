"""
Phase A geometry-aware hardware simulation.

Pure numerical boundary:
- phantom.py : the single fixed scanner-XYZ digital phantom
- context.py : typed plain-number simulation contexts (no ScanTask)
- kspace.py  : forward Cartesian k-space / raw-ADC synthesis

Adapter layers (sequences/gre_3D.py, sequences/localizer.py) convert
ScanTask / resolved-geometry data into the typed contexts; this package
never touches ScanTask persistence.

PHASE A LIMITATION
------------------
The simulator samples the phantom using the RESOLVED acquisition geometry
and the reconstruction uses the same resolved geometry. This validates the
planning -> raw -> reconstruction -> display geometry chain but does NOT
independently validate physical gradient-transform correctness. A Phase B
simulator may derive k(t) from physical Gx/Gy/Gz waveforms and ADC timing.
No aliasing model is included in Phase A; the simulated acquisition
represents an ideal selected FOV.
"""

from common.simulation.context import (
    GRE3DSimulationContext,
    Localizer2DSimulationContext,
)
from common.simulation.kspace import (
    synthesize_gre3d_raw,
    synthesize_localizer_kspace,
)
from common.simulation.phantom import (
    PHANTOM_MARKERS,
    phantom_bounds_m,
    sample_phantom,
)

__all__ = [
    "GRE3DSimulationContext",
    "Localizer2DSimulationContext",
    "synthesize_gre3d_raw",
    "synthesize_localizer_kspace",
    "PHANTOM_MARKERS",
    "phantom_bounds_m",
    "sample_phantom",
]
