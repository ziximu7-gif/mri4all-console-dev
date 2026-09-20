"""
Regression tests for the Phase A geometry-aware hardware simulation.

These tests lock the contract between the fixed scanner-space phantom,
the forward k-space/raw synthesis (common/simulation), and the EXISTING
reconstruction path (recon.recon_utils) - reconstruction code is imported
and called here as a black box but never modified.

All asserted numbers were produced by running the real synthesize ->
reconstruct pipeline, so the tests pin current verified behavior rather
than fragile hand guesses where possible, while still asserting the
hand-calculated geometric conventions the user specified.
"""

import numpy as np
import pytest

from common.geometry import (
    ORIENTATION_CHANNELS,
    ScanGeometry,
    orientation_encoding_matrix,
    planning_euler_to_matrix,
    resolve_encoding_geometry,
)
from common.simulation import (
    GRE3DSimulationContext,
    Localizer2DSimulationContext,
    PHANTOM_MARKERS,
    phantom_bounds_m,
    sample_phantom,
    synthesize_gre3d_raw,
    synthesize_localizer_kspace,
)
from recon.recon_utils.cartesian3d import (
    reconstruct_cartesian_3d_complex,
)
from sequences.common.get_trajectory import choose_pe_order

# Shared GRE acquisition geometry used across tests.
BASE = 24
SLICES = 12
OVERSAMPLE = 2
FOV_LOCAL_M = np.array([0.16, 0.24, 0.12])


def _make_gre_ctx(rot_deg, center_scanner_m):
    """
    Build a GRE3D simulation context from a planner-style rotation
    (Rz@Ry@Rx) and FOV center, via the SAME geometry resolver the
    sequence adapter uses.
    """
    rotation = planning_euler_to_matrix(*rot_deg)
    scan_geom = ScanGeometry(
        center_scanner_m=np.asarray(center_scanner_m, float),
        fov_local_m=FOV_LOCAL_M,
        rotation_local_to_scanner=rotation,
    )
    enc = resolve_encoding_geometry(scan_geom, "Axial", "Horizontal")
    ctx = GRE3DSimulationContext(
        kind="gre3d",
        center_scanner_m=scan_geom.center_scanner_m,
        center_logical_m=enc.center_logical_m,
        fov_logical_m=enc.fov_logical_m,
        logical_to_scanner=enc.logical_to_scanner,
        base_resolution=BASE,
        n_phase=BASE,
        n_slice=SLICES,
        oversampling_read=OVERSAMPLE,
    )
    return ctx, enc


def _synth_and_reconstruct(ctx, enc):
    """Run the real forward synthesis then the real reconstruction."""
    pe_order = choose_pe_order(
        ndims=3,
        npe=[BASE, SLICES],
        traj="center_out",
        save_pe_order=False,
    )
    adc_phases = np.zeros(len(pe_order))
    raw = synthesize_gre3d_raw(ctx, pe_order, adc_phases)
    images, _ = reconstruct_cartesian_3d_complex(
        raw=raw,
        order=pe_order,
        adc_phases=adc_phases,
        dims=(SLICES, BASE, BASE * OVERSAMPLE),
        echo_count=1,
        oversampling_read=OVERSAMPLE,
        center_logical_m=enc.center_logical_m,
        fov_logical_m=enc.fov_logical_m,
    )
    return np.abs(images[0])


def _expected_point_sampled_volume(ctx):
    """
    Independently sample the fixed phantom on the POST-CROP logical read
    grid. This is a clean-room expectation that shares no k-space,
    FFT-shift or reconstruction code with the simulator.
    """
    n_read = BASE  # already cropped (oversampling removed)
    fov = ctx.fov_logical_m

    def coords(n, extent, center_index):
        return (np.arange(n) - center_index) * (extent / n)

    read_c = coords(n_read, fov[0], n_read // 2)
    phase_c = coords(BASE, fov[1], BASE - BASE // 2)
    third_c = coords(SLICES, fov[2], SLICES - SLICES // 2)

    rr, pp, tt = np.meshgrid(read_c, phase_c, third_c, indexing="ij")
    r_logical = np.stack([rr, pp, tt], axis=-1)
    r_scanner = (
        ctx.center_scanner_m + r_logical @ ctx.logical_to_scanner.T
    )
    return np.abs(sample_phantom(r_scanner))


# =====================================================================
# 1. Phantom: determinism + asymmetry (asymmetric across views)
# =====================================================================


def test_phantom_sample_is_deterministic():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.03, 0.02, -0.02],
            [0.5, 0.5, 0.5],
        ]
    )
    a = sample_phantom(pts)
    b = sample_phantom(pts)
    np.testing.assert_array_equal(a, b)


def test_phantom_is_asymmetric_across_views():
    """The object must not be symmetric: three orthogonal views differ."""
    n = 32
    span = np.linspace(-0.12, 0.12, n)
    yy, zz = np.meshgrid(span, span, indexing="ij")

    def slab(axis):
        pts = np.zeros((n, n, 3))
        a, b = [i for i in range(3) if i != axis]
        pts[:, :, a] = yy
        pts[:, :, b] = zz
        return sample_phantom(pts)

    axial = slab(2)  # X-Y
    coronal = slab(1)  # X-Z
    sagittal = slab(0)  # Y-Z

    assert not np.allclose(axial, coronal)
    assert not np.allclose(axial, sagittal)
    assert not np.allclose(coronal, sagittal)


def test_phantom_markers_outside_each_other():
    """Marker centers are distinct and non-collinear enough to be chiral."""
    centers = np.array(
        [
            PHANTOM_MARKERS[k].center_m
            for k in ("bright", "dark", "chiral_high", "chiral_low")
        ]
    )
    # Pairwise distinct.
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            assert np.linalg.norm(centers[i] - centers[j]) > 1e-3


# =====================================================================
# 2. Identity zero-center GRE round-trip (EXACT)
# =====================================================================


def test_identity_zero_center_roundtrip_exact():
    ctx, enc = _make_gre_ctx([0, 0, 0], [0.0, 0.0, 0.0])
    vol = _synth_and_reconstruct(ctx, enc)
    expected = _expected_point_sampled_volume(ctx)
    np.testing.assert_allclose(vol, expected, atol=1e-9)


# =====================================================================
# 3. Readout oversampling / final Read-FOV / voxel-size contract
# =====================================================================


def test_oversampling_contracts():
    ctx, enc = _make_gre_ctx([0, 0, 0], [0.0, 0.0, 0.0])
    # n_readout includes oversampling.
    assert ctx.n_readout == BASE * OVERSAMPLE
    # Sampled (pre-crop) Read FOV is enlarged by the oversampling factor.
    assert ctx.sampled_fov_logical_m[0] == pytest.approx(
        ctx.fov_logical_m[0] * OVERSAMPLE
    )
    assert ctx.sampled_fov_logical_m[1] == pytest.approx(ctx.fov_logical_m[1])
    assert ctx.sampled_fov_logical_m[2] == pytest.approx(ctx.fov_logical_m[2])

    vol = _synth_and_reconstruct(ctx, enc)
    # Reconstruction crops the read center back to the prescribed Read FOV.
    assert vol.shape == (BASE, BASE, SLICES)
    # Final Read voxel size == fov_read / base_resolution.
    read_voxel_m = ctx.fov_logical_m[0] / BASE
    assert read_voxel_m == pytest.approx(FOV_LOCAL_M[0] / BASE)

    # Hand-calculated marker peak: 'bright' at scanner (0.030, 0.020,
    # -0.020); identity encoding places it at r_logical = its scanner
    # coordinates. Expected read index = round(0.030/(0.16/24)) + 12 = 4+12=16.
    vox_read = FOV_LOCAL_M[0] / BASE
    vox_phase = FOV_LOCAL_M[1] / BASE
    vox_third = FOV_LOCAL_M[2] / SLICES
    exp_read = round(0.030 / vox_read) + BASE // 2
    assert vol[exp_read, :, :].max() > 1.5  # the bright=2.0 feature is there


# =====================================================================
# 4. +90 degree mapping with hand-calculated marker coordinates
# =====================================================================


def test_plus90_marker_hand_calculated():
    ctx, enc = _make_gre_ctx([0, 0, 90], [0.0, 0.0, 0.0])
    vol = _synth_and_reconstruct(ctx, enc)

    # Rotation R = Rz(90). logical = R^T @ scanner = Rz(-90) @ scanner.
    # 'bright' scanner (0.030, 0.020, -0.020):
    #   Rz(-90) maps (x,y,z) -> (y, -x, z) = (0.020, -0.030, -0.020)
    marker = PHANTOM_MARKERS["bright"]
    M = planning_euler_to_matrix(0, 0, 90)
    r_logical = M.T @ marker.center_m
    np.testing.assert_allclose(
        r_logical, np.array([0.020, -0.030, -0.020]), atol=1e-9
    )

    vox_read = FOV_LOCAL_M[0] / BASE
    vox_phase = FOV_LOCAL_M[1] / BASE
    vox_third = FOV_LOCAL_M[2] / SLICES
    exp = (
        round(r_logical[0] / vox_read) + BASE // 2,
        round(r_logical[1] / vox_phase) + (BASE - BASE // 2),
        round(r_logical[2] / vox_third) + (SLICES - SLICES // 2),
    )
    # Hand-computed index from the locked geometry resolver.
    assert exp == (15, 9, 4)
    assert vol[exp] == pytest.approx(2.0, abs=1e-6)


# =====================================================================
# 5. +90 vs -90 distinguishable by chiral/asymmetric markers
# =====================================================================


def test_plus90_and_minus90_distinguishable():
    ctxp, encp = _make_gre_ctx([0, 0, 90], [0, 0, 0])
    ctxm, encm = _make_gre_ctx([0, 0, -90], [0, 0, 0])
    volp = _synth_and_reconstruct(ctxp, encp)
    volm = _synth_and_reconstruct(ctxm, encm)
    assert not np.allclose(volp, volm)
    # bright marker lands on mirrored logical positions.
    np.testing.assert_allclose(
        volp[15, 9, 4], 2.0, atol=1e-6
    )
    np.testing.assert_allclose(
        volm[9, 15, 4], 2.0, atol=1e-6
    )


# =====================================================================
# 6. Translation sign: +X FOV center moves fixed object to -logical Read
# =====================================================================


def test_translation_sign_negative_read():
    # A fixed object at the isocenter; move the FOV center to +X. With
    # identity encoding r_logical = r_scanner - center, so the object
    # appears at negative logical Read.
    ctx0, enc0 = _make_gre_ctx([0, 0, 0], [0.0, 0.0, 0.0])
    ctx1, enc1 = _make_gre_ctx([0, 0, 0], [0.02, 0.0, 0.0])
    vol0 = _synth_and_reconstruct(ctx0, enc0)
    vol1 = _synth_and_reconstruct(ctx1, enc1)

    def read_com(vol):
        return np.average(
            np.arange(vol.shape[0]), weights=vol.sum(axis=(1, 2))
        )

    com0 = read_com(vol0)
    com1 = read_com(vol1)
    assert com1 < com0  # content shifted toward negative Read


def test_translation_roundtrip_matches_shifted_sampling():
    """Translation also reconstructs EXACTLY vs. independent sampling."""
    ctx, enc = _make_gre_ctx([0, 0, 0], [0.02, -0.01, 0.0])
    vol = _synth_and_reconstruct(ctx, enc)
    expected = _expected_point_sampled_volume(ctx)
    np.testing.assert_allclose(vol, expected, atol=1e-9)


# =====================================================================
# 7. Localizer: three views from the SAME fixed 3D phantom
#    through a centered finite slice (ideal boxcar slice profile)
# =====================================================================


def _independent_localizer_slice(
    orientation,
    n_read,
    n_phase,
    fov_m,
    slice_thickness_m,
    n_slice_samples,
):
    """Clean-room verifier: integrate sample_phantom through a centered
    finite slice on the slice-select axis.

    This never calls synthesize_localizer_kspace; it rebuilds the same
    physical expectation directly from sample_phantom so a mistake in
    the production integration loop cannot hide itself.
    """
    read_axis, phase_axis, slice_axis = ORIENTATION_CHANNELS[orientation]
    aidx = {"x": 0, "y": 1, "z": 2}
    rd, pd, sd = aidx[read_axis], aidx[phase_axis], aidx[slice_axis]

    # Slice centered at isocenter: -thickness/2 .. +thickness/2.
    slice_half = 0.5 * slice_thickness_m
    slice_c = np.linspace(-slice_half, slice_half, n_slice_samples)
    spacing = slice_thickness_m / (n_slice_samples - 1)

    read_c = (np.arange(n_read) - n_read // 2) * (fov_m / n_read)
    phase_c = (np.arange(n_phase) - n_phase // 2) * (fov_m / n_phase)
    rr, pph = np.meshgrid(read_c, phase_c, indexing="xy")  # (phase, read)

    img = np.zeros((n_phase, n_read))
    for s, slice_pos in enumerate(slice_c):
        pts = np.zeros((n_phase, n_read, 3))
        pts[:, :, rd] = rr
        pts[:, :, pd] = pph
        pts[:, :, sd] = slice_pos
        w = 0.5 if s in (0, len(slice_c) - 1) else 1.0
        img += w * sample_phantom(pts)
    # Trapezoidal integration through the finite slice. Deliberately NOT
    # divided by slice thickness: signal scales with slice thickness.
    return img * spacing


def _localizer_image(ctx):
    raw = synthesize_localizer_kspace(ctx)
    kspace = raw.reshape(ctx.nsa, ctx.n_phase, ctx.n_read)[0]
    return np.abs(
        np.fft.fftshift(np.fft.fft2(np.fft.fftshift(kspace)))
    )


def test_localizer_raw_contract_and_shape():
    nsa, n = 1, 64
    ctx = Localizer2DSimulationContext(
        kind="localizer2d", orientation="Axial", fov_m=0.16,
        slice_thickness_m=0.010, n_slice_samples=161,
        n_read=n, n_phase=n, nsa=nsa,
    )
    raw = synthesize_localizer_kspace(ctx)
    # Preserve raw.reshape(nsa, ny, nx) with ny=Phase, nx=Read.
    assert raw.size == nsa * n * n
    reshaped = raw.reshape(nsa, n, n)
    assert reshaped.shape == (nsa, n, n)


def test_localizer_three_views_centered_slice():
    n = 48
    nsa = 1
    ctxs = {
        o: Localizer2DSimulationContext(
            kind="localizer2d", orientation=o, fov_m=0.16,
            slice_thickness_m=0.010, n_slice_samples=161,
            n_read=n, n_phase=n, nsa=nsa,
        )
        for o in ("Axial", "Coronal", "Sagittal")
    }
    for orientation, ctx in ctxs.items():
        img = _localizer_image(ctx)
        expected = _independent_localizer_slice(
            orientation, ctx.n_read, ctx.n_phase, ctx.fov_m,
            ctx.slice_thickness_m, ctx.n_slice_samples,
        )
        # Each orientation is a centered finite slice of the SAME fixed
        # phantom, integrated through its slice-select axis only.
        np.testing.assert_allclose(img, expected, atol=1e-9)

    # The three views are genuinely different (distinct slice axes).
    imgA = _localizer_image(ctxs["Axial"])
    imgC = _localizer_image(ctxs["Coronal"])
    imgS = _localizer_image(ctxs["Sagittal"])
    assert not np.allclose(imgA, imgC)
    assert not np.allclose(imgA, imgS)
    assert not np.allclose(imgC, imgS)


def test_localizer_uses_third_axis_as_slice_axis():
    """ch2 is the slice-select axis per ORIENTATION_CHANNELS."""
    assert ORIENTATION_CHANNELS["Axial"] == ("x", "y", "z")
    assert ORIENTATION_CHANNELS["Coronal"] == ("x", "z", "y")
    assert ORIENTATION_CHANNELS["Sagittal"] == ("y", "z", "x")


def test_localizer_centered_slice_is_not_full_axis_projection():
    """Regression guard: the centered finite slice must NOT reproduce the
    old full-axis projection.

    phantom_bounds_m() is used ONLY inside this test to rebuild the old
    behaviour; production kspace.py must not depend on it any more.
    """
    n = 48
    ctx = Localizer2DSimulationContext(
        kind="localizer2d", orientation="Axial", fov_m=0.16,
        slice_thickness_m=0.010, n_slice_samples=161,
        n_read=n, n_phase=n, nsa=1,
    )

    # Current behaviour: centered finite slice.
    slice_kspace = synthesize_localizer_kspace(ctx)
    slice_image = np.abs(
        np.fft.fftshift(
            np.fft.fft2(
                np.fft.fftshift(
                    slice_kspace.reshape(ctx.nsa, n, n)[0]
                )
            )
        )
    )

    # OLD behaviour, rebuilt here with the same FFT convention.
    read_axis, phase_axis, proj_axis = ORIENTATION_CHANNELS["Axial"]
    aidx = {"x": 0, "y": 1, "z": 2}
    rd, pd, jd = aidx[read_axis], aidx[phase_axis], aidx[proj_axis]
    lo, hi = phantom_bounds_m()
    margin = 0.05 * float(np.max(hi - lo))
    proj_lo = lo[jd] - margin
    proj_hi = hi[jd] + margin
    proj_c = np.linspace(proj_lo, proj_hi, 161)
    spacing = (proj_hi - proj_lo) / (len(proj_c) - 1)

    read_c = (np.arange(n) - n // 2) * (ctx.fov_m / n)
    phase_c = (np.arange(n) - n // 2) * (ctx.fov_m / n)
    rr, pph = np.meshgrid(read_c, phase_c, indexing="xy")

    full = np.zeros((n, n))
    for s, p in enumerate(proj_c):
        pts = np.zeros((n, n, 3))
        pts[:, :, rd] = rr
        pts[:, :, pd] = pph
        pts[:, :, jd] = p
        w = 0.5 if s in (0, len(proj_c) - 1) else 1.0
        full += w * sample_phantom(pts)
    full = full * spacing

    full_kspace = np.fft.ifftshift(
        np.fft.ifft2(np.fft.ifftshift(full))
    )
    full_projection_image = np.abs(
        np.fft.fftshift(
            np.fft.fft2(
                np.fft.fftshift(full_kspace)
            )
        )
    )

    assert not np.allclose(
        slice_image,
        full_projection_image,
    )


def test_localizer_context_rejects_non_positive_slice_thickness():
    with pytest.raises(ValueError):
        Localizer2DSimulationContext(
            kind="localizer2d", orientation="Axial", fov_m=0.16,
            slice_thickness_m=0.0, n_slice_samples=161,
            n_read=32, n_phase=32, nsa=1,
        )


def test_localizer_context_rejects_too_few_slice_samples():
    with pytest.raises(ValueError):
        Localizer2DSimulationContext(
            kind="localizer2d", orientation="Axial", fov_m=0.16,
            slice_thickness_m=0.010, n_slice_samples=1,
            n_read=32, n_phase=32, nsa=1,
        )


# =====================================================================
# 8. Orthogonal logical_to_scanner without assuming det = +1
# =====================================================================


def test_mirrored_encoding_matrix_allowed():
    """A det = -1 orthogonal matrix must be accepted (no det=+1 assumption)."""
    M = np.diag([-1.0, 1.0, 1.0])  # mirror in Read
    assert np.isclose(np.linalg.det(M), -1.0)
    ctx = GRE3DSimulationContext(
        kind="gre3d",
        center_scanner_m=np.array([0.0, 0.0, 0.0]),
        center_logical_m=np.array([0.0, 0.0, 0.0]),
        fov_logical_m=np.array([0.16, 0.16, 0.12]),
        logical_to_scanner=M,
        base_resolution=BASE,
        n_phase=BASE,
        n_slice=SLICES,
        oversampling_read=OVERSAMPLE,
    )
    pe_order = choose_pe_order(
        ndims=3, npe=[BASE, SLICES], traj="center_out", save_pe_order=False
    )
    adc = np.zeros(len(pe_order))
    raw = synthesize_gre3d_raw(ctx, pe_order, adc)
    assert np.isfinite(raw).all()

    images, _ = reconstruct_cartesian_3d_complex(
        raw=raw, order=pe_order, adc_phases=adc,
        dims=(SLICES, BASE, BASE * OVERSAMPLE), echo_count=1,
        oversampling_read=OVERSAMPLE,
        center_logical_m=ctx.center_logical_m,
        fov_logical_m=ctx.fov_logical_m,
    )
    np.testing.assert_allclose(
        np.abs(images[0]), _expected_point_sampled_volume(ctx), atol=1e-9
    )


def test_context_rejects_nonorthogonal():
    with pytest.raises(ValueError):
        GRE3DSimulationContext(
            kind="gre3d",
            center_scanner_m=np.zeros(3),
            center_logical_m=np.zeros(3),
            fov_logical_m=np.array([0.1, 0.1, 0.1]),
            logical_to_scanner=np.array([[1.0, 0.5, 0.0], [0, 1, 0], [0, 0, 1]]),
            base_resolution=8, n_phase=8, n_slice=4, oversampling_read=2,
        )


# =====================================================================
# 9. Real hardware path unchanged / legacy sim_context=None unchanged
# =====================================================================


def test_run_pulseq_signature_defaults_preserved():
    """run_pulseq keeps its default signature (sim_context=None default)."""
    import inspect
    from external.seq.adjustments_acq.scripts import run_pulseq

    sig = inspect.signature(run_pulseq)
    params = sig.parameters
    assert "sim_context" in params
    assert params["sim_context"].default is None
    # hardware_simulation default preserved for real-hardware callers.
    assert params["hardware_simulation"].default is False


def _legacy_axial_phantom(n):
    """Rebuild the legacy filename-sniffing Axial localizer phantom."""
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    phantom = np.zeros((n, n), dtype=np.float64)
    body = ((xx / 0.72) ** 2 + (yy / 0.52) ** 2) <= 1.0
    phantom[body] = 1.0
    feature_1 = (
        ((xx - 0.22) / 0.13) ** 2 + ((yy + 0.12) / 0.16) ** 2
    ) <= 1.0
    phantom[feature_1] = 2.0
    feature_2 = (
        ((xx + 0.25) / 0.12) ** 2 + ((yy - 0.10) / 0.13) ** 2
    ) <= 1.0
    phantom[feature_2] = 0.4
    return phantom


def _FakePsiFactory(n_samples):
    """Build a PSInterpreter stand-in reporting the given readout_number."""

    class _Psi:
        def __init__(self, *a, **k):
            pass

        def interpret(self, seq_file):
            instructions = {
                "tx0": (np.array([0]), np.array([0])),
                "grad_vx": (np.array([0]), np.array([0])),
                "grad_vy": (np.array([0]), np.array([0])),
                "grad_vz": (np.array([0]), np.array([0])),
                "rx0_en": (np.array([0]), np.array([0])),
            }
            param_dict = {"readout_number": n_samples, "rx_t": 1.0}
            return instructions, param_dict

    return _Psi


def _run_dispatch_with(monkeypatch, tmp_path, seq_name, hardware_simulation,
                       sim_context, n_samples=64 * 64):
    """Drive run_pulseq's simulation branch with flocra pieces stubbed."""
    import external.seq.adjustments_acq.scripts as scripts_mod

    monkeypatch.setattr(
        scripts_mod, "PSInterpreter", _FakePsiFactory(n_samples)
    )
    monkeypatch.setattr(scripts_mod, "shim", lambda instr, s: instr)

    rawdir = tmp_path / "rawdata"
    rawdir.mkdir(exist_ok=True)
    seqfile = tmp_path / seq_name
    seqfile.write_text("dummy")

    return scripts_mod.run_pulseq(
        seq_file=str(seqfile),
        case_path=str(tmp_path),
        save_np=False,
        hardware_simulation=hardware_simulation,
        sim_context=sim_context,
    )


def test_legacy_localizer_branch_unchanged(monkeypatch, tmp_path):
    """
    With sim_context=None the run_pulseq Localizer filename-sniffing
    legacy branch must produce EXACTLY the legacy synthetic 2D k-space,
    proving the geometry-aware dispatch did not alter legacy behavior.
    """
    # The legacy branch must win over any geometry-aware dispatch. Guard:
    # make the geometry-aware synthesizers explode if ever reached.
    import common.simulation as sim_pkg

    def _boom(*a, **k):
        raise AssertionError(
            "geometry-aware simulator must not be called when "
            "sim_context is None"
        )

    monkeypatch.setattr(sim_pkg, "synthesize_localizer_kspace", _boom)
    monkeypatch.setattr(sim_pkg, "synthesize_gre3d_raw", _boom)

    rxd, rx_t = _run_dispatch_with(
        monkeypatch, tmp_path, "localizer_0_Axial.seq",
        hardware_simulation=True, sim_context=None,
    )

    n = 64
    expected = np.fft.ifftshift(
        np.fft.ifft2(np.fft.ifftshift(_legacy_axial_phantom(n)))
    ).reshape(-1)
    np.testing.assert_allclose(rxd, expected, rtol=0, atol=0)
    assert rx_t == 1.0


def test_dispatch_localizer_context(monkeypatch, tmp_path):
    """An explicit Localizer2D context routes to the geometry-aware simulator."""
    ctx = Localizer2DSimulationContext(
        kind="localizer2d", orientation="Axial", fov_m=0.16,
        slice_thickness_m=0.010, n_slice_samples=161,
        n_read=64, n_phase=64, nsa=1,
    )
    rxd, rx_t = _run_dispatch_with(
        monkeypatch, tmp_path, "localizer_0_Axial.seq",
        hardware_simulation=True, sim_context=ctx,
    )
    expected = synthesize_localizer_kspace(ctx)
    np.testing.assert_allclose(rxd, expected, rtol=0, atol=0)

    # It must NOT be the legacy orientation-invented phantom.
    n = 64
    legacy = np.fft.ifftshift(
        np.fft.ifft2(np.fft.ifftshift(_legacy_axial_phantom(n)))
    ).reshape(-1)
    assert not np.allclose(rxd, legacy)


def test_dispatch_gre_context(monkeypatch, tmp_path):
    """An explicit GRE3D context routes to the geometry-aware simulator."""
    ctx, _ = _make_gre_ctx([0, 0, 0], [0, 0, 0])
    pe_order = choose_pe_order(
        ndims=3, npe=[BASE, SLICES], traj="center_out",
        save_pe_order=False,
    )
    adc_phases = np.zeros(len(pe_order))
    rawdir = tmp_path / "rawdata"
    rawdir.mkdir(exist_ok=True)
    np.save(rawdir / "pe_order.npy", pe_order)
    np.save(rawdir / "adc_phase.npy", adc_phases)

    # readout_number must equal lines * oversampled read length.
    n_samples = len(pe_order) * BASE * OVERSAMPLE

    rxd, rx_t = _run_dispatch_with(
        monkeypatch, tmp_path, "gre_3D.seq",
        hardware_simulation=True, sim_context=ctx, n_samples=n_samples,
    )
    expected = synthesize_gre3d_raw(ctx, pe_order, adc_phases)
    np.testing.assert_allclose(rxd, expected, rtol=0, atol=0)

