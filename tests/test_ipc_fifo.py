"""
Regression tests for IPC FIFO ownership and creation semantics.

Background: the acquisition service used to crash at import time with
FileExistsError on /tmp/mri4all/pipes/acq_pipe because

1. several sequence modules created a module-level
   Communicator(Communicator.ACQ) merely by being imported (the
   sequence registry auto-imports every sequence), and
2. Communicator.mkfifo() used a non-atomic unlink/recreate pattern
   that raced across the UI / ACQ / RECON processes.

These tests pin the fixed architecture:

- importing sequence modules must not create service IPC resources;
- FIFO creation is idempotent: an existing valid FIFO is reused
  (same inode), never unlinked/recreated;
- an existing non-FIFO path fails clearly with RuntimeError;
- concurrent FIFO creation on the same path must not raise.
"""

import os
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from common.ipc import Communicator

REPO_ROOT = Path(__file__).resolve().parents[1]


def _mkfifo_unbound(path: str) -> None:
    # Communicator.mkfifo does not touch instance state; call it
    # unbound so the test does not need a Qt object or real pipes.
    Communicator.mkfifo(None, path)


def test_mkfifo_creates_missing_fifo(tmp_path):
    fifo_path = str(tmp_path / "acq_pipe")

    _mkfifo_unbound(fifo_path)

    assert stat.S_ISFIFO(os.stat(fifo_path).st_mode)


def test_mkfifo_reuses_existing_fifo_without_recreating(tmp_path):
    fifo_path = str(tmp_path / "acq_pipe")
    os.mkfifo(fifo_path)
    inode_before = os.stat(fifo_path).st_ino

    # Second creation attempt must be a no-op on the same FIFO.
    _mkfifo_unbound(fifo_path)

    assert stat.S_ISFIFO(os.stat(fifo_path).st_mode)
    assert os.stat(fifo_path).st_ino == inode_before


def test_mkfifo_rejects_existing_non_fifo(tmp_path):
    blocked_path = tmp_path / "acq_pipe"
    blocked_path.write_text("not a fifo")

    with pytest.raises(RuntimeError, match="not a FIFO"):
        _mkfifo_unbound(str(blocked_path))

    # The offending file must be left untouched for diagnosis.
    assert blocked_path.read_text() == "not a fifo"


def test_concurrent_mkfifo_same_path(tmp_path):
    fifo_path = str(tmp_path / "acq_pipe")
    errors = []

    def worker():
        try:
            _mkfifo_unbound(fifo_path)
        except Exception as e:  # noqa: BLE001 - collect for assertion
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    assert stat.S_ISFIFO(os.stat(fifo_path).st_mode)


def test_sender_only_communicator_does_not_touch_input_fifo(
    tmp_path, monkeypatch
):
    # A sender-only (non-owning) Communicator computes its pipe paths
    # but must not create or unlink the inbound FIFO, which belongs to
    # the legitimate owner of that pipe end.
    monkeypatch.setattr(Communicator, "base", str(tmp_path))

    in_path = tmp_path / "acq_pipe"
    assert not in_path.exists()

    sender = Communicator(Communicator.ACQ, owns_input_fifo=False)

    assert sender.in_file == str(in_path)
    assert sender.out_file == str(tmp_path / "ui_recon_acq")
    assert not in_path.exists(), "sender-only mode created the inbound FIFO"

    # Even when the owner's FIFO already exists, sender-only mode must
    # leave it untouched (same inode -> not unlinked/recreated).
    os.mkfifo(str(in_path))
    inode_before = in_path.stat().st_ino

    sender2 = Communicator(Communicator.ACQ, owns_input_fifo=False)

    assert sender2.in_file == str(in_path)
    assert in_path.exists()
    assert in_path.stat().st_ino == inode_before


def test_owning_communicator_creates_input_fifo(tmp_path, monkeypatch):
    # Default (owner) behavior must be preserved: the inbound FIFO is
    # created idempotently.
    monkeypatch.setattr(Communicator, "base", str(tmp_path))

    in_path = tmp_path / "acq_pipe"

    owner = Communicator(Communicator.ACQ)

    assert owner.owns_input_fifo is True
    assert stat.S_ISFIFO(in_path.stat().st_mode)


# Runs in a FRESH subprocess so sys.modules caching cannot hide
# import-time side effects, and with Communicator.base redirected to a
# temp dir so any residual import-time FIFO creation becomes visible.
_SUBPROCESS_SCRIPT = """
import sys
from pathlib import Path

import common.ipc as ipc
ipc.Communicator.base = sys.argv[1]

import sequences  # noqa: F401 - registry auto-imports every sequence

names = sorted(p.name for p in Path(sys.argv[1]).glob("*"))
print("CREATED:" + ",".join(names))
"""


def test_import_sequences_has_no_ipc_side_effects(tmp_path):
    base = tmp_path / "pipes"

    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT, str(base)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr

    marker_lines = [
        line for line in result.stdout.splitlines()
        if line.startswith("CREATED:")
    ]
    assert marker_lines, (
        "subprocess did not report marker; stdout:\n" + result.stdout
    )
    assert marker_lines[-1] == "CREATED:", (
        "importing sequences created IPC files: " + marker_lines[-1]
    )
