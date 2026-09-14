# AGENTS.md — MRI4ALL Local Agent Working Agreement

This repository is being developed with a human-owned architecture and an AI coding agent used mainly for implementation.

Before changing code, read:

1. `MRI4ALL_ARCHITECTURE.md` (or the copy placed under `docs/`)
2. the relevant source files
3. the current `git diff`
4. the current `HEAD`

## Authority model

The human is the architecture owner.

You are an implementation and analysis agent.

You may propose architecture changes, but do not implement an unapproved architecture change.

## Required pre-edit response

Before making edits, report:

- current branch and HEAD;
- current dirty files;
- the call path you inspected;
- files you propose to change;
- architecture invariants you will preserve;
- tests you intend to run;
- any ambiguity that could change coordinate meaning, persistent data format, acquisition behavior, reconstruction behavior, or DICOM meaning.

If an ambiguity affects architecture, stop and ask.

## Scope rules

Do not silently:

- change coordinate-system conventions;
- change Euler sign/order;
- change `logical_to_scanner` semantics;
- rename persistent `ScanTask.other` keys;
- change reconstruction from FOV-native to scanner-native;
- resample data unless explicitly requested;
- change scanner-to-patient/DICOM orientation assumptions;
- integrate planning into TSE3D;
- alter B0 shim behavior;
- modify local `pypulseq/` without a regression test;
- upgrade dependencies;
- perform broad refactors or formatting sweeps.

## Editing rules

Prefer minimal diffs.

Do not change unrelated files.

When adding geometry math, prefer a pure helper with a focused unit test instead of embedding nontrivial matrix math directly in a UI callback.

For every matrix, state its direction in a comment where ambiguity is possible, for example:

```text
logical -> scanner
scanner -> logical
FOV-local -> scanner
```

Do not infer a transpose direction from naming alone.

## Verification

Run focused tests for the changed area.

For Localizer / GRE geometry work, commonly relevant tests are:

```bash
python -m pytest tests/test_geometry.py -q
python -m pytest tests/test_cartesian_translation.py -q
python -m pytest tests/test_gre_gradient_transform.py -q
python -m pytest tests/test_gre_slab_selection.py -q
```

If B0/shim code is changed:

```bash
python -m pytest tests/test_b0_shim.py -q
```

Never weaken assertions just to make tests pass.

## Required final report

After implementation, provide:

1. changed files;
2. one-sentence purpose of each change;
3. architecture invariants preserved;
4. tests run and exact result;
5. `git diff --stat`;
6. unresolved risks or assumptions;
7. anything that should be reviewed manually in the GUI.

Do not create a commit unless explicitly asked.
