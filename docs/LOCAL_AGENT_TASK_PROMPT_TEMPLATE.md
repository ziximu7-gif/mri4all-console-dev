# Local Coding Agent Task Prompt Template

Use this template for one implementation task at a time.

---

You are working in the local repository for MRI4ALL Console.

## Mandatory context

Before doing anything:

1. read `AGENTS.md`;
2. read `MRI4ALL_ARCHITECTURE.md` (or `docs/MRI4ALL_ARCHITECTURE.md`);
3. run `git status`;
4. run `git rev-parse --abbrev-ref HEAD`;
5. run `git rev-parse HEAD`;
6. inspect the relevant current files rather than relying on remembered code.

The human owns architecture decisions. Your role is to investigate and implement an explicitly defined design with minimal changes.

## Mode

`PLAN_ONLY`

Do not modify files in PLAN_ONLY mode.

First return an implementation plan. I will review it and then explicitly switch you to `IMPLEMENT`.

## Task

[Describe exactly one feature or bug.]

## User-visible goal

[What should the operator see or be able to do?]

## Architectural decision already made

[State the intended behavior. Do not ask the agent to decide this.]

Example:

- Localizer remains scanner-base Axial/Coronal/Sagittal.
- A rotated FOV defines FOV-local X'/Y'/Z'.
- GRE Orientation + Readout Direction define Read/Phase/Third inside that FOV frame.
- Reconstructed GRE remains FOV-native.
- Completed-scan overlays are read-only.

## Non-goals

Do not:

- [non-goal 1]
- [non-goal 2]
- [non-goal 3]

## Allowed files

Prefer changes only in:

- `path/to/file1.py`
- `path/to/file2.py`
- `tests/test_x.py`

If another file is required, stop and explain why before editing it.

## Forbidden / high-risk files

Do not modify unless I explicitly approve:

- `common/types.py`
- `pypulseq/`
- TSE3D-related files
- B0/shim application code
- dependency/version files

Adjust this list per task.

## Invariants to preserve

- `rotation_local_to_scanner` means FOV-local -> scanner.
- `logical_to_scanner` means Read/Phase/Third -> scanner.
- `logical_to_scanner = rotation_local_to_scanner @ encoding_matrix`.
- GRE reconstruction axes remain Read/Phase/Third.
- PE Ordering changes sampling order, not spatial encoding direction.
- No patient-space DICOM orientation claim without a defined scanner -> Patient LPS transform.
- Existing B0 shim behavior must remain unchanged.

## Acceptance criteria

1. [observable criterion]
2. [geometry/math criterion]
3. [backward-compatibility criterion]
4. [test criterion]

## Required tests

Run:

```bash
[exact test command]
```

Add a focused regression test if the implementation changes geometry or low-level gradient composition.

## PLAN_ONLY output format

Return only:

### Current state
- branch:
- HEAD:
- dirty files:

### Call path inspected
- ...

### Proposed implementation
- file:
  - exact change:
  - reason:

### Invariants preserved
- ...

### Risks / ambiguities
- ...

### Tests
- ...

Do not edit anything yet.

---

# After I approve the plan

I will send:

`IMPLEMENT THE APPROVED PLAN`

Then:

1. implement only the approved plan;
2. keep the diff minimal;
3. run the approved tests;
4. do not make a commit unless explicitly asked;
5. report changed files, test output summary, `git diff --stat`, and unresolved risks.
