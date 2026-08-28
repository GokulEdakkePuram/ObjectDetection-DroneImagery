# Experiment log

A running journal. The point is to record what was expected *before* each run,
so that being wrong stays visible instead of getting quietly rewritten into a
tidy narrative afterwards.

Template for each entry:

---

## YYYY-MM-DD — <short title>

**Question.** What am I trying to find out?

**Expectation.** What I think will happen, and why. Written before running.

**Setup.** Config, commit SHA, hardware, wall-clock time.

**Result.** Numbers. Link the run directory.

**Reading.** Did the expectation hold? If not, what is the most likely
explanation, and what would distinguish between the candidates?

**Next.** The single most informative experiment to run after this one.

---

## 2026-08-28 — Pipeline smoke test

**Question.** Does the whole path — download, convert, train, validate, export
— actually run end to end before any real compute is committed to it?

**Expectation.** Yes, with meaningless metrics (1 epoch on 2% of the data).

**Setup.** `configs/smoke.yaml`, YOLO11n, 640 px, MPS on an M2 Pro (16 GB).

**Result.** _to fill in_

**Reading.** _to fill in_

**Next.** `baseline_640`, the control for the resolution ablation.
