#!/usr/bin/env python
"""
Reproducible metrics for Chapter 4 (Results & Discussion).

Recomputes every CTE / heading / steering / windowed-RMSE / confusion-matrix
figure directly from the raw telemetry JSON in this folder, so each number in
the thesis can be regenerated with a single command:

    python metrics/compute_metrics.py

All statistics use population standard deviation (ddof=0) to match the values
reported in the thesis tables. Windowed RMSE uses non-overlapping full
500-frame windows; any partial tail window is dropped (stated in Section 4.4.1).
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW = 500            # frames per non-overlapping window (~8.3 s at 60 Hz)
CTE_THRESH = 2.0        # px, dead-band for the "desired action" label
STEER_THRESH = 2.0      # steering units, dead-band for the executed action

# Physical scale: the bird's-eye-view canvas is ARENA_PX x ARENA_PX pixels and
# maps to a square arena of (approximately) ARENA_CM per side. The physical size
# is not exact (~2.0-3.0 m), so cm figures are reported as approximate.
ARENA_PX = 800
ARENA_CM = 250.0
CM_PER_PX = ARENA_CM / ARENA_PX   # ~0.3125 cm/px at the nominal 250 cm

# Sign convention (confirmed from telemetry): steering > 0 = Right, CTE > 0 = Right.
# CTE > 0 means the robot is displaced to the right of the path, so the *corrective*
# steering direction is Left; this is why the desired-action rule maps CTE>0 -> Left.

SCENARIOS = {
    "45 degree": "Scenario Runs/45 degree.json",
    "90 degree": "Scenario Runs/90 degree.json",
    "u turn":    "Scenario Runs/u turn.json",
}
PRELIM = {  # label -> (file, table caption)
    "Baseline A":          "Preliminary Tests/A.json",
    "Baseline B (sess.1)": "Preliminary Tests/B.json",
    "Baseline B (sess.2)": "Preliminary Tests/B_2.json",
    "Egocentric (final)":  "Preliminary Tests/Ego.json",
}


def load(rel):
    with open(os.path.join(HERE, rel), encoding="utf-8") as fh:
        return json.load(fh)


def mean(xs):
    return sum(xs) / len(xs)


def pstdev(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def rmse(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def percentile(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def windowed_rmse(cte, size=WINDOW):
    """RMSE of each non-overlapping full window; partial tail dropped."""
    wins = [cte[i:i + size] for i in range(0, len(cte), size)]
    wins = [w for w in wins if len(w) == size]
    return [rmse(w) for w in wins]


def action_from_cte(c):
    if c > CTE_THRESH:
        return "Left"
    if c < -CTE_THRESH:
        return "Right"
    return "Straight"


def action_from_steer(s):
    # steering > 0 = Right (confirmed convention). The corrective action labelled
    # "Left" for CTE>0 is executed by a negative (leftward) steering command, so the
    # diagonal of the confusion matrix represents correct corrective steering.
    if s > STEER_THRESH:
        return "Right"
    if s < -STEER_THRESH:
        return "Left"
    return "Straight"


def smoothed_heading(frames, win=15):
    """Centered moving-average of heading_deg to suppress per-frame ArUco jitter."""
    h = [f["heading_deg"] for f in frames]
    out = []
    half = win // 2
    for i in range(len(h)):
        a = max(0, i - half)
        b = min(len(h), i + half + 1)
        out.append(sum(h[a:b]) / (b - a))
    return out


def steering_yaw_bins(frames, bin_width=5, win=15, min_n=20):
    """Mean measured yaw-rate (deg/s) per steering bin, using smoothed heading.

    Per-frame heading differencing is dominated by marker jitter (yaw spikes of
    several hundred deg/s); smoothing first recovers the true monotonic
    steering->yaw relationship. Positive steering -> positive (rightward) yaw.
    """
    sm = smoothed_heading(frames, win)
    ts = [f["timestamp_ms"] for f in frames]
    stg = [f["steering"] for f in frames]
    bins = {}
    for i in range(1, len(frames)):
        dt = (ts[i] - ts[i - 1]) / 1000.0
        if dt <= 0:
            continue
        dh = sm[i] - sm[i - 1]
        if dh > 180:
            dh -= 360
        elif dh < -180:
            dh += 360
        key = round(stg[i - 1] / bin_width) * bin_width
        bins.setdefault(key, []).append(dh / dt)
    return {k: v for k, v in bins.items() if len(v) >= min_n}


def confusion(frames):
    """3x3 counts: rows = desired (from CTE), cols = executed (from steering)."""
    labels = ["Left", "Straight", "Right"]
    mat = {r: {col: 0 for col in labels} for r in labels}
    for fr in frames:
        mat[action_from_cte(fr["cte_px"])][action_from_steer(fr["steering"])] += 1
    return labels, mat


def fmt_confusion(labels, mat):
    out = ["        " + "".join(f"{c:>10}" for c in labels) + "   (executed)"]
    for r in labels:
        out.append(f"{r:>8}" + "".join(f"{mat[r][c]:>10}" for c in labels))
    out.append("(desired, from CTE)")
    return "\n".join(out)


def main():
    print("=" * 78)
    print("PRELIMINARY TESTS  (Table 4.1 — Physical Track Testing)")
    print("=" * 78)
    print(f"{'Run':<22}{'Frames':>9}{'RMSE (px)':>12}{'Mean|CTE| (px)':>16}")
    for label, rel in PRELIM.items():
        d = load(rel)
        print(f"{label:<22}{d['total_frames']:>9}{d['rmse_px']:>12.4f}{d['mean_abs_cte_px']:>16.4f}")

    print()
    print("=" * 78)
    print("SCENARIO RUNS  (Tables 4.2 / 4.2b and Section 4.4)")
    print("=" * 78)
    combined = []
    for name, rel in SCENARIOS.items():
        d = load(rel)
        fr = d["frames"]
        combined.extend(fr)
        cte = [f["cte_px"] for f in fr]
        acte = [abs(x) for x in cte]
        hdeg = [f["heading_deg"] for f in fr]
        steer = [f["steering"] for f in fr]
        ts = [f["timestamp_ms"] for f in fr]
        dur = (ts[-1] - ts[0]) / 1000.0
        wr = windowed_rmse(cte)

        print(f"\n--- {name}  ({len(fr)} frames, {dur:.1f} s, ~{len(fr)/dur:.1f} Hz) ---")
        print(f"  CTE RMSE          : {rmse(cte):.4f} px  (~{rmse(cte)*CM_PER_PX:.2f} cm)   (stored {d['rmse_px']})")
        print(f"  mean|CTE| +/- SD  : {mean(acte):.2f} +/- {pstdev(acte):.2f} px  (~{mean(acte)*CM_PER_PX:.2f} cm)   (stored mean {d['mean_abs_cte_px']})")
        print(f"  P95 |CTE|         : {percentile(acte, 0.95):.2f} px  (~{percentile(acte,0.95)*CM_PER_PX:.2f} cm)")
        print(f"  max |CTE|         : {max(acte):.2f} px  (~{max(acte)*CM_PER_PX:.2f} cm)")
        print(f"  heading_deg       : {mean(hdeg):.2f} +/- {pstdev(hdeg):.2f} deg, peak |{max(abs(x) for x in hdeg):.2f}| deg")
        print(f"  max |steering|    : {max(abs(s) for s in steer)}  (range {min(steer)} .. {max(steer)})")
        print(f"  windowed RMSE     : n={len(wr)} of {WINDOW}-frame windows; "
              f"{mean(wr):.2f} +/- {pstdev(wr):.2f} px  (range {min(wr):.2f}..{max(wr):.2f})")

    print("\n" + "=" * 78)
    print("CONFUSION MATRICES  (Section 4.5)   thresholds: "
          f"CTE +/-{CTE_THRESH} px, steering +/-{STEER_THRESH}")
    print("=" * 78)
    for name, rel in SCENARIOS.items():
        d = load(rel)
        labels, mat = confusion(d["frames"])
        print(f"\n[{name}]")
        print(fmt_confusion(labels, mat))
    labels, mat = confusion(combined)
    print("\n[combined - all scenarios]")
    print(fmt_confusion(labels, mat))
    tot = sum(mat[r][c] for r in labels for c in labels)
    diag = sum(mat[l][l] for l in labels)
    print(f"diagonal agreement = {diag}/{tot} = {100*diag/tot:.1f}%")

    print("\n" + "=" * 78)
    print("STEERING (normalized) -> MEASURED YAW-RATE  (physical interpretation)")
    print(f"arena px->cm scale: {ARENA_PX} px = ~{ARENA_CM:.0f} cm  ->  {CM_PER_PX:.4f} cm/px (approx)")
    print("=" * 78)
    for name, rel in SCENARIOS.items():
        d = load(rel)
        bins = steering_yaw_bins(d["frames"])
        print(f"\n[{name}]  (5-unit steering bins, smoothed yaw, n>=20)")
        for k in sorted(bins):
            v = bins[k]
            print(f"  steer ~{k:>4}: n={len(v):5d}  mean yaw = {mean(v):7.1f} deg/s")


if __name__ == "__main__":
    main()
