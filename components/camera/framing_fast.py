"""A faster framing solver (armature_core, components/camera/ext/armature), its own code untouched: wrappers of two of its internal functions.
  _bisect — f(lo) is not recomputed at every step, and there are STEPS steps instead of 80: the bracket error is width / 2^(STEPS+1)
            (radius 0.5–400 m — 0.0007 mm; with 24 steps on random cases — up to 0.012 mm, above the 0.01 threshold);
            refusals (non-numeric, unreachable) come from the solver's own _bisect, in its own words;
  _extent  — the cloud by one numpy projection instead of a per-point project loop; the position, axes and angles of the frame come from the solver's
            camera_position, camera_basis, half_fovs; a point behind the camera gives None, as in the original. The numbers are bit-identical
            (checked on 60 cases).
Measured: solve_camera on 166 points 0.58 → 0.011 s; the intent frame is the same to 0.0005 mm (11 cases).
Called by cam_intent.py: install(framing) after importing armature_core.framing; a repeated call changes nothing."""
import math

import numpy as np

STEPS = 28


def install(F):
    """Replace F._bisect and F._extent with the wrappers (module armature_core.framing) → F."""
    if getattr(F, "_sk_fast", False):
        return F
    bisect0, extent0 = F._bisect, F._extent

    def bisect(f, lo, hi, want, iters=80):
        if not F._finite(want):
            return bisect0(f, lo, hi, want, iters)
        flo, fhi = f(lo), f(hi)
        if not (F._finite(flo) and F._finite(fhi)) or (flo - want) * (fhi - want) > 0:
            return bisect0(f, lo, hi, want, iters)               # the solver's refusal — its own FramingError
        for _ in range(min(iters, STEPS)):
            mid = 0.5 * (lo + hi); fm = f(mid)
            if (flo - want) * (fm - want) <= 0:
                hi = mid
            else:
                lo, flo = mid, fm
        return 0.5 * (lo + hi)

    def extent(points, target, radius, az, el, lens, sensor, width, height):
        try:
            P = np.asarray(points, dtype=float)[:, :3]
        except (TypeError, ValueError, IndexError):
            P = None
        if P is None or not len(P):
            return extent0(points, target, radius, az, el, lens, sensor, width, height)
        pos = F.camera_position(target, radius, el, az)
        right, up, back = (np.asarray(v, dtype=float) for v in F.camera_basis(target, pos))
        V = P - np.asarray(pos, dtype=float); z = V @ back
        if z[0] >= -1e-9:                                         # the order of the solver's refusals: the first point behind the camera gives None
            return None
        hx, hy = F.half_fovs(lens, sensor, width, height)          # a wrong lens or frame — its FramingError
        if (z >= -1e-9).any():
            return None
        x = 0.5 + 0.5 * ((V @ right) / -z) / math.tan(hx)
        y = 0.5 - 0.5 * ((V @ up) / -z) / math.tan(hy)
        return float(x.min()), float(x.max()), float(y.min()), float(y.max())

    F._bisect, F._extent, F._sk_fast = bisect, extent, True
    return F
