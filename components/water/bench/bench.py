"""Wave engine bench without Blender (components/water/waves.py): height and period against the JONSWAP / Pierson–Moskowitz
formulas, field variance against the spectrum, slopes against Cox–Munk, foam fraction against Monahan, frame cost.
  .venv/bin/python -m components.water.bench.bench"""
import math
import sys
import time

import numpy as np

from components.water import waves as W


def hs_formula(u, F):
    """Hs by dimensionless formulas: ε = 1.6·10⁻⁷·χ (JONSWAP), the ceiling is 0.21·U19.5²/g (Pierson–Moskowitz)."""
    pm = 0.21 * (1.075 * u) ** 2 / W.G
    if F is None: return pm
    chi = W.G * F / u ** 2
    return min(4 * math.sqrt(1.6e-7 * chi) * u * u / W.G, pm)


def main():
    ok, rows = [], []
    print("Hs, m: engine / formula (wind × fetch)")
    for u in (2, 5, 8, 12, 18):
        line = []
        for F in (20, 200, 2000, 20000, 200000, None):
            hs, hf = W.Sea(u10=u, fetch=F).stats()["hs"], hs_formula(u, F)
            line.append(f"{hs:.3g}/{hf:.3g}"); ok.append(0.7 < hs / hf < 1.35)
        print(f"  {u:>2} m/s: " + "  ".join(line))
    print("field variance / spectrum (over 6 seeds):")
    for u, F in ((8, None), (8, 2000), (3, 20)):
        r = []
        for s in range(6):
            sea = W.Sea(u10=u, fetch=F, seed=s, near=0.5 if F == 20 else 2.0)
            v = sum(float(f["dz"].var()) for f in sea.eval(3.0)); m0 = (sea.stats()["hs"] / 4) ** 2
            r.append(v / m0)
        print(f"  {u} m/s, fetch {F}: {np.mean(r):.3f} ± {np.std(r):.3f}"); ok.append(0.85 < np.mean(r) < 1.15)
    print("slopes: bands + remainder = Cox–Munk (mean square):")
    for u, F in ((3, None), (8, None), (12, None), (8, 2000)):
        st = W.Sea(u10=u, fetch=F).stats()
        print(f"  {u} m/s, {F}: bands {sum(st['slope2']):.4f}, remainder {st['slope2_rest']:.4f}, C—M {st['slope2_cox_munk']:.4f}, "
              f"along/across {st['slope2_along'] / max(st['slope2_cross'], 1e-9):.2f}")
    print("foam: threshold after Monahan, the fraction at other moments / target")
    for u, F in ((3, None), (6, None), (10, None), (14, None), (20, None), (8, 2000), (14, 2000), (5, 20)):
        t0 = time.time(); sea = W.Sea(u10=u, fetch=F, near=0.5 if F == 20 else 2.0); tc = time.time() - t0
        c = np.mean([sea.coverage(t) for t in (5.0, 17.0)]); want = sea.stats()["whitecaps"]
        off = sea.fp["thr"] <= -9
        print(f"  {u} m/s, {F}: threshold {sea.fp['thr']:.3f}, {c * 100:.2f} % / {want * 100:.2f} %, build {tc:.1f} s" + (" (no foam)" if off else ""))
        ok.append(off and want < 2e-3 or 0.5 < c / max(want, 1e-9) < 2.0)
    print("drop ring: front speed / group speed of the peak")
    r = W.Rain(rate=0.0, L=2.0, N=512, film=0.0); I = np.zeros((r.N, r.N)); I[256, 256] = 1.0
    r._cache[0] = np.fft.fft2(I) * r.A
    rad = []
    for t in (0.15, 0.35):
        F = r.eval(t); a_ = np.abs(F["dz"]); yy, xx = np.mgrid[0:r.N, 0:r.N]; R_ = np.hypot(yy - 256, xx - 256) * r.L / r.N
        prof = np.array([a_[(R_ >= q) & (R_ < q + 0.004)].mean() for q in np.arange(0, 0.6, 0.004)])
        lead = np.arange(0, 0.6, 0.004)[np.nonzero(prof > 0.25 * prof.max())[0].max()]; rad.append(lead)
    v = (rad[1] - rad[0]) / 0.2; kpk = 1 / 0.004; cg = float(W.domega_dk(np.array(kpk)))
    print(f"  {v:.3f} m/s / {cg:.3f} m/s (spectrum peak of the dimple k ≈ 1/s = {kpk:.0f} rad/m)"); ok.append(0.6 < v / cg < 2.2)
    print("storm: height by Beaufort, the share of marble veins (rank threshold τ = √(1 − A) of two tiles)")
    from components.water import marble
    M = marble.tile()
    for b_, u in ((8, 19.0), (10, 26.5)):
        sea = W.Sea(u10=u, wind_to=(1, 0), hs=W.beaufort_hs(u), near=0.05); hs = sea.stats()["hs"]
        A = float(np.interp(u, [13.9, 17.2, 20.8, 24.5, 28.5, 32.7], [0.0, 0.08, 0.16, 0.26, 0.38, 0.5]))
        m2 = np.maximum(M[..., 1], np.roll(M[..., 1], (97, 211), (0, 1))); share = float((m2 > math.sqrt(1 - A)).mean())
        print(f"  {b_} Beaufort: Hs {hs:.1f} m / {W.beaufort_hs(u):.1f}, veins {share * 100:.1f} % / {A * 100:.0f} %")
        ok.append(0.85 < hs / W.beaufort_hs(u) < 1.15 and abs(share - A) < 0.03)
    print("wave travel downwind; a rogue wave (group focusing)")
    sea = W.Sea(u10=10.0, wind_to=(1.0, 0.0), near=2.0, seed=3, chop=0.0); b0 = sea.bands[0]
    z0, z1 = sea.eval(0.0, i=0)["dz"], sea.eval(0.5, i=0)["dz"]
    cc = np.fft.ifft2(np.fft.fft2(z1) * np.conj(np.fft.fft2(z0))).real; ix = int(np.unravel_index(np.argmax(cc), cc.shape)[1])
    dxw = (ix if ix < b0.N // 2 else ix - b0.N) * b0.L / b0.N
    hs9 = W.beaufort_hs(22.6); sea = W.Sea(u10=22.6, wind_to=(-1.0, 0.0), hs=hs9, chop=1.25, near=0.5, seed=7, focus={"xy": (-220.0, 0.0), "t": 30.0})
    X, Y = np.meshgrid(np.arange(-700, 300, 2.0), np.arange(-150, 151, 15.0)); xy = np.stack([X.ravel(), Y.ravel()], 1)
    cf = float(W.surface_at(sea, sea.eval(30.0), xy, iters=3)[0].max()); cb = float(W.surface_at(sea, sea.eval(20.0), xy, iters=3)[0].max())
    print(f"  in 0.5 s the waves moved {dxw:+.1f} m downwind; the crest at focus is {cf:.1f} m (Hs {hs9:.1f}), 10 s before — {cb:.1f} m")
    ok.append(dxw > 0 and cf > 1.25 * hs9 and cf > 1.6 * cb)
    print("torrent: Bélanger jump, standing wave, Manning")
    from components.water.flow import River, jump, manning
    J = jump(3.0, 0.8, 0.9); q_ = 3.0 * 0.8
    d2_ = J["d1"] / 2 * (math.sqrt(1 + 8 * (q_ / J["d1"]) ** 2 / (9.81 * J["d1"])) - 1)   # recomputed from the discharge and d₁
    mom = q_ * q_ / (9.81 * J["d1"]) + J["d1"] ** 2 / 2 - (q_ * q_ / (9.81 * J["d2"]) + J["d2"] ** 2 / 2)   # momentum before = after
    print(f"  d₁ {J['d1']:.3f} → d₂ {J['d2']:.3f} m (Fr₁ {J['fr1']:.2f}), momentum before − after {mom:.2e}, Manning {manning(0.8, 0.035, 0.045):.2f} m/s")
    ok.append(abs(d2_ - J["d2"]) < 1e-6 and abs(mom) < 1e-6)
    R = River([[q, 0.0] for q in np.arange(0, 80, 2.0)], 9.0, None, (-5, -8, 85, 8), res=0.1, slope=0.035, depth=0.8, n=0.045)
    row = R.Hb[R.ny // 2]; xs = R.x0 + np.arange(R.nx) * R.res; seg = (xs > 10) & (xs < 70)
    zc = np.nonzero(np.diff(np.sign(row[seg])) != 0)[0]; lam = 2 * float(np.median(np.diff(xs[seg][zc]))) if len(zc) > 3 else 0.0
    want = 2 * math.pi * R.v0 ** 2 / 9.81
    print(f"  standing wave along the axis: λ {lam:.2f} m / 2πv²/g {want:.2f} m")
    ok.append(abs(lam / want - 1) < 0.15)
    print("waterfalls: lip, fall, contact")
    from components.water.fall import Fall
    for k in ("plunge", "horsetail", "ribbon"):
        F = Fall(k); s_ = F.summary(); hc = (F.q ** 2 / 9.81) ** (1 / 3)
        tf = s_["fall_s"]; t0_ = math.sqrt(2 * F.H / 9.81)
        print(f"  {k}: h_c {s_['h_c']:.3f} / {hc:.3f} m, fall {tf:.2f} s (free {t0_:.2f}), impact {s_['v_impact']} m/s, contact {s_['contact']}")
        good = abs(s_["h_c"] - hc) < 1e-3 and (s_["contact"] > 0.8 if F.p["face"] == "slope" else (s_["contact"] < 0.05 and 1.0 <= tf / t0_ < 1.35))
        ok.append(good)
    sea = W.Sea(u10=8); t0 = time.time()
    for k in range(10): sea.eval(1.0 + k * 0.04)
    te = (time.time() - t0) / 10; t0 = time.time(); sea.foam(1.0); tf = time.time() - t0
    print(f"cost: fields of {len(sea.bands)} bands {te * 1e3:.1f} ms, foam {tf * 1e3:.0f} ms per frame")
    print(f"result: {sum(ok)}/{len(ok)}")
    return all(ok)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
