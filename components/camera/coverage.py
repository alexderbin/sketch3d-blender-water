# our order of rules (key events are not eaten by an exchange; one axis); the cut is the boundary of a piece
"""Coverage of a take by beats — which shot is on each piece of time. Dialogue coverage rules carried over to combat, in an order of our own
(a key event is not eaten by an exchange):
  1) a key event — a shot by weight: a death from a gunshot — over the shooter's shoulder at the target at the moment of the fall; a hit
     and a finishing blow — "two"; an explosion — a wide shot of those hit; after a death from a gunshot, if there is room — a reaction:
     a medium shot of the shooter;
  2) an exchange of fire — over the shooter's shoulder at the target for the firing window (two bursts of one shooter — one shot);
  3) running and attack — following the runner;
  4) a wide shot at the start (who is where) and in the gaps — a "breath".
A key shot gets a smooth push-in to the peak at the moment of the event (camera signal envelopes): "key" is the event frame,
"push_in" is the fraction.
Edit: the cut comes 0.15 s before the event starts (the action begins already in the new shot); a shot is no shorter than "min" and no
longer than "max" (a long one is split in two with a wide shot); two identical ones in a row are one shot; one axis (bl_take sets the camera side).
.venv/bin/python -m components.camera.coverage <events.json> <shots.json> [pace=2.0] [min=0.7] [max=3.0]
Input — components/time/bl_take.py (pass 1): fps, frames, sides, events [{type, k0, k1, who, target|off, weight, …}].
The answer is the __RESULT__ line: shots, by shot kind, average length, key events covered, cuts per minute."""
import collections, json, sys

src, dst = sys.argv[1], sys.argv[2]; opt = dict(a.split("=", 1) for a in sys.argv[3:])
E = json.load(open(src)); FPS, NF = float(E["fps"]), int(E["frames"])
TEMPO, MIN, MAX = float(opt.get("pace", E.get("pace", 2.0))), float(opt.get("min", 0.7)), float(opt.get("max", 3.0))
fr = lambda s: int(round(s * FPS)); MINF, MAXF, LEAD = max(2, fr(MIN)), max(3, fr(MAX)), fr(0.15)
EV = sorted(E["events"], key=lambda e: (-e["weight"], e["k0"]))
ALL = sorted({a for e in E["events"] for a in (e.get("who"), e.get("target"), e.get("off")) if isinstance(a, str)} | set(E.get("actors", [])))


def demand(plan, a, b, prio, why, **kw):
    return {"plan": plan, "a": max(0, a), "b": min(NF, b), "prio": prio, "why": why, **kw}


D, VOLLEY = [], set()
deaths = sorted((e for e in EV if e["type"] == "death" and e.get("off") in ALL), key=lambda e: e["k0"]); g = []
for e in deaths + [None]:                                            # volley: 3 or more deaths within 1.5 s — one shared shot for all participants
    if e is not None and (not g or e["k0"] - g[0]["k0"] <= fr(1.5)): g.append(e); continue   # (an over-the-shoulder shot for each would crowd each other out)
    if len(g) >= 3:                                                  # volley — from the shooters' side: over the shoulder of the one who fires most often,
        sh_ = collections.Counter(x["off"] for x in g).most_common(1)[0][0]; VOLLEY |= {id(x) for x in g}   # at the middle of those falling, wider
        tg = sorted((x["who"] for x in g), key=lambda v: sum(abs(x["k0"] - [y["k0"] for y in g if y["who"] == v][0]) for x in g))[0]
        D.append(demand("via", g[0]["k0"] - LEAD - fr(0.3), g[-1]["k0"] + fr(1.0), 3, f"volley: {', '.join(x['who'] for x in g)} fall", who=sh_, target=tg,
                        lens=26, group=sorted({x["who"] for x in g} | {x["off"] for x in g}), key=g[len(g) // 2]["k0"], push_in=0.12))
    g = [e] if e is not None else []
D.append(demand("wide", 0, fr(1.4), 2.5, "who is where", group=ALL, who=ALL[0] if ALL else None))   # a wide shot at the start — above the exchange
for e in EV:                                                         # 1) key events — by weight; 2) exchange; 3) movement
    k0, k1 = e["k0"], e["k1"]
    if id(e) in VOLLEY: continue
    if e["type"] == "death" and e.get("off") and e["off"] in ALL:
        D.append(demand("via", k0 - LEAD - fr(0.25), k0 + fr(1.0), 3, f"{e['who']} falls from the shot of {e['off']}", who=e["off"], target=e["who"], key=k0, push_in=0.18))
        D.append(demand("medium", k0 + fr(1.0), k0 + fr(1.9), 1.5, f"reaction: {e['off']} after the hit", who=e["off"]))
    elif e["type"] == "strike":
        D.append(demand("two_shot", k0 - LEAD - fr(0.45), k0 + fr(1.2), 3, f"{e['who']} strikes {e.get('target')}", who=e["who"], target=e.get("target"), key=k0, push_in=0.07))
    elif e["type"] == "explosion":
        grp = [a for a in e.get("affected", []) if a in ALL] or ALL
        D.append(demand("wide", k0 - LEAD - fr(0.2), k0 + fr(1.8), 3, "explosion: who was hit", group=grp, who=grp[0]))
    elif e["type"] == "fire" and e.get("target"):
        D.append(demand("via", k0 - LEAD, min(k1 + fr(0.3), k0 + fr(2.5)), 2, f"{e['who']} shoots at {e['target']}", who=e["who"], target=e["target"]))
    elif e["type"] == "run" and k1 - k0 >= MINF:
        D.append(demand("follow", k0, k1, 1, f"{e['who']} runs", who=e["who"]))
own, plans = [None] * NF, sorted(D, key=lambda d: -d["prio"])       # a piece of the frame goes to whoever has higher priority (key events first)
for i, d in enumerate(plans):
    free = [k for k in range(d["a"], d["b"]) if own[k] is None]
    if len(free) < (MINF if d["prio"] < 3 else 2): continue          # a scrap is enough for a key event — we do not lose it
    for k in free: own[k] = i
first = next((k for k in range(NF) if own[k] is not None), NF)
est = {"plan": "wide", "prio": 0, "why": "who is where", "group": ALL, "who": ALL[0] if ALL else None}
segs, cur = [], None
for k in range(NF):                                                  # gaps: a wide shot at the start; later the previous shot continues
    d = plans[own[k]] if own[k] is not None else (est if (k < first or not segs) else None)
    if d is None: segs[-1][2] = k + 1; continue
    if cur is not d: segs.append([d, k, k + 1]); cur = d
    else: segs[-1][2] = k + 1
key_ = lambda d: (d["plan"], d.get("who"), d.get("target"), tuple(d.get("group") or ()))
out = []
for d, a, b in segs:                                                 # identical ones in a row — one shot
    if out and key_(out[-1][0]) == key_(d): out[-1][2] = b
    else: out.append([d, a, b])
changed = True
while changed and len(out) > 1:                                      # short ones go to a more important neighbour (cuts no more often than "min")
    changed = False
    for j, (d, a, b) in enumerate(out):
        if b - a >= MINF: continue
        nb = [x for x in (j - 1, j + 1) if 0 <= x < len(out)]
        t = max(nb, key=lambda x: (out[x][0]["prio"], out[x][2] - out[x][1]))
        if t < j: out[t][2] = b
        else: out[t][1] = a
        del out[j]; changed = True; break
j = 0
while j < len(out) - 1:                                              # two wide shots in a row — almost the same angle (a jump cut): one shot
    if out[j][0]["plan"] == out[j + 1][0]["plan"] == "wide" and out[j + 1][2] - out[j][1] <= MAXF:
        out[j][2] = out[j + 1][2]; del out[j + 1]
    else: j += 1
shots = []
for d, a, b in out:                                                  # long ones are split in two: the second half as a wide shot (or a close-up)
    parts = [(a, b)] if b - a <= MAXF else [(a + (b - a) * q // ((b - a + MAXF - 1) // MAXF), a + (b - a) * (q + 1) // ((b - a + MAXF - 1) // MAXF))
                                           for q in range((b - a + MAXF - 1) // MAXF)]
    for q, (pa, pb) in enumerate(parts):
        dd = d if q % 2 == 0 or d["plan"] == "wide" else est
        if shots and key_(shots[-1]) == key_(dd): shots[-1]["k1"] = pb; continue
        shots.append({"k0": pa, "k1": pb, **{k: v for k, v in dd.items() if k not in ("a", "b", "prio")}})
keys = [e for e in E["events"] if e["weight"] >= 3]
def covered(e):                                                      # a key event is covered: in a shot whose heroes are its participants
    who = {e.get("who"), e.get("target"), e.get("off")} - {None}
    sh = next((s for s in shots if s["k0"] <= e["k0"] < s["k1"]), None)
    return bool(sh) and (sh["plan"] == "wide" or bool(who & ({sh.get("who"), sh.get("target")} | set(sh.get("group") or ()))))
json.dump({"fps": FPS, "frames": NF, "shots": shots}, open(dst, "w"), ensure_ascii=False, indent=1)
L = [(s["k1"] - s["k0"]) / FPS for s in shots]
print("__RESULT__" + json.dumps({"n_shots": len(shots), "by_plans": dict(collections.Counter(s["plan"] for s in shots)),
                                 "mean_s": round(sum(L) / max(len(L), 1), 2), "min_s": round(min(L), 2) if L else None,
                                 "cuts_per_minute": round((len(shots) - 1) / (NF / FPS) * 60, 1),
                                 "n_key_events": len(keys), "key_covered": sum(covered(e) for e in keys)}, ensure_ascii=False))
