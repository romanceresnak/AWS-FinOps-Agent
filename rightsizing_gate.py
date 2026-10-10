"""Rightsizing with abstention: a synthetic fleet with ground truth by construction.

Each instance gets 14 days of observed 5-minute CPU (the default Compute Optimizer
lookback) and 30 hidden "future" days from the same generator. Downsizing one size
(half the vCPUs) doubles utilisation. Ground truth: a downsize is WRONG if, in the
future 30 days, the smaller instance would run above 90% CPU for >= 15 minutes in total.

Policies
  threshold : downsize if observed p95 CPU < 40%  (projected p95 < 80% on the smaller size)
  gate      : per-day evidence. A day is "safe" if its peak, doubled, stays < 90%.
              confidence = (safe_days + 1) / (days + 2)   (Laplace rule of succession)
              >= 0.90 and no unsafe day -> downsize, < 0.50 -> keep, otherwise -> defer to a human
"""
import json, sys
import numpy as np

SEED = 7
STEP_MIN = 5
PER_DAY = 24 * 60 // STEP_MIN
OBS_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 14
FUT_DAYS = 30
MAX_HIST = 93
PRICE_LARGE_H = 0.1955 / 2          # m7g.large, eu-central-1 on-demand (half of m7g.xlarge $0.1955)
SAVING_MONTH = PRICE_LARGE_H * 730  # saved per correct downsize xlarge -> large

rng = np.random.default_rng(SEED)

def diurnal(days, base, amp):
    t = np.arange(days * PER_DAY)
    return base + amp * np.sin(2 * np.pi * (t % PER_DAY) / PER_DAY - np.pi / 2)

def series(kind, days, start_day, p):
    x = diurnal(days, p["base"], p["amp"]) + rng.normal(0, p["noise"], days * PER_DAY)
    if kind == "daily_spike":                       # nightly report, ~20 min
        for d in range(days):
            i = d * PER_DAY + p["hour"] * 12
            x[i:i + 4] = rng.uniform(55, 80, 4)
    if kind == "weekly_batch":                      # Sunday job, ~45 min
        for d in range(days):
            if (start_day + d) % 7 == 6:
                i = d * PER_DAY + 2 * 12
                x[i:i + 9] = rng.uniform(55, 85, 9)
    if kind == "monthly_batch":                     # month-end close, ~2 h
        for d in range(days):
            if (start_day + d) % 30 == 29:
                i = d * PER_DAY + 22 * 12
                x[i:i + 24] = rng.uniform(60, 90, 24)
    return np.clip(x, 0, 100)

def make(kind):
    p = dict(base=10, amp=3, noise=2, hour=int(rng.integers(0, 24)))
    obs_days = None
    if kind == "steady_idle":   p.update(base=rng.uniform(6, 16), amp=rng.uniform(2, 6))
    if kind == "steady_busy":   p.update(base=rng.uniform(40, 55), amp=rng.uniform(5, 10), noise=4)
    if kind == "noisy_moderate":p.update(base=rng.uniform(22, 34), amp=rng.uniform(4, 8), noise=rng.uniform(4, 8))
    if kind in ("daily_spike", "weekly_batch", "monthly_batch"):
        p.update(base=rng.uniform(5, 14), amp=rng.uniform(2, 5))
    if kind == "new_instance":
        obs_days = int(rng.integers(2, 6))
        p.update(base=rng.uniform(8, 30), amp=rng.uniform(2, 8), noise=rng.uniform(2, 6))
    # month-end falls inside the 14-day window for roughly half of the monthly jobs
    # every instance has 93 days of history (or less if new); the policy sees only the
    # last OBS_DAYS of it, so the 14-day and 93-day runs judge exactly the same fleet
    hist_days = obs_days if kind == "new_instance" else MAX_HIST
    start = int(rng.integers(0, 30))
    hist = series(kind, hist_days, start, p)
    fut = series(kind, FUT_DAYS, start + hist_days, p)
    keep = min(hist_days, OBS_DAYS)
    return hist[-keep * PER_DAY:], fut

def truth_safe(fut):
    return int(np.sum(fut * 2 > 90)) * STEP_MIN < 15

def threshold_policy(obs):
    return "downsize" if np.percentile(obs, 95) < 40 else "keep"

def gate_policy(obs):
    days = obs.reshape(-1, PER_DAY)
    safe = int(np.sum(days.max(axis=1) * 2 < 90))
    conf = (safe + 1) / (len(days) + 2)
    if conf >= 0.90 and safe == len(days): d = "downsize"   # one bad day blocks automation
    elif conf < 0.50: d = "keep"
    else: d = "defer"
    return d, conf, safe, len(days)

MIX = {"steady_idle": 60, "steady_busy": 40, "noisy_moderate": 20, "daily_spike": 25,
       "weekly_batch": 20, "monthly_batch": 15, "new_instance": 20}

rows = []
SAMPLE = {}
for kind, n in MIX.items():
    for _ in range(n):
        obs, fut = make(kind)
        SAMPLE.setdefault(kind, obs[-3 * PER_DAY:].round(1).tolist())
        g, conf, safe, nd = gate_policy(obs)
        rows.append(dict(kind=kind, safe=truth_safe(fut), threshold=threshold_policy(obs),
                         gate=g, conf=round(conf, 3), safe_days=safe, days=nd))

def score(policy):
    r = {"downsize": 0, "keep": 0, "defer": 0, "wrong_downsize": 0, "missed_saving": 0}
    for x in rows:
        d = x[policy]; r[d] += 1
        if d == "downsize" and not x["safe"]: r["wrong_downsize"] += 1
        if d == "keep" and x["safe"]: r["missed_saving"] += 1
    r["correct_downsize"] = r["downsize"] - r["wrong_downsize"]
    r["saving_month_usd"] = round(r["correct_downsize"] * SAVING_MONTH, 0)
    return r

out = {"instances": len(rows), "truly_safe": sum(x["safe"] for x in rows),
       "saving_per_downsize_usd_month": round(SAVING_MONTH, 2),
       "threshold": score("threshold"), "gate": score("gate"), "by_kind": {}}
for kind in MIX:
    sub = [x for x in rows if x["kind"] == kind]
    out["by_kind"][kind] = {
        "n": len(sub), "truly_safe": sum(x["safe"] for x in sub),
        "threshold": {d: sum(x["threshold"] == d for x in sub) for d in ("downsize", "keep")},
        "threshold_wrong": sum(x["threshold"] == "downsize" and not x["safe"] for x in sub),
        "gate": {d: sum(x["gate"] == d for x in sub) for d in ("downsize", "keep", "defer")},
        "gate_wrong": sum(x["gate"] == "downsize" and not x["safe"] for x in sub),
        "deferred_safe": sum(x["gate"] == "defer" and x["safe"] for x in sub),
    }
out["sample_last_3_days"] = {k: SAMPLE[k] for k in ("daily_spike", "steady_idle")}
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "sample_last_3_days"}, indent=2))
