import csv, math
from collections import defaultdict

MIN_SHARE = 0.15   # do not starve a channel
MAX_SHIFT = 0.20   # max daily reallocation per channel
CHANNELS = ["Search", "Social", "Display"]

def load_data(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["spend"] = float(r["spend"]); r["impressions"] = int(r["impressions"])
            r["clicks"] = int(r["clicks"]); r["conversions"] = int(r["conversions"])
            rows.append(r)
    return rows

def daily_metrics(rows):
    by_day = defaultdict(list)
    for r in rows: by_day[r["date"]].append(r)
    return dict(sorted(by_day.items()))

def score(day_rows, metric="conversions"):
    perf = {}
    for r in day_rows:
        if metric == "ctr":
            perf[r["channel"]] = (r["clicks"] / max(1, r["impressions"]))
        else:
            perf[r["channel"]] = r["conversions"]
    return perf

def reallocate(prev_split, perf):
    # normalize perf to shares
    total = sum(perf.values()) or 1.0
    target = {c: (perf[c] / total) for c in CHANNELS}
    new = {}
    for c in CHANNELS:
        # move toward target, but clamp shifts and keep a minimum share
        delta = max(-MAX_SHIFT, min(MAX_SHIFT, target[c] - prev_split[c]))
        new[c] = max(MIN_SHARE, prev_split[c] + delta)
    # renormalize
    s = sum(new.values())
    return {c: v/s for c, v in new.items()}

def run(path, metric="conversions"):
    rows = load_data(path)
    days = daily_metrics(rows)
    split = {c: 1/len(CHANNELS) for c in CHANNELS}
    history = []
    for d, recs in days.items():
        perf = score(recs, metric)
        split = reallocate(split, perf)
        rationale = f"{d}: perf={perf} → split={split}"
        history.append(rationale)
        print(rationale)
    return history

# Example:
# run("campaign_data.csv", metric="conversions")
