#!/usr/bin/env python3
import argparse
import collections
import sqlite3
import statistics

GROUPS = [
    (("moe_", "tmoe", "sgl_"), "MoE (gate + route + experts)"),
    (("comm_",), "all-reduce / gather"),
    (("attn",), "attention"),
    (("tc_gemv",), "dense projections (tcgen05: wqkv, wq_b, wo_a, wo_b)"),
    (("hc_",), "mHC side branch (dots, mix, prep)"),
    (("idx_", "topk", "compress", "rope", "scale_bf16"), "indexer / top-k / compressor / rope"),
    (("markov", "gemv_fp8b_persist", "spec_draft", "block_max", "embed"), "draft phase (Markov head, FP8 draft head)"),
    (("gemv_bf16_persist", "samp_", "spec_"), "verify head + sampler + accept"),
    (("engram",), "engram"),
    (("gemv",), "mma.sync GEMVs (drafter blocks, main_proj)"),
]

def group(name):
    return next((g for prefixes, g in GROUPS if name.startswith(prefixes)), "other")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--first", default="spec_draft_meta", help="the kernel that starts a step")
    ap.add_argument("--kernels", action="store_true", help="per kernel instead of per group")
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    names = dict(db.execute("select id, value from StringIds"))
    rows = [(s, e, names[n]) for s, e, n in db.execute(
        "select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and graphNodeId is not null order by start", (a.device,))]
    starts = [i for i, r in enumerate(rows) if r[2] == a.first]
    steps = [rows[starts[i]:starts[i + 1] + 1] for i in range(len(starts) - 1)][len(starts) // 5:]
    key = (lambda n: n) if a.kernels else group
    acc, calls, walls = collections.defaultdict(list), collections.defaultdict(list), []
    for st in steps:
        t0, t1, ks = st[0][0], st[-1][0], sorted(st[:-1])
        ev = sorted(set([t0, t1] + [min(max(t, t0), t1) for r in ks for t in r[:2]]))
        part, active, j = collections.defaultdict(float), [], 0
        for x, y in zip(ev, ev[1:]):
            while j < len(ks) and ks[j][0] <= x:
                active.append(ks[j])
                j += 1
            active = [r for r in active if r[1] > x]
            part[key(min(active)[2]) if active else "(no kernel running)"] += (y - x) / 1e3
        if part.get("(no kernel running)", 0.0) > 50:
            continue
        walls.append((t1 - t0) / 1e3)
        for k in set(part) | set(acc):
            acc[k].append(part.get(k, 0.0))
        n = collections.Counter(key(r[2]) for r in ks)
        for k in set(n) | set(calls):
            calls[k].append(n.get(k, 0))
    w = statistics.mean(walls)
    print(f"device {a.device}: {len(walls)} steps, mean step {w:.1f} us")
    print(f"| {'kernel' if a.kernels else 'group'} | launches / step | us / step | share |\n|---|---:|---:|---:|")
    for k, v in sorted(acc.items(), key=lambda x: -statistics.mean(x[1]))[:a.top]:
        print(f"| {k} | {statistics.median(calls[k]) if calls[k] else 0:.0f} | {statistics.mean(v):.1f} | {100 * statistics.mean(v) / w:.1f}% |")

if __name__ == "__main__":
    main()
