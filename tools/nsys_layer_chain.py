#!/usr/bin/env python3
import argparse, collections, sqlite3, statistics

SGL_NAMES = [
    ('LowLatencyBlockscaledGemm', 'LL_gemm'), ('_rmsnorm_mxfp8', 'rmsnorm_mxfp8'), ('QRopeLowLatencyGemm', 'q_rope_wq_b'),
    ('sparse_decode_small', 'sparse_decode'), ('wo_a_fused', 'wo_a'), ('WoBPushMxfp8Gemm', 'wo_b_push'),
    ('fused_router_vr', 'router'), ('RoutedExpertGemm1', 'gemm1'), ('RoutedExpertGemm2', 'gemm2'),
    ('mhc_stats_decode', 'mhc_stats'), ('_kv_norm_rope_fp8_store', 'kv_store'), ('_prefetch_top_experts', 'prefetch'),
    ('SplitKGemm', 'sh_splitk'), ('silu_mul_clamp', 'silu'), ('PersistentDenseGemm', 'dense_gemm'),
    ('ffn_lamport_join', 'lamport_join'), ('MXFP8QuantizeLinear', 'mxfp8_q'), ('_fold_expert_use', 'fold'),
]

def short(name):
    if 'moe_finalize_all_reduce_kernel' in name:
        args = name.replace('(unsigned int)', '').split('<', 1)[1]
        topk = args.split(',')[2].strip().rstrip('u')
        return 'attn_AR' if topk == '1' else 'ffn_AR'
    for k, v in SGL_NAMES:
        if k in name:
            return v
    return name.split('(')[0].split('<')[0][:32]

def is_ffn_ar(n):
    return n == 'ffn_AR' or n.startswith(('comm_lp_sgl', 'comm_dec_ffn', 'comm_lp_ffn'))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sqlite')
    ap.add_argument('--device', type=int, default=None)
    ap.add_argument('--all-layers', action='store_true', help='keep indexer / compressor / engram layers too')
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    cur = db.cursor()
    names = dict(cur.execute('select id, value from StringIds').fetchall())
    cols = [r[1] for r in cur.execute('pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)').fetchall()]
    col = 'demangledName' if 'demangledName' in cols else 'shortName'
    dev = a.device if a.device is not None else min(r[0] for r in cur.execute('select distinct deviceId from CUPTI_ACTIVITY_KIND_KERNEL'))
    ev = [(s, e, short(names[n])) for s, e, n in cur.execute(f'select start, end, {col} from CUPTI_ACTIVITY_KIND_KERNEL where deviceId = {dev} order by start')]
    idx = [i for i, (s, e, n) in enumerate(ev) if is_ffn_ar(n)]
    stats = collections.defaultdict(lambda: ([], []))
    periods = []
    for a0, b0 in zip(idx, idx[1:]):
        t0, t1 = ev[a0][1], ev[b0][1]
        if (t1 - t0) / 1e3 > 200:
            continue
        seg = ev[a0 + 1:b0 + 1]
        if not a.all_layers and any(x in n.lower() for _, _, n in seg for x in ('idx_', 'topk', 'compress', 'paged_mqa', 'c2_decode', 'engram')):
            continue
        periods.append((t1 - t0) / 1e3)
        seen = set()
        for s, e, n in seg:
            if n in seen:
                continue
            seen.add(n)
            stats[n][0].append((s - t0) / 1e3)
            stats[n][1].append((e - t0) / 1e3)
    if not periods:
        print('no layers found')
        return
    print(f'{a.sqlite}: device {dev}, {len(periods)} layers, period median {statistics.median(periods):.2f} us')
    print('  (start / end: medians, us from the previous FFN all-reduce end)')
    for n, (ss, ee) in sorted(stats.items(), key=lambda kv: statistics.median(kv[1][1])):
        if len(ee) < 0.8 * len(periods):
            continue
        print(f'  {n:<28} start {statistics.median(ss):8.2f}  end {statistics.median(ee):8.2f}')

if __name__ == '__main__':
    main()
