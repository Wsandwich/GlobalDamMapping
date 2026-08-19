"""
02_mc_dof_simulation_v4.py — MC DOF 核心模拟（河段经验分位数版）

相对 v3 的变化：
- 保留相同的 Bernoulli precision 采样、河网 BFS 和流域级模拟逻辑；
- 每个流域内临时保存各次河段 DOF，河段区间直接取 2.5%/97.5% 经验分位数；
- 不再使用 mean ± 1.96 * std，确保区间符合 DOF 的 [0, 100] 取值范围；
- 临时河段模拟矩阵不会写入结果文件，最终文件体量与 v3 同一量级；
- 默认输出 mc_results_v4.pkl，不覆盖 v3 结果。

输出：/root/autodl-tmp/data/result3_finalv2/mc_cache/mc_results_v4.pkl
"""

import argparse
import math
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numba
import numpy as np


# ── 配置 ─────────────────────────────────────────────────────────────────────
CACHE_PATH = '/root/autodl-tmp/data/result3_finalv2/mc_cache/basin_cache.pkl'
RESULTS_PATH = '/root/autodl-tmp/data/result3_finalv2/mc_cache/mc_results_v4.pkl'
N_SIM = 1000
YEARS = [2010, 2015, 2020]
SEED = 42
N_WORKERS = 20
DRF_UPSTREAM = 5.0
DRF_DOWNSTREAM = 5.0
SEG_CI_PERCENTILES = (2.5, 97.5)
SEG_CI_METHOD = 'empirical_percentile'


def parse_years(value):
    return [int(v.strip()) for v in value.split(',') if v.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run v4 MC DOF simulations with empirical segment percentiles.'
    )
    parser.add_argument(
        '--cache-path', default=CACHE_PATH,
        help='Input basin_cache.pkl path.',
    )
    parser.add_argument(
        '--results-path', default=None,
        help='Output mc_results_v4.pkl path. Defaults next to --cache-path.',
    )
    parser.add_argument(
        '--n-sim', type=int, default=N_SIM,
        help='Number of Monte Carlo simulations.',
    )
    parser.add_argument(
        '--years', default=','.join(map(str, YEARS)),
        help='Comma-separated years to simulate, e.g. 2010,2015,2020.',
    )
    parser.add_argument('--seed', type=int, default=SEED, help='Master RNG seed.')
    parser.add_argument(
        '--n-workers', type=int, default=N_WORKERS,
        help='Number of basin worker processes.',
    )
    return parser.parse_args()


# ── Numba JIT BFS ─────────────────────────────────────────────────────────────
@numba.njit(cache=True)
def bfs_dof_numba(dam_noids, disch_array, log10_disch, ndoid_array,
                  up_ptr, up_idx, wfall_array, n_streams,
                  drf_upstream, drf_downstream):
    """计算给定坝样本下所有河段的 DOF；dam_noids 使用 1-based NOID。"""
    dof_array = np.zeros(n_streams, numba.float32)
    scale_up = 100.0 / math.log10(
        drf_upstream if drf_upstream > 1.0 else 1.000000000000001
    )
    scale_down = 100.0 / math.log10(
        drf_downstream if drf_downstream > 1.0 else 1.000000000000001
    )

    visited = np.zeros(n_streams, numba.boolean)
    queue = np.empty(n_streams, numba.int64)
    touched = np.empty(n_streams, numba.int64)

    for dam_i in range(len(dam_noids)):
        dam_noid = numba.int64(dam_noids[dam_i])
        dam_idx = dam_noid - 1
        discharge_barrier = disch_array[dam_idx]

        if discharge_barrier == 0.0:
            dof_array[dam_idx] = numba.float32(100.0)
            continue

        dis_low = discharge_barrier / drf_upstream
        dis_high = discharge_barrier * drf_downstream
        log10_barrier = log10_disch[dam_idx]

        # Upstream BFS
        n_t = numba.int64(0)
        h = numba.int64(0)
        t = numba.int64(0)
        visited[dam_idx] = True
        touched[n_t] = dam_idx
        n_t += 1
        queue[t] = dam_noid
        t += 1

        while h < t:
            node = queue[h]
            h += 1
            ni = node - 1
            if wfall_array[ni] != 0:
                continue
            dl = disch_array[ni]
            if dis_low <= dl <= dis_high:
                a = log10_barrier - log10_disch[ni]
                if a < 0.0:
                    a = 0.0
                score = 100.0 - a * scale_up
                if score < 0.0:
                    score = 0.0
                elif score > 100.0:
                    score = 100.0
                if dof_array[ni] < score:
                    dof_array[ni] = numba.float32(score)
                for j in range(up_ptr[ni], up_ptr[ni + 1]):
                    nb = up_idx[j]
                    nbi = nb - 1
                    if not visited[nbi]:
                        visited[nbi] = True
                        touched[n_t] = nbi
                        n_t += 1
                        queue[t] = nb
                        t += 1

        for i in range(n_t):
            visited[touched[i]] = False

        # Downstream BFS
        n_t = numba.int64(0)
        h = numba.int64(0)
        t = numba.int64(0)
        visited[dam_idx] = True
        touched[n_t] = dam_idx
        n_t += 1
        queue[t] = dam_noid
        t += 1

        while h < t:
            node = queue[h]
            h += 1
            ni = node - 1
            dl = disch_array[ni]
            if dis_low <= dl <= dis_high:
                a = log10_disch[ni] - log10_barrier
                if a < 0.0:
                    a = 0.0
                score = 100.0 - a * scale_down
                if score < 0.0:
                    score = 0.0
                elif score > 100.0:
                    score = 100.0
                if dof_array[ni] < score:
                    dof_array[ni] = numba.float32(score)
                downstream = ndoid_array[ni]
                if downstream > 0:
                    downstream_idx = downstream - 1
                    if not visited[downstream_idx]:
                        visited[downstream_idx] = True
                        touched[n_t] = downstream_idx
                        n_t += 1
                        queue[t] = downstream
                        t += 1

        for i in range(n_t):
            visited[touched[i]] = False

    return dof_array


# ── CSR 构建 ──────────────────────────────────────────────────────────────────
def build_upstream_csr(upstream_lookup, n_streams):
    """将 upstream_lookup dict 转为 CSR 格式。"""
    up_ptr = np.zeros(n_streams + 1, dtype=np.int64)
    for node, neighbors in upstream_lookup.items():
        ni = int(node) - 1
        if 0 <= ni < n_streams:
            up_ptr[ni + 1] = len(neighbors)
    for i in range(1, n_streams + 1):
        up_ptr[i] += up_ptr[i - 1]

    up_idx = np.empty(int(up_ptr[n_streams]), dtype=np.int64)
    pos = up_ptr.copy()
    for node, neighbors in upstream_lookup.items():
        ni = int(node) - 1
        if 0 <= ni < n_streams:
            for nb in neighbors:
                up_idx[pos[ni]] = int(nb)
                pos[ni] += 1
    return up_ptr, up_idx


def empirical_segment_stats(seg_sims):
    """从 (N_SIM, n_streams) 数组计算均值和经验 95% 区间。"""
    mean = seg_sims.mean(axis=0, dtype=np.float64).astype(np.float32)
    ci = np.percentile(
        seg_sims,
        SEG_CI_PERCENTILES,
        axis=0,
        method='linear',
        overwrite_input=True,
    ).astype(np.float32)
    ci_lo, ci_hi = ci[0], ci[1]

    tolerance = 1e-5
    if (
        np.any(ci_lo < -tolerance)
        or np.any(ci_hi > 100.0 + tolerance)
        or np.any(ci_lo > ci_hi)
    ):
        raise ValueError('经验分位数区间超出 DOF 合法范围或上下界顺序错误')

    return {'mean': mean, 'ci_lo': ci_lo, 'ci_hi': ci_hi}


# ── 单流域 MC ─────────────────────────────────────────────────────────────────
def run_basin_mc(args):
    basin_id, cache_entry, base_seed, n_sim, years = args
    rng = np.random.default_rng(base_seed)

    n_streams = cache_entry['n_streams']
    ndoid_array = cache_entry['ndoid_array'].astype(np.int64)
    wfall_array = cache_entry['wfall_array'].astype(np.int64)
    disch_array = cache_entry['disch_array'].astype(np.float64)
    log10_disch = cache_entry['log10_disch'].astype(np.float64)
    up_ptr, up_idx = build_upstream_csr(cache_entry['upstream_lookup'], n_streams)

    basin_sims = {year: np.zeros(n_sim, dtype=np.float32) for year in years}
    # 仅在 worker 内临时存在；返回前转换成 mean/ci_lo/ci_hi 后释放。
    seg_sims = {
        year: np.zeros((n_sim, n_streams), dtype=np.float32)
        for year in years
    }

    for sim_i in range(n_sim):
        for year in years:
            dam_data = cache_entry['dams'].get(year)
            if dam_data is None or len(dam_data['noids']) == 0:
                continue

            certain_mask = dam_data['certain_mask']
            noids = dam_data['noids']
            confidence = dam_data['confidence']
            uncertain_idx = np.where(~certain_mask)[0]

            if len(uncertain_idx) > 0:
                keep = rng.random(len(uncertain_idx)) < confidence[uncertain_idx]
                kept_uncertain = noids[uncertain_idx[keep]]
            else:
                kept_uncertain = np.array([], dtype=np.int64)

            kept_noids = np.concatenate(
                [noids[certain_mask], kept_uncertain]
            ).astype(np.int64)
            if len(kept_noids) == 0:
                continue

            dof_arr = bfs_dof_numba(
                kept_noids,
                disch_array,
                log10_disch,
                ndoid_array,
                up_ptr,
                up_idx,
                wfall_array,
                n_streams,
                DRF_UPSTREAM,
                DRF_DOWNSTREAM,
            )

            basin_sims[year][sim_i] = float(dof_arr.mean())
            seg_sims[year][sim_i] = dof_arr

    seg_stats = {
        year: empirical_segment_stats(seg_sims[year])
        for year in years
    }
    return basin_id, basin_sims, seg_stats


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    global CACHE_PATH, RESULTS_PATH, N_SIM, YEARS, SEED, N_WORKERS

    args = parse_args()
    CACHE_PATH = args.cache_path
    RESULTS_PATH = args.results_path or os.path.join(
        os.path.dirname(CACHE_PATH), 'mc_results_v4.pkl'
    )
    N_SIM = args.n_sim
    YEARS = parse_years(args.years)
    SEED = args.seed
    N_WORKERS = args.n_workers

    if N_SIM <= 0:
        raise ValueError('--n-sim 必须为正整数')
    if not YEARS:
        raise ValueError('--years 不能为空')
    if N_WORKERS <= 0:
        raise ValueError('--n-workers 必须为正整数')

    t_start = time.time()
    print(f'加载缓存: {CACHE_PATH}')
    with open(CACHE_PATH, 'rb') as file_obj:
        basin_cache = pickle.load(file_obj)
    print(
        f'共 {len(basin_cache)} 个流域，N_SIM={N_SIM}, workers={N_WORKERS}, '
        f'河段区间={SEG_CI_PERCENTILES[0]}%/{SEG_CI_PERCENTILES[1]}% 经验分位数'
    )

    print('Numba 预热编译中...')
    first_entry = basin_cache[sorted(basin_cache.keys())[0]]
    up0, ui0 = build_upstream_csr(
        first_entry['upstream_lookup'], first_entry['n_streams']
    )
    bfs_dof_numba(
        np.array([1], dtype=np.int64),
        first_entry['disch_array'].astype(np.float64),
        first_entry['log10_disch'].astype(np.float64),
        first_entry['ndoid_array'].astype(np.int64),
        up0,
        ui0,
        first_entry['wfall_array'].astype(np.int64),
        first_entry['n_streams'],
        DRF_UPSTREAM,
        DRF_DOWNSTREAM,
    )
    print('Numba 编译完成\n')

    basin_ids = sorted(basin_cache.keys())
    rng_master = np.random.default_rng(SEED)
    seeds = rng_master.integers(0, 2**31, size=len(basin_ids))
    args_list = [
        (basin_id, basin_cache[basin_id], int(seeds[i]), N_SIM, tuple(YEARS))
        for i, basin_id in enumerate(basin_ids)
    ]

    all_basin_sims = {}
    all_seg_stats = {}
    errors = []
    done = 0

    with ProcessPoolExecutor(
        max_workers=min(N_WORKERS, max(1, len(basin_ids)))
    ) as executor:
        futures = {
            executor.submit(run_basin_mc, worker_args): worker_args[0]
            for worker_args in args_list
        }
        for future in as_completed(futures):
            basin_id = futures[future]
            try:
                bid, basin_sims, seg_stats = future.result()
                all_basin_sims[bid] = basin_sims
                all_seg_stats[bid] = seg_stats
                done += 1
                if done % 50 == 0 or done == len(basin_ids):
                    elapsed = time.time() - t_start
                    print(f'  [{done}/{len(basin_ids)}] 完成 {bid}  已用 {elapsed:.0f}s')
            except Exception as exc:
                import traceback
                errors.append((basin_id, str(exc)))
                print(f'  {basin_id} 错误: {exc}\n{traceback.format_exc()}')

    if errors:
        failed = ', '.join(basin_id for basin_id, _ in errors)
        raise RuntimeError(
            f'有 {len(errors)} 个流域失败，未写入不完整结果: {failed}'
        )

    elapsed = time.time() - t_start
    print(
        f'\nMC 模拟完成，成功 {len(all_basin_sims)} 个流域，'
        f'总耗时 {elapsed:.1f}s ({elapsed / 60:.1f}分钟)'
    )

    results = {
        'basin_sims': all_basin_sims,
        'seg_stats': all_seg_stats,
        'N_SIM': N_SIM,
        'YEARS': YEARS,
        'VERSION': 4,
        'SEG_CI_METHOD': SEG_CI_METHOD,
        'SEG_CI_PERCENTILES': SEG_CI_PERCENTILES,
        'SEED': SEED,
    }

    results_dirname = os.path.dirname(RESULTS_PATH)
    if results_dirname:
        os.makedirs(results_dirname, exist_ok=True)
    print(f'保存结果到 {RESULTS_PATH} ...')
    with open(RESULTS_PATH, 'wb') as file_obj:
        pickle.dump(results, file_obj, protocol=4)
    print('完成。')


if __name__ == '__main__':
    main()
