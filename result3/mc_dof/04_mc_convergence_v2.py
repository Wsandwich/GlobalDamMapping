"""
mc_convergence_v2.py — MC DOF 收敛性分析（改版）

改进：用 Running Standard Error（SE = std/sqrt(k)）替代逐步相对变化
  - SE 随模拟次数增加总体趋于下降，但不要求逐步严格单调
  - 四张子图：全局CI收窄 / 全局SE对数曲线 / 高emb流域散点 / 分层SE对比
  - Nature-style 双栏图件，输出 400-dpi PNG
"""

import math
import argparse
import os
import pickle
import time
import numpy as np
import numba
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

matplotlib.rcParams['font.family'] = 'Nimbus Sans'
plt.rcParams.update({
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 8,
    'axes.titlesize': 8.5,
    'axes.labelsize': 8,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'legend.fontsize': 6.2,
    'legend.frameon': False,
    'axes.unicode_minus': False,
})

COLORS = {
    'blue_main': '#0F4D92',
    'blue_mid': '#3775BA',
    'blue_soft': '#B4C0E4',
    'teal': '#42949E',
    'red': '#B64342',
    'violet': '#9A4D8E',
    'neutral': '#767676',
    'neutral_light': '#D8D8D8',
}
FORMAL_N_SIM = 1000
THRESHOLD_LABEL_X = 400

# ── 配置 ─────────────────────────────────────────────────────────────────────
CACHE_PATH    = '/root/autodl-tmp/data/result3_finalv2/mc_cache/basin_cache.pkl'
OUT_PATH      = '/root/autodl-tmp/data/result3_finalv2/figures/diagnostics/mc_dof_convergence_v2.png'
N_MAX         = 2000
YEAR          = 2020
SEED          = 42
N_WORKERS     = 20
DRF_UPSTREAM  = 5.0
DRF_DOWNSTREAM= 5.0
USE_ALL_BASINS = True   # True → 使用全部流域；False → 分层抽样
N_PER_STRATUM = 5
STRATA = [
    (1,    10,   'Very low (1–9)'),
    (10,   100,  'Low (10–99)'),
    (100,  1000, 'Moderate (100–999)'),
    (1000, 10000,'High (1,000–9,999)'),
    (10000,9e9,  'Very high (≥10,000)'),
]


def parse_args():
    parser = argparse.ArgumentParser(description='MC DOF convergence analysis.')
    parser.add_argument('--cache-path', default=CACHE_PATH)
    parser.add_argument('--out-path', default=OUT_PATH)
    parser.add_argument('--n-max', type=int, default=N_MAX)
    parser.add_argument('--year', type=int, default=YEAR)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--n-workers', type=int, default=N_WORKERS)
    return parser.parse_args()

# ── Numba JIT BFS ─────────────────────────────────────────────────────────────

@numba.njit(cache=True)
def bfs_dof_numba(dam_noids, disch_array, log10_disch, ndoid_array,
                  up_ptr, up_idx, wfall_array, n_streams,
                  drf_upstream, drf_downstream):
    dof_array  = np.zeros(n_streams, numba.float32)
    scale_up   = 100.0 / math.log10(drf_upstream   if drf_upstream   > 1.0 else 1.000000000000001)
    scale_down = 100.0 / math.log10(drf_downstream if drf_downstream > 1.0 else 1.000000000000001)
    visited = np.zeros(n_streams, numba.boolean)
    queue   = np.empty(n_streams, numba.int64)
    touched = np.empty(n_streams, numba.int64)
    for di in range(len(dam_noids)):
        dam_noid = numba.int64(dam_noids[di])
        dam_idx  = dam_noid - 1
        discharge_barrier = disch_array[dam_idx]
        if discharge_barrier == 0.0:
            dof_array[dam_idx] = numba.float32(100.0)
            continue
        dis_low       = discharge_barrier / drf_upstream
        dis_high      = discharge_barrier * drf_downstream
        log10_barrier = log10_disch[dam_idx]
        n_t = numba.int64(0); h = numba.int64(0); t = numba.int64(0)
        visited[dam_idx] = True
        touched[n_t] = dam_idx; n_t += 1
        queue[t] = dam_noid;    t   += 1
        while h < t:
            node = queue[h]; h += 1; ni = node - 1
            if wfall_array[ni] != 0:
                continue
            dl = disch_array[ni]
            if dis_low <= dl <= dis_high:
                a = log10_barrier - log10_disch[ni]
                if a < 0.0: a = 0.0
                s = 100.0 - a * scale_up
                if s < 0.0: s = 0.0
                elif s > 100.0: s = 100.0
                if dof_array[ni] < s:
                    dof_array[ni] = numba.float32(s)
                for j in range(up_ptr[ni], up_ptr[ni + 1]):
                    nb = up_idx[j]; nbi = nb - 1
                    if not visited[nbi]:
                        visited[nbi] = True
                        touched[n_t] = nbi; n_t += 1
                        queue[t]     = nb;  t   += 1
        for i in range(n_t): visited[touched[i]] = False
        n_t = numba.int64(0); h = numba.int64(0); t = numba.int64(0)
        visited[dam_idx] = True
        touched[n_t] = dam_idx; n_t += 1
        queue[t] = dam_noid;    t   += 1
        while h < t:
            node = queue[h]; h += 1; ni = node - 1
            dl = disch_array[ni]
            if dis_low <= dl <= dis_high:
                a = log10_disch[ni] - log10_barrier
                if a < 0.0: a = 0.0
                s = 100.0 - a * scale_down
                if s < 0.0: s = 0.0
                elif s > 100.0: s = 100.0
                if dof_array[ni] < s:
                    dof_array[ni] = numba.float32(s)
                do = ndoid_array[ni]
                if do > 0:
                    di2 = do - 1
                    if not visited[di2]:
                        visited[di2] = True
                        touched[n_t] = di2; n_t += 1
                        queue[t]     = do;  t   += 1
        for i in range(n_t): visited[touched[i]] = False
    return dof_array


def build_upstream_csr(upstream_lookup, n_streams):
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


# ── 单流域模拟 ────────────────────────────────────────────────────────────────

def run_basin_convergence(args):
    basin_id, c, base_seed, n_sim, year = args
    rng = np.random.default_rng(base_seed)
    n_streams   = c['n_streams']
    ndoid_array = c['ndoid_array'].astype(np.int64)
    wfall_array = c['wfall_array'].astype(np.int64)
    disch_array = c['disch_array'].astype(np.float64)
    log10_disch = c['log10_disch'].astype(np.float64)
    up_ptr, up_idx = build_upstream_csr(c['upstream_lookup'], n_streams)
    dam_data = c['dams'].get(year)

    sim_means = np.zeros(n_sim, dtype=np.float64)
    if dam_data is None or len(dam_data['noids']) == 0:
        return basin_id, sim_means

    certain_mask  = dam_data['certain_mask']
    noids         = dam_data['noids']
    confidence    = dam_data['confidence']
    uncertain_idx = np.where(~certain_mask)[0]

    for sim_i in range(n_sim):
        if len(uncertain_idx) > 0:
            keep = rng.random(len(uncertain_idx)) < confidence[uncertain_idx]
            kept_uncertain = noids[uncertain_idx[keep]]
        else:
            kept_uncertain = np.array([], dtype=np.int64)
        kept_noids = np.concatenate([noids[certain_mask], kept_uncertain]).astype(np.int64)
        if len(kept_noids) == 0:
            sim_means[sim_i] = 0.0
            continue
        dof_arr = bfs_dof_numba(
            kept_noids, disch_array, log10_disch,
            ndoid_array, up_ptr, up_idx, wfall_array, n_streams,
            DRF_UPSTREAM, DRF_DOWNSTREAM
        )
        sim_means[sim_i] = float(dof_arr.mean())

    return basin_id, sim_means


# ── 分层抽样 ──────────────────────────────────────────────────────────────────

def select_basins(cache, rng_sel):
    emb_count = {}
    for bid, c in cache.items():
        d = c['dams'].get(YEAR)
        emb_count[bid] = int((~d['certain_mask']).sum()) if d is not None else 0

    selected = {}
    for lo, hi, label in STRATA:
        candidates = [b for b, cnt in emb_count.items() if lo <= cnt < hi]
        n = min(N_PER_STRATUM, len(candidates))
        if n == 0:
            selected[label] = []
            continue
        chosen = list(rng_sel.choice(candidates, size=n, replace=False))
        selected[label] = chosen
        print(f"  {label}: 候选{len(candidates)}个，选{n}个")
    return selected, emb_count


# ── 收敛指标计算（Running SE） ────────────────────────────────────────────────

def calc_se_metrics(sim_means_dict):
    """
    对每个流域计算 running SE = running_std / sqrt(k)
    全局：各模拟次的跨流域均值 → running mean / running SE
    """
    bids   = list(sim_means_dict.keys())
    matrix = np.stack([sim_means_dict[b] for b in bids], axis=0)  # (n_basins, N_MAX)
    ks     = np.arange(1, matrix.shape[1] + 1)

    # 全局（跨流域均值的 running mean / running SE）
    global_per_sim = matrix.mean(axis=0)
    running_mean   = np.cumsum(global_per_sim) / ks
    running_var    = np.array([
        global_per_sim[:k].var() if k > 1 else 0.0
        for k in range(1, len(global_per_sim) + 1)
    ])
    running_se     = np.sqrt(running_var) / np.sqrt(ks)  # SE of the mean

    # per-basin running SE
    per_basin_se = {}
    per_basin_mean = {}
    for i, bid in enumerate(bids):
        sims = matrix[i]
        rm   = np.cumsum(sims) / ks
        rv   = np.array([sims[:k].var() if k > 1 else 0.0 for k in range(1, len(sims)+1)])
        per_basin_se[bid]   = np.sqrt(rv) / np.sqrt(ks)
        per_basin_mean[bid] = rm

    return running_mean, running_se, per_basin_mean, per_basin_se, bids, matrix


# ── 绘图 ──────────────────────────────────────────────────────────────────────

def add_formal_n_reference(ax, n_max):
    if n_max < FORMAL_N_SIM:
        return
    ax.axvline(
        FORMAL_N_SIM, color=COLORS['neutral'], lw=0.8,
        ls=(0, (3, 2)), alpha=0.8, zorder=0,
    )


def save_png(fig, out_path):
    root, ext = os.path.splitext(out_path)
    png_path = out_path if ext.lower() == '.png' else f'{root}.png'
    fig.savefig(
        png_path, dpi=400, bbox_inches='tight', facecolor='white',
    )
    return png_path

def plot_convergence_v2(running_mean, running_se, per_basin_mean, per_basin_se,
                        bids, matrix, selected_basins, emb_count, out_path):

    ks = np.arange(1, len(running_mean) + 1)
    n_max = len(ks)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    # 为上下两行的横轴标签与标题预留更清晰的呼吸空间。
    fig.set_constrained_layout_pads(h_pad=0.08, hspace=0.12)

    # a, 全局累计均值及其 MC 误差：主要证据
    ax = axes[0, 0]
    ci = 1.96 * running_se
    ax.plot(
        ks, running_mean, color=COLORS['blue_main'], lw=1.35,
        label='Running mean', zorder=3,
    )
    ax.fill_between(ks, running_mean - ci, running_mean + ci,
                    alpha=0.35, color=COLORS['blue_soft'], linewidth=0,
                    label='95% CI (±1.96 SE)')
    add_formal_n_reference(ax, n_max)
    ax.scatter(
        [ks[-1]], [running_mean[-1]], s=18, color=COLORS['blue_main'],
        edgecolor='white', linewidth=0.5, zorder=4,
    )
    ax.annotate(
        f'{running_mean[-1]:.4f}', (ks[-1], running_mean[-1]),
        xytext=(-5, 6), textcoords='offset points', ha='right',
        fontsize=6.2, color=COLORS['blue_main'],
    )
    ax.set_xlabel('Simulations, N')
    ax.set_ylabel('Mean DOF')
    ax.set_title(f'Global mean DOF ({YEAR})')
    ax.legend(loc='best', handlelength=2.0)

    # b, 全局累计标准误：阈值只作为相对参考
    ax = axes[0, 1]
    se_safe = np.where(running_se > 0, running_se, np.nan)
    ax.semilogy(
        ks, se_safe, color=COLORS['blue_main'], lw=1.25,
        label='Global running SE',
    )

    # 标注收敛参考线：SE 下降到初始值的 10% / 5% / 1%
    se_init = np.nanmax(se_safe[:min(20, len(se_safe))])
    if se_init > 0:
        for frac, color, ls in [
            (0.10, COLORS['teal'], (0, (3, 2))),
            (0.05, COLORS['red'], (0, (5, 2))),
            (0.01, COLORS['violet'], (0, (1, 2))),
        ]:
            thr = se_init * frac
            ax.axhline(
                thr, color=color, ls=ls, lw=0.8,
            )
            ax.text(
                min(THRESHOLD_LABEL_X, 0.60 * n_max),
                thr, f'{frac:.0%} of initial SE',
                ha='left', va='bottom',
                fontsize=5.8, color=color,
            )
    add_formal_n_reference(ax, n_max)
    if np.isfinite(se_safe[-1]):
        ax.scatter(
            [ks[-1]], [se_safe[-1]], s=18, color=COLORS['blue_main'],
            edgecolor='white', linewidth=0.5, zorder=4,
        )
        ax.annotate(
            f'{se_safe[-1]:.2e}', (ks[-1], se_safe[-1]),
            xytext=(-5, 6), textcoords='offset points', ha='right',
            fontsize=6.2, color=COLORS['blue_main'],
        )
    ax.set_xlabel('Simulations, N')
    ax.set_ylabel('Running SE')
    ax.set_title('Global Monte Carlo error')

    # c, 不确定土石坝最多的 5 个流域：最不利情况检查
    ax = axes[1, 0]
    top_bids = sorted(bids, key=lambda b: emb_count.get(b, 0), reverse=True)[:5]
    colors5 = ['#0F4D92', '#3775BA', '#42949E', '#7884B4', '#9A4D8E']
    bid_to_row = {bid: i for i, bid in enumerate(bids)}
    for i, bid in enumerate(top_bids):
        sims = matrix[bid_to_row[bid]]
        rm   = per_basin_mean[bid]
        c_   = colors5[i]
        n_emb = emb_count.get(bid, 0)
        ax.scatter(
            ks, sims, s=1.2, alpha=0.12, color=c_, linewidths=0,
            rasterized=True,
        )
        ax.plot(
            ks, rm, '-', lw=1.05, color=c_,
            label=f'{bid}: {n_emb / 1000:.0f}k',
        )
    add_formal_n_reference(ax, n_max)
    ax.set_xlabel('Simulations, N')
    ax.set_ylabel('Mean DOF')
    ax.set_title('Most uncertain basins')
    ax.legend(
        title='HYBAS_ID: uncertain dams', title_fontsize=5.8,
        loc='lower left', bbox_to_anchor=(0.0, 0.08),
        handlelength=1.6, labelspacing=0.2,
    )

    # d, 分层中位数和四分位带：替代不可读的全流域“意大利面图”
    ax = axes[1, 1]
    stratum_colors = ['#B4C0E4', '#7884B4', '#42949E', '#9A4D8E', '#B64342']
    for color, (label, layer_bids) in zip(stratum_colors, selected_basins.items()):
        arrays = [per_basin_se[bid] for bid in layer_bids if bid in per_basin_se]
        if not arrays:
            continue
        layer = np.stack(arrays, axis=0)
        layer = np.where(layer > 0, layer, np.nan)
        q25 = np.full(n_max, np.nan)
        median = np.full(n_max, np.nan)
        q75 = np.full(n_max, np.nan)
        q25[1:], median[1:], q75[1:] = np.nanpercentile(
            layer[:, 1:], [25, 50, 75], axis=0,
        )
        ax.fill_between(
            ks, q25, q75, color=color, alpha=0.16, linewidth=0,
        )
        ax.semilogy(
            ks, median, color=color, lw=1.15,
            label=f'{label}; n={len(arrays):,}',
        )
    add_formal_n_reference(ax, n_max)
    ax.set_xlabel('Simulations, N')
    ax.set_ylabel('Running SE')
    ax.set_title('Basin-level SE by uncertainty')
    ax.legend(loc='best', handlelength=1.7, labelspacing=0.2)

    for ax in axes.flat:
        ax.margins(x=0.02)
        ax.tick_params(direction='out')

    png_path = save_png(fig, out_path)
    plt.close(fig)
    print(f"图表已保存: {png_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

sim_means_dict_ref = {}  # 供绘图函数访问

def main():
    global CACHE_PATH, OUT_PATH, N_MAX, YEAR, SEED, N_WORKERS

    args = parse_args()
    CACHE_PATH = args.cache_path
    OUT_PATH = args.out_path
    N_MAX = args.n_max
    YEAR = args.year
    SEED = args.seed
    N_WORKERS = args.n_workers

    if N_MAX <= 0:
        raise ValueError('--n-max 必须为正整数')
    if N_WORKERS <= 0:
        raise ValueError('--n-workers 必须为正整数')

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print("加载缓存...")
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)

    print("Numba 预热...")
    c0 = cache[sorted(cache.keys())[0]]
    up0, ui0 = build_upstream_csr(c0['upstream_lookup'], c0['n_streams'])
    bfs_dof_numba(np.array([1], dtype=np.int64),
                  c0['disch_array'].astype(np.float64), c0['log10_disch'].astype(np.float64),
                  c0['ndoid_array'].astype(np.int64), up0, ui0,
                  c0['wfall_array'].astype(np.int64), c0['n_streams'],
                  DRF_UPSTREAM, DRF_DOWNSTREAM)
    print("Numba 编译完成\n")

    # ── 流域选择 ──────────────────────────────────────────────────────────────
    emb_count = {}
    for bid, c in cache.items():
        d = c['dams'].get(YEAR)
        emb_count[bid] = int((~d['certain_mask']).sum()) if d is not None else 0

    if USE_ALL_BASINS:
        print("使用全部流域...")
        all_bids = sorted(cache.keys())
        # 将全部流域按分层归类（用于绘图着色）
        selected_basins = {label: [] for _, _, label in STRATA}
        for bid in all_bids:
            cnt = emb_count[bid]
            for lo, hi, label in STRATA:
                if lo <= cnt < hi:
                    selected_basins[label].append(bid)
                    break
        print(f"共 {len(all_bids)} 个流域，每个运行 {N_MAX} 次模拟\n")
    else:
        rng_sel = np.random.default_rng(0)
        print("分层抽样流域:")
        selected_basins, emb_count = select_basins(cache, rng_sel)
        all_bids = [b for bids in selected_basins.values() for b in bids]
        print(f"\n共选 {len(all_bids)} 个流域，每个运行 {N_MAX} 次模拟\n")

    rng_master = np.random.default_rng(SEED)
    seeds = rng_master.integers(0, 2**31, size=len(all_bids))
    args_list = [(bid, cache[bid], int(seeds[i]), N_MAX, YEAR)
                 for i, bid in enumerate(all_bids)]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(all_bids))) as executor:
        futures = {executor.submit(run_basin_convergence, args): args[0]
                   for args in args_list}
        for fut in as_completed(futures):
            bid, sim_means = fut.result()
            sim_means_dict_ref[bid] = sim_means
            print(f"  完成: {bid}  (emb={emb_count.get(bid,0):,})")

    print(f"\n模拟完成，耗时 {time.time()-t0:.1f}s")

    print("\n计算收敛指标（Running SE）...")
    running_mean, running_se, per_basin_mean, per_basin_se, bids, matrix = \
        calc_se_metrics(sim_means_dict_ref)

    # 打印全局 SE 下降情况
    se_safe = np.where(running_se > 0, running_se, np.nan)
    se_init = np.nanmax(se_safe[:20])
    print(f"\n全局 Running Mean: {running_mean[0]:.4f} → {running_mean[-1]:.4f}")
    print(f"全局 Running SE:   {se_safe[0]:.6f} → {se_safe[-1]:.6f}")
    print(f"\nSE 收敛情况（相对初始最大 SE={se_init:.6f}）:")
    for frac in [0.50, 0.20, 0.10, 0.05, 0.01]:
        thr = se_init * frac
        idx = np.where(se_safe <= thr)[0]
        if len(idx) > 0:
            print(f"  SE ≤ {frac:.0%} 初始值（{thr:.6f}）: 第 {idx[0]+1:3d} 次")
        else:
            print(f"  SE ≤ {frac:.0%} 初始值（{thr:.6f}）: {N_MAX} 次内未达到")

    print("\n生成图表...")
    plot_convergence_v2(running_mean, running_se, per_basin_mean, per_basin_se,
                        bids, matrix, selected_basins, emb_count, OUT_PATH)


if __name__ == '__main__':
    main()
