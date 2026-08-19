"""
exp8_spatial_mc_fast_nogdpc_repl2010.py

Exp8 加速版「去 GDP per Capita + replacement 2010」单文件变体。

这个脚本参考 exp8_spatial_mc_fast_nogdpc.py 的写法，把 replacement
workflow 原先拆到 attr_slm_latest 里的配置、数据清洗、预计算、VIF、
OLS/SLM、Effects、MC 聚合都收回到本文件中。它仍然复用 common_mc.py
里的基础工具、HLZ 读取、MC 缓存读取和采样函数，和原始 Exp8 脚本保持一致。

默认模型:
  ATTR_EXP=final9_repl2010_hfpurb

默认数据:
  ATTR_LEV=lev06
  HLZ_DATA_DIR=/root/autodl-tmp/code/result1/output_finalv2/hlz/hlz_v8_replacement2010_finalv2
  HLZ_SHP_TEMPLATE=BasinATLAS_v10_{lev}_with_stats_hlz_v8_repl2010_finalv2.shp

空间权重:
  流域质心 Haversine KNN, k=5, row-standardized

示例:
  /root/miniconda3/envs/geovis/bin/python \
    /root/autodl-tmp/code/result2/exp8_spatial_mc_fast_nogdpc_repl2010.py

切换实验:
  ATTR_EXP=final10_repl2010_hfpurb_pop N_JOBS=24 \
    /root/miniconda3/envs/geovis/bin/python \
    /root/autodl-tmp/code/result2/exp8_spatial_mc_fast_nogdpc_repl2010.py
"""

import contextlib
import io
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from esda.moran import Moran
from joblib import Parallel, delayed
from libpysal.weights import W, lag_spatial
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

# 让 common_mc 在 import 时直接读取 v8 replacement shapefile。
RESULT1_RUN_TAG = os.environ.get('RESULT1_RUN_TAG', 'finalv2').strip()
RESULT1_SUFFIX = f'_{RESULT1_RUN_TAG}' if RESULT1_RUN_TAG else ''
RESULT1_OUTPUT_ROOT = os.environ.get(
    'RESULT1_OUTPUT_ROOT', '/root/autodl-tmp/code/result1/output_finalv2'
).strip()
RESULT2_RUN_TAG = os.environ.get('RESULT2_RUN_TAG', 'finalv2').strip()
RUN_SUFFIX = f'_{RESULT2_RUN_TAG}' if RESULT2_RUN_TAG else ''
RESULT2_OUTPUT_ROOT = os.environ.get(
    'RESULT2_OUTPUT_ROOT', '/root/autodl-tmp/code/result2/output_finalv2'
).strip()

os.environ.setdefault('CONDA_PREFIX', '/root/miniconda3/envs/geovis')
os.environ.setdefault('PROJ_LIB', '/root/miniconda3/envs/geovis/share/proj')
os.environ.setdefault('PROJ_DATA', '/root/miniconda3/envs/geovis/share/proj')
os.environ.setdefault(
    'HLZ_DATA_DIR',
    (
        os.path.join(
            RESULT1_OUTPUT_ROOT,
            'hlz',
            f'hlz_v8_replacement2010{RESULT1_SUFFIX}',
        )
        if RESULT1_OUTPUT_ROOT
        else '/root/autodl-tmp/code/result1/output_finalv2/hlz/hlz_v8_replacement2010_finalv2'
    ),
)
os.environ.setdefault(
    'HLZ_SHP_TEMPLATE',
    f'BasinATLAS_v10_{{lev}}_with_stats_hlz_v8_repl2010{RESULT1_SUFFIX}.shp',
)
os.environ.setdefault(
    'MC_CACHE_DIR',
    (
        os.path.join(RESULT1_OUTPUT_ROOT, f'mc_results{RESULT1_SUFFIX}')
        if RESULT1_OUTPUT_ROOT
        else '/root/autodl-tmp/code/result1/output_finalv2/mc_results_finalv2'
    ),
)
os.environ.setdefault(
    'MC_CACHE_TEMPLATE',
    f'precomputed_{{lev}}_{{year}}{RESULT1_SUFFIX}.pkl',
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from result2.common_mc import (  # noqa: E402
    N_SIM as N_SIM_DEFAULT,
    SEED,
    EXP8_DAM_COLS,
    EXP8_DAM_LABELS,
    sig_stars,
    to_num,
    make_subsets,
    load_gdf as _load_gdf_base,
    load_cache,
    precompute_recall,
    mc_sample_update,
)

try:
    from spreg import ML_Lag
    HAS_SPREG = True
except ImportError:
    HAS_SPREG = False
    print('[警告] spreg 不可用，SLM 将使用 OLS+Wy 近似（结果有偏）')


# ══════════════════════════════════════════════════════════════════════════════
# 实验配置
# ══════════════════════════════════════════════════════════════════════════════
LABELS = {
    # HydroATLAS / BasinATLAS 原始属性与派生属性
    'dis_m3_pyr': 'Discharge',
    'dis_m3_log': 'log(Discharge)',
    'run_mm_syr': 'Runoff',
    'run_mm_log': 'log(Runoff)',
    'gwt_cm_sav': 'Groundwater Depth',
    'inu_pc_slt': 'Inundation %',
    'lka_pc_sse': 'Lake %',
    'lka_pc_log': 'log(Lake %)',
    'ele_mt_sav': 'Elevation',
    'slp_dg_sav': 'Slope',
    'sgr_dk_sav': 'Stream Gradient',
    'ari_ix_sav': 'Aridity Index',
    'pre_mm_syr': 'Precipitation',
    'pet_mm_syr': 'PET',
    'tmp_dc_syr': 'Temperature',
    'cmi_ix_syr': 'Climate Moisture',
    'snw_pc_syr': 'Snow Cover %',
    'ire_pc_sse': 'Irrigated %',
    'crp_pc_sse': 'Cropland %',
    'pst_pc_sse': 'Pasture %',
    'for_pc_sse': 'Forest %',
    'gla_pc_sse': 'Glacier %',
    'prm_pc_sse': 'Permafrost %',
    'pac_pc_sse': 'Protected %',
    'urb_pc_sse': 'Urban %',
    'kar_pc_sse': 'Karst %',
    'swc_pc_syr': 'Soil Water',
    'ero_kh_sav': 'Erosion',
    'ero_kh_log': 'log(Erosion)',
    'cly_pc_sav': 'Clay %',
    'snd_pc_sav': 'Sand %',
    'rdd_mk_sav': 'Road Density',
    'rdd_mk_log': 'log(Road Density)',
    'hft_ix_u09': 'Human Footprint 09',
    'hft_ix_u93': 'Human Footprint 93',
    'hft_chg': 'Delta Human Footprint',
    'hdi_ix_sav': 'HDI',
    'nli_ix_sav': 'Night Lights',
    'nli_ix_log': 'log(Night Lights)',
    'ppd_pk_sav': 'Population Density',
    'ppd_pk_log': 'log(Pop Density)',
    'gdp_ud_sav': 'GDP/area',
    'gdp_ud_log': 'log(GDP/area)',

    # v8 replacement 2010 新增字段
    'hfp10': 'Human Footprint 2010',
    'urb10_ghs': 'Urban 2010 %',
    'glo10_mn': 'GloFAS Mean Discharge 2010',
    'glo10_mx': 'GloFAS Max Discharge 2010',
    'glo10_mn_log': 'log(GloFAS Mean Discharge 2010)',
    'glo10_mx_log': 'log(GloFAS Max Discharge 2010)',
    'gsw10_pc': 'Surface Water 2010 %',
    'gsw10_perm': 'Permanent Water 2010 %',
    'gsw10_sea': 'Seasonal Water 2010 %',


    # 2010 社会经济变量
    'watot_2010': 'Total Water Use',
    'gdpt_10': 'GDP Total',
    'gdpp_10': 'GDP per Capita',
    'elec_10': 'Electricity Consumption',
    'spamI_10': 'Agri Irrigated',
    'pop_2010': 'Population',
}

EXPERIMENTS = {
    'final9_repl2010': {
        'x_cols': [
            'watot_2010', 'gdpt_10', 'elec_10', 'spamI_10',
            'glo10_mn_log', 'inu_pc_slt', 'hfp10', 'ari_ix_sav',
            'urb10_ghs',
        ],
        'desc': (
            '2010 replacement model: socio4 + GloFAS2010 + inundation '
            '+ HFP2010 + AI + GHSL urban2010'
        ),
    },
    'final9_repl2010_hfpurb': {
        'x_cols': [
            'watot_2010', 'gdpt_10', 'elec_10', 'spamI_10',
            'dis_m3_log', 'inu_pc_slt', 'hfp10', 'ari_ix_sav',
            'urb10_ghs',
        ],
        'desc': (
            '2010 replacement model: socio4 + HydroATLAS discharge '
            '+ inundation + HFP2010 + AI + GHSL urban2010'
        ),
    },
    'final9_repl2010_hfpurb_gsw': {
        'x_cols': [
            'watot_2010', 'gdpt_10', 'elec_10', 'spamI_10',
            'dis_m3_log', 'gsw10_pc', 'hfp10', 'ari_ix_sav',
            'urb10_ghs',
        ],
        'desc': (
            '2010 replacement model: socio4 + HydroATLAS discharge '
            '+ JRC GSW2010 water % + HFP2010 + AI + GHSL urban2010'
        ),
    },
    'final10_repl2010_hfpurb_pop': {
        'x_cols': [
            'watot_2010', 'gdpt_10', 'elec_10', 'spamI_10', 'pop_2010',
            'dis_m3_log', 'inu_pc_slt', 'hfp10', 'ari_ix_sav',
            'urb10_ghs',
        ],
        'desc': (
            '2010 replacement model + Population 2010: socio4 + POP2010 '
            '+ HydroATLAS discharge + inundation + HFP2010 + AI '
            '+ GHSL urban2010'
        ),
    },
}

ATTR_EXP = os.environ.get('ATTR_EXP', 'final9_repl2010_hfpurb')
if ATTR_EXP not in EXPERIMENTS:
    raise SystemExit(
        f'未知 ATTR_EXP={ATTR_EXP!r}；可用实验: {sorted(EXPERIMENTS)}'
    )

X_COLS_2010 = list(EXPERIMENTS[ATTR_EXP]['x_cols'])
X_LABELS_2010 = {c: LABELS.get(c, c) for c in X_COLS_2010}
EXP_DESC = EXPERIMENTS[ATTR_EXP]['desc']

LEV = os.environ.get('ATTR_LEV', 'lev06')
N_SIM = int(os.environ.get('N_SIM_OVERRIDE', os.environ.get('N_SIM', N_SIM_DEFAULT)))
N_JOBS = int(os.environ.get('N_JOBS', '20'))
MIN_SAMPLES = 10


def _env_flag(name, default='1'):
    return os.environ.get(name, default).strip().lower() not in {
        '0', 'false', 'no', 'off', ''
    }


# 两项优化都不改变模型或随机抽样；环境变量用于等价性和性能基准。
OPT_STATIC_MORAN = _env_flag('OPT_STATIC_MORAN', '1')
OPT_CACHED_IMPACTS = _env_flag('OPT_CACHED_IMPACTS', '1')

RESULTS_BASE_DIR = os.environ.get(
    'OUTPUT_BASE_DIR',
    RESULT2_OUTPUT_ROOT or '/root/autodl-tmp/code/result2/output_finalv2',
)
DEFAULT_OUTPUT_DIR = os.path.join(
    RESULTS_BASE_DIR,
    f'{ATTR_EXP}{RUN_SUFFIX}',
)
if LEV != 'lev06':
    DEFAULT_OUTPUT_DIR = os.path.join(
        RESULTS_BASE_DIR,
        LEV,
        f'{ATTR_EXP}{RUN_SUFFIX}',
    )
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', DEFAULT_OUTPUT_DIR)


def tagged_filename(filename):
    stem, ext = os.path.splitext(filename)
    return f'{stem}{RUN_SUFFIX}{ext}'

ALL_DAM_COLS = EXP8_DAM_COLS
ALL_DAM_LABELS = EXP8_DAM_LABELS
MC_DYNAMIC_COLS = {'dDam_e', 'dDam_total'}
MC_STATIC_COLS = [dc for dc in ALL_DAM_COLS if dc not in MC_DYNAMIC_COLS]


def log_name(col):
    """与旧 attr pipeline 一致：取原列名前 6 字符后加 _log。"""
    return f'{col[:6]}_log'


SENTINEL_COLS = [
    'slp_dg_sav', 'sgr_dk_sav', 'snw_pc_syr', 'snw_pc_smx',
    'cly_pc_sav', 'slt_pc_sav', 'snd_pc_sav', 'soc_th_sav',
    'wet_cl_smj', 'ele_mt_smn',
]

LOG_COLS = [
    'dis_m3_pyr', 'run_mm_syr', 'pop_ct_ssu', 'ppd_pk_sav',
    'gdp_ud_sav', 'nli_ix_sav', 'rdd_mk_sav', 'ero_kh_sav',
    'lka_pc_sse', 'lkv_mc_usu',
]

EXTRA_LOG_COLS = {
    'glo10_mn': 'glo10_mn_log',
    'glo10_mx': 'glo10_mx_log',
}


def get_experiment(name):
    cfg = EXPERIMENTS[name]
    missing_labels = [c for c in cfg['x_cols'] if c not in LABELS]
    if missing_labels:
        raise KeyError(f'实验 {name} 的列缺 label: {missing_labels}')
    return list(cfg['x_cols']), {c: LABELS[c] for c in cfg['x_cols']}, cfg['desc']


def load_gdf(lev):
    """
    读取 v8 replacement shapefile，并在本文件内完成属性回归所需的清洗和派生列。
    """
    g = _load_gdf_base(lev)

    for c in SENTINEL_COLS:
        if c in g.columns:
            g[c] = to_num(g[c]).replace(-999, np.nan)

    for c in LOG_COLS:
        if c in g.columns:
            v = to_num(g[c])
            g[log_name(c)] = np.log1p(v.clip(lower=0))

    for c, out_c in EXTRA_LOG_COLS.items():
        if c in g.columns:
            v = to_num(g[c])
            g[out_c] = np.log1p(v.clip(lower=0))

    if 'hft_ix_u09' in g.columns and 'hft_ix_u93' in g.columns:
        g['hft_chg'] = to_num(g['hft_ix_u09']) - to_num(g['hft_ix_u93'])

    return g


def validate_x_columns(gdf):
    missing = [c for c in X_COLS_2010 if c not in gdf.columns]
    if missing:
        raise KeyError(f'当前 shapefile 缺少回归变量: {missing}')


if os.environ.get('ATTR_QUIET') != '1':
    print(f'[Exp8 replacement standalone] ATTR_EXP={ATTR_EXP} | {EXP_DESC}')
    print(f'  lev={LEV}')
    print(f'  X_COLS={X_COLS_2010}')
    print(f'  N_SIM={N_SIM} N_JOBS={N_JOBS}')
    print(
        f'  OPT_STATIC_MORAN={int(OPT_STATIC_MORAN)} '
        f'OPT_CACHED_IMPACTS={int(OPT_CACHED_IMPACTS)}'
    )
    print('  SPATIAL_WEIGHTS=Haversine centroid KNN (k=5, row-standardized)')
    print(f'  HLZ_DATA_DIR={os.environ["HLZ_DATA_DIR"]}')
    print(f'  HLZ_SHP_TEMPLATE={os.environ["HLZ_SHP_TEMPLATE"]}')
    print(f'  MC_CACHE_DIR={os.environ["MC_CACHE_DIR"]}')
    print(f'  MC_CACHE_TEMPLATE={os.environ["MC_CACHE_TEMPLATE"]}')
    print(f'  OUTPUT_DIR={OUTPUT_DIR}')


# ══════════════════════════════════════════════════════════════════════════════
# FDR
# ══════════════════════════════════════════════════════════════════════════════
def apply_fdr_by_region(df, p_col='p_value', alpha=0.05):
    df = df.copy()
    df['p_fdr'] = np.nan
    for region in df['region'].unique():
        mask = df['region'] == region
        p_vals = df.loc[mask, p_col].values
        valid = ~np.isnan(p_vals)
        if valid.sum() > 0:
            _, p_adj, _, _ = multipletests(p_vals[valid], method='fdr_bh')
            df.loc[df.index[mask][valid], 'p_fdr'] = p_adj
    df['sig_fdr'] = df['p_fdr'].apply(sig_stars)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 全球球面 KNN 权重
# ══════════════════════════════════════════════════════════════════════════════
def make_haversine_knn(gdf, k=5):
    """
    使用流域质心之间的 Haversine（大圆）距离构建行标准化 KNN 权重。

    BallTree 要求坐标顺序为 [latitude, longitude] 且单位为弧度。权重的
    id_order 严格沿用 gdf.index，保证 W、X 和 y 的行顺序一致。
    """
    n = len(gdf)
    if n < 2:
        raise ValueError('Haversine KNN 至少需要 2 个有效几何')
    if not gdf.index.is_unique:
        raise ValueError('Haversine KNN 要求 GeoDataFrame index 唯一')

    k_use = min(int(k), n - 1)
    centroids = gdf.geometry.centroid
    lat = centroids.y.to_numpy(dtype=float)
    lon = centroids.x.to_numpy(dtype=float)
    coords_rad = np.radians(np.column_stack([lat, lon]))
    if not np.isfinite(coords_rad).all():
        raise ValueError('Haversine KNN 检测到非有限质心坐标')

    tree = BallTree(coords_rad, metric='haversine')
    candidates = tree.query(
        coords_rad,
        k=k_use + 1,
        return_distance=False,
    )

    ids = list(gdf.index)
    neighbors = {}
    for row_pos, candidate_pos in enumerate(candidates):
        # 显式排除自身；对质心坐标重复的流域也能稳定保留 k 个其他邻居。
        neighbor_pos = [int(pos) for pos in candidate_pos if int(pos) != row_pos]
        if len(neighbor_pos) < k_use:
            raise RuntimeError(
                f'Haversine KNN 邻居不足: row={row_pos}, '
                f'expected={k_use}, actual={len(neighbor_pos)}'
            )
        neighbors[ids[row_pos]] = [ids[pos] for pos in neighbor_pos[:k_use]]

    weights = W(
        neighbors,
        id_order=ids,
        silence_warnings=True,
    )
    weights.transform = 'r'
    return weights


# ══════════════════════════════════════════════════════════════════════════════
# 预计算：权重 + X_scaled + W_dense + Moran 权重 + exist_x
# ══════════════════════════════════════════════════════════════════════════════
def precompute_all(gdf_base):
    subsets = make_subsets(gdf_base)
    weights_base = {}
    weights_dam = {}
    weights_moran = {}

    for region, df_sub in subsets.items():
        df_geo = df_sub[df_sub.geometry.notna() & (~df_sub.geometry.is_empty)].copy()
        if len(df_geo) < MIN_SAMPLES:
            continue
        n = len(df_geo)

        w_base = make_haversine_knn(df_geo, k=min(5, n - 1))
        weights_base[region] = (df_geo.index, w_base)

        weights_moran[region] = {}
        moran_vars = list(ALL_DAM_COLS) + [
            c for c in X_COLS_2010 if c in df_geo.columns
        ]
        for var_col in moran_vars:
            if var_col not in df_geo.columns:
                continue
            y = to_num(df_geo[var_col])
            vmask = y.notna().values
            n_valid = vmask.sum()
            if n_valid < MIN_SAMPLES:
                continue
            if n_valid < n:
                iloc_v = np.where(vmask)[0]
                w_var = make_haversine_knn(
                    df_geo.iloc[iloc_v],
                    k=min(5, n_valid - 1),
                )
            else:
                iloc_v = np.arange(n)
                w_var = w_base
            weights_moran[region][var_col] = (iloc_v, w_var)

        exist_x = [c for c in X_COLS_2010 if c in df_geo.columns]
        weights_dam[region] = {'__exist_x__': exist_x}
        for dc in ALL_DAM_COLS:
            if dc not in df_geo.columns:
                continue
            data_test = df_geo[exist_x + [dc]].apply(
                pd.to_numeric, errors='coerce'
            ).dropna()
            idx = data_test.index
            if len(idx) < MIN_SAMPLES:
                continue
            try:
                w_sub = make_haversine_knn(
                    df_geo.loc[idx],
                    k=min(5, len(idx) - 1),
                )
            except Exception:
                continue

            X_scaled = StandardScaler().fit_transform(data_test[exist_x].values)
            X_const = sm.add_constant(X_scaled)
            y_base = data_test[dc].values
            W_dense = w_sub.sparse.toarray() if len(idx) <= 3000 else None

            weights_dam[region][dc] = (
                idx, w_sub, X_scaled, X_const, W_dense, y_base, exist_x
            )

    n_moran = sum(len(v) for v in weights_moran.values())
    n_slm = sum(
        len([k for k in v if k != '__exist_x__']) for v in weights_dam.values()
    )
    print(
        f'  [预计算] {len(weights_base)} 纬度带 | '
        f'Moran权重 {n_moran} 对 | SLM权重 {n_slm} 对'
    )
    return weights_base, weights_dam, weights_moran


def compute_vif(weights_dam):
    rows = []
    for region, dam_weights in weights_dam.items():
        exist_x = dam_weights.get('__exist_x__', [])
        if len(exist_x) < 2:
            continue
        ref = dam_weights.get('dDam_total') or next(
            (v for k, v in dam_weights.items() if k != '__exist_x__'), None
        )
        if ref is None:
            continue
        X_scaled = ref[2]
        for j, c in enumerate(exist_x):
            try:
                v = variance_inflation_factor(X_scaled, j)
            except Exception:
                v = np.nan
            rows.append({
                'region': region,
                'variable': X_LABELS_2010.get(c, c),
                'var_col': c,
                'VIF': float(v),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# OLS / SLM / Effects
# ══════════════════════════════════════════════════════════════════════════════
def _sparse_impact_trace_cache(W, max_power=30):
    """预计算 trace(W^k)；截断规则与原逐次 Neumann 计算完全一致。"""
    W = W.tocsr()
    n = W.shape[0]
    traces = []
    wk = W.copy()
    for k in range(1, max_power + 1):
        traces.append(float(wk.diagonal().sum()))
        if k < max_power:
            wk = (wk @ W).tocsr()
            if wk.nnz > 60 * n:
                break
    return {'n': n, 'traces': tuple(traces)}


def precompute_impact_trace_cache(weights_dam):
    """仅为 MC 动态坝型预计算稀疏空间效应 trace，并复用相同样本的 W。"""
    cache = {}
    by_id_order = {}
    for region, dam_weights in weights_dam.items():
        for dc in MC_DYNAMIC_COLS:
            if dc not in dam_weights:
                continue
            _, w_sub, _, _, W_dense, _, _ = dam_weights[dc]
            if W_dense is not None:
                continue
            signature = tuple(w_sub.id_order)
            if signature not in by_id_order:
                by_id_order[signature] = _sparse_impact_trace_cache(w_sub.sparse)
            cache[(region, dc)] = by_id_order[signature]
    return cache, len(by_id_order)


def _spatial_multiplier_effects(
    betas_arr, rho, W, x_labels, z_stats=None, trace_cache=None
):
    """
    W 可为稠密 ndarray 或 scipy 稀疏矩阵。
    大 n 时用稀疏 Neumann trace 近似 direct effect，total multiplier 在行标准化
    权重下为 1/(1-rho)。
    """
    import scipy.sparse as _sp

    if _sp.issparse(W):
        n = W.shape[0]
        total_mult = 1.0 / (1.0 - rho)
        trace_s = float(n)
        k_max = min(30, max(5, int(np.ceil(np.log(1e-4) / np.log(abs(rho) + 1e-9)))))
        rho_k = rho
        if trace_cache is not None:
            if trace_cache['n'] != n:
                raise ValueError('impact trace cache 与 W 维度不一致')
            for tr_wk in trace_cache['traces'][:k_max]:
                trace_s += rho_k * tr_wk
                rho_k *= rho
        else:
            wk = W.tocsr().copy()
            for k in range(1, k_max + 1):
                trace_s += rho_k * float(wk.diagonal().sum())
                rho_k *= rho
                if k < k_max:
                    wk = (wk @ W).tocsr()
                    if wk.nnz > 60 * n:
                        break
        direct_mult = trace_s / n
    else:
        n = W.shape[0]
        s_mat = np.linalg.inv(np.eye(n) - rho * W)
        direct_mult = np.trace(s_mat) / n
        total_mult = s_mat.sum() / n
    indirect_mult = total_mult - direct_mult

    rows = []
    for i, (bk, lbl) in enumerate(zip(betas_arr, x_labels)):
        row = {
            'variable': lbl,
            'beta': float(bk),
            'direct_effect': float(bk) * direct_mult,
            'indirect_effect': float(bk) * indirect_mult,
            'total_effect': float(bk) * total_mult,
        }
        if z_stats is not None and i < len(z_stats):
            row['z_value'] = float(z_stats[i][0])
            row['p_value'] = float(z_stats[i][1])
        rows.append(row)
    return rows


def _run_slm_row(region, dl, dc, idx, w_sub, X_scaled, X_const, W_dense, y_val,
                 exist_x, impact_trace=None):
    if len(y_val) < len(exist_x) + 10:
        return None, []
    try:
        ols_m = sm.OLS(y_val, X_const).fit()
    except Exception:
        return None, []

    try:
        mi_resid = Moran(ols_m.resid, w_sub, permutations=0)
        resid_mi = mi_resid.I
        resid_mi_p = mi_resid.p_norm
    except Exception:
        resid_mi = resid_mi_p = np.nan

    row = {
        'region': region,
        'dam_type': dl,
        'OLS_R2': ols_m.rsquared,
        'OLS_adj_R2': ols_m.rsquared_adj,
        'OLS_AIC': ols_m.aic,
        'resid_Morans_I': resid_mi,
        'resid_Morans_p': resid_mi_p,
        'n': len(y_val),
    }

    effects_rows = []
    if HAS_SPREG:
        try:
            ml_method = 'LU' if len(y_val) > 3000 else 'full'
            with contextlib.redirect_stdout(io.StringIO()):
                slm_fit = ML_Lag(
                    y_val.reshape(-1, 1), X_scaled, w=w_sub, method=ml_method
                )
            rho_val = float(slm_fit.rho)
            row['SLM_R2'] = float(slm_fit.pr2)
            row['SLM_rho'] = rho_val
            row['SLM_rho_p'] = float(slm_fit.z_stat[-1][1])
            row['R2_improvement'] = float(slm_fit.pr2) - ols_m.rsquared
            try:
                W = W_dense if W_dense is not None else w_sub.sparse
                betas_x = slm_fit.betas[1:].flatten()
                z_stats_x = slm_fit.z_stat[1:-1]
                x_labels = [X_LABELS_2010.get(c, c) for c in exist_x]
                for eff_r in _spatial_multiplier_effects(
                    betas_x, rho_val, W, x_labels, z_stats_x,
                    trace_cache=impact_trace,
                ):
                    eff_r.update({'region': region, 'dam_type': dl})
                    effects_rows.append(eff_r)
            except Exception:
                pass
        except Exception:
            row.update({
                'SLM_R2': np.nan,
                'SLM_rho': np.nan,
                'SLM_rho_p': np.nan,
                'R2_improvement': np.nan,
            })
    else:
        try:
            wy = lag_spatial(w_sub, y_val)
            slm_m = sm.OLS(y_val, np.column_stack([X_const, wy])).fit()
            row['SLM_R2'] = slm_m.rsquared
            row['SLM_rho'] = slm_m.params[-1]
            row['SLM_rho_p'] = slm_m.pvalues[-1]
            row['R2_improvement'] = slm_m.rsquared - ols_m.rsquared
            row['SLM_note'] = 'OLS_fallback_biased'
        except Exception:
            row.update({
                'SLM_R2': np.nan,
                'SLM_rho': np.nan,
                'SLM_rho_p': np.nan,
                'R2_improvement': np.nan,
            })
    return row, effects_rows


def _slm_static(weights_dam, impact_trace_cache=None):
    slm_rows = []
    effects_rows = []
    for region, dam_weights in weights_dam.items():
        for dc, dl in zip(ALL_DAM_COLS, ALL_DAM_LABELS):
            if dc in MC_DYNAMIC_COLS or dc not in dam_weights:
                continue
            idx, w_sub, X_scaled, X_const, W_dense, y_base, exist_x = dam_weights[dc]
            row, effs = _run_slm_row(
                region, dl, dc, idx, w_sub, X_scaled, X_const, W_dense,
                y_base, exist_x,
                impact_trace=(impact_trace_cache or {}).get((region, dc)),
            )
            if row is not None:
                slm_rows.append(row)
                effects_rows.extend(effs)
    return slm_rows, effects_rows


def _moran_once_fast(gdf, weights_base, weights_moran, include_cols=None):
    test_vars = {dc: dl for dc, dl in zip(ALL_DAM_COLS, ALL_DAM_LABELS)}
    for ic in X_COLS_2010:
        if ic in gdf.columns:
            test_vars[ic] = X_LABELS_2010.get(ic, ic)
    if include_cols is not None:
        include_cols = set(include_cols)
        test_vars = {c: label for c, label in test_vars.items() if c in include_cols}

    rows = []
    for region, (idx_base, _) in weights_base.items():
        if region not in weights_moran:
            continue
        df_sub = gdf.loc[idx_base]
        region_wmoran = weights_moran[region]
        for var_col, var_label in test_vars.items():
            if var_col not in region_wmoran:
                continue
            iloc_v, w_sub = region_wmoran[var_col]
            y_clean = to_num(df_sub.iloc[iloc_v][var_col]).values
            finite_mask = ~np.isnan(y_clean)
            if finite_mask.sum() < MIN_SAMPLES:
                continue
            y_use = y_clean[finite_mask]
            w_use = w_sub
            if finite_mask.sum() < len(y_clean):
                sub_gdf = df_sub.iloc[iloc_v[finite_mask]]
                try:
                    w_use = make_haversine_knn(
                        sub_gdf, k=min(5, len(sub_gdf) - 1)
                    )
                except Exception:
                    continue
            try:
                mi = Moran(y_use, w_use, permutations=0)
                rows.append({
                    'region': region,
                    'var_col': var_col,
                    'variable': var_label,
                    'Morans_I': mi.I,
                    'E_I': mi.EI,
                    'z_score': mi.z_norm,
                    'p_value': mi.p_norm,
                    'n': len(y_use),
                })
            except Exception:
                continue
    return rows


def _slm_once_dynamic(gdf, weights_dam, impact_trace_cache=None):
    slm_rows = []
    effects_rows = []
    for region, dam_weights in weights_dam.items():
        for dc, dl in zip(ALL_DAM_COLS, ALL_DAM_LABELS):
            if dc not in MC_DYNAMIC_COLS or dc not in dam_weights:
                continue
            idx, w_sub, X_scaled, X_const, W_dense, _, exist_x = dam_weights[dc]
            df_sub = gdf.loc[idx]
            y_val = to_num(df_sub[dc]).values
            valid_mask = ~np.isnan(y_val)
            if valid_mask.sum() < len(exist_x) + 10:
                continue
            if valid_mask.sum() != len(y_val):
                y_v = y_val[valid_mask]
                x_v = X_scaled[valid_mask]
                xc_v = sm.add_constant(x_v)
                try:
                    w_v = make_haversine_knn(
                        df_sub.iloc[np.where(valid_mask)[0]],
                        k=min(5, valid_mask.sum() - 1),
                    )
                except Exception:
                    continue
                row, effs = _run_slm_row(
                    region, dl, dc, idx[valid_mask], w_v, x_v, xc_v, None,
                    y_v, exist_x, impact_trace=None,
                )
            else:
                row, effs = _run_slm_row(
                    region, dl, dc, idx, w_sub, X_scaled, X_const, W_dense,
                    y_val, exist_x,
                    impact_trace=(impact_trace_cache or {}).get((region, dc)),
                )
            if row is not None:
                slm_rows.append(row)
                effects_rows.extend(effs)
    return slm_rows, effects_rows


# ══════════════════════════════════════════════════════════════════════════════
# 聚合与 MC batch
# ══════════════════════════════════════════════════════════════════════════════
def _aggregate(records_list, key_cols, val_cols):
    collector = defaultdict(lambda: {c: [] for c in val_cols})
    for records in records_list:
        for r in records:
            key = tuple(r.get(k) for k in key_cols)
            for c in val_cols:
                v = r.get(c, np.nan)
                collector[key][c].append(float(v) if v is not None else np.nan)

    rows = []
    for key, vals in collector.items():
        row = dict(zip(key_cols, key))
        for c in val_cols:
            arr = np.array(vals[c], dtype=float)
            arr = arr[~np.isnan(arr)]
            row[f'{c}_mean'] = np.mean(arr) if len(arr) else np.nan
            row[f'{c}_ci_lo'] = np.percentile(arr, 2.5) if len(arr) else np.nan
            row[f'{c}_ci_hi'] = np.percentile(arr, 97.5) if len(arr) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _run_batch(sim_indices, gdf_base, cache, recall,
               weights_base, weights_dam, weights_moran,
               dynamic_moran_only=False, impact_trace_cache=None):
    for var in (
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
    ):
        os.environ[var] = '1'

    results = []
    for sim_i in sim_indices:
        rng = np.random.default_rng(SEED + sim_i)
        gdf_sim = mc_sample_update(gdf_base.copy(), cache, rng, recall)
        moran_rows = _moran_once_fast(
            gdf_sim, weights_base, weights_moran,
            include_cols=MC_DYNAMIC_COLS if dynamic_moran_only else None,
        )
        slm_rows, effects_rows = _slm_once_dynamic(
            gdf_sim, weights_dam, impact_trace_cache=impact_trace_cache
        )
        results.append((moran_rows, slm_rows, effects_rows))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    import time as _time

    t0 = _time.perf_counter()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'加载 shapefile ({LEV}) ...')
    gdf_base = load_gdf(LEV)
    validate_x_columns(gdf_base)
    gdf_base = gdf_base[
        gdf_base.geometry.notna() & (~gdf_base.geometry.is_empty)
    ].copy()
    print(f'  有效几何: {len(gdf_base)}')

    print('加载 pkl 缓存 ...')
    cache = load_cache(LEV)
    recall = precompute_recall(cache, gdf_base['HYBAS_ID'].values)

    print('预计算所有固定权重 ...')
    weights_base, weights_dam, weights_moran = precompute_all(gdf_base)

    static_moran_rows = []
    if OPT_STATIC_MORAN:
        static_moran_cols = set(MC_STATIC_COLS) | set(X_COLS_2010)
        static_moran_rows = _moran_once_fast(
            gdf_base, weights_base, weights_moran,
            include_cols=static_moran_cols,
        )
        print(f'  静态 Moran 预计算: {len(static_moran_rows)} 行')

    impact_trace_cache = {}
    if OPT_CACHED_IMPACTS:
        import time as _trace_time
        trace_t0 = _trace_time.perf_counter()
        impact_trace_cache, n_unique_trace = precompute_impact_trace_cache(weights_dam)
        trace_dt = _trace_time.perf_counter() - trace_t0
        max_terms = max(
            (len(v['traces']) for v in impact_trace_cache.values()), default=0
        )
        print(
            f'  Effects trace 缓存: {len(impact_trace_cache)} 模型 / '
            f'{n_unique_trace} 个唯一 W, 最多 {max_terms} 阶, {trace_dt:.2f}s'
        )

    vif_df = compute_vif(weights_dam)
    if not vif_df.empty:
        vif_df.to_csv(
            os.path.join(OUTPUT_DIR, tagged_filename('vif.csv')),
            index=False,
            encoding='utf-8-sig',
        )
        print(f"  VIF: {len(vif_df)} 行 → {tagged_filename('vif.csv')}")

    print('静态坝型 (g/b/a) OLS + ML_Lag ...')
    static_slm_rows, static_effects_rows = _slm_static(
        weights_dam, impact_trace_cache=impact_trace_cache
    )
    print(
        f'  静态 SLM: {len(static_slm_rows)} 行, '
        f'Effects: {len(static_effects_rows)} 行'
    )

    indices = list(range(N_SIM))
    batch_size = max(1, -(-N_SIM // max(1, N_JOBS)))
    batches = [indices[i:i + batch_size] for i in range(0, N_SIM, batch_size)]
    print(f'\nMC 模拟 ({N_SIM} 次, N_JOBS={N_JOBS}, batch_size={batch_size}) ...')
    batch_results = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(_run_batch)(
            batch, gdf_base, cache, recall,
            weights_base, weights_dam, weights_moran,
            dynamic_moran_only=OPT_STATIC_MORAN,
            impact_trace_cache=impact_trace_cache,
        )
        for batch in batches
    )

    moran_sims = []
    slm_sims = []
    effects_sims = []
    for batch in batch_results:
        for moran_rows, slm_rows, effects_rows in batch:
            moran_sims.append(moran_rows + static_moran_rows)
            slm_sims.append(slm_rows + static_slm_rows)
            effects_sims.append(effects_rows + static_effects_rows)

    print('\n聚合 MC 结果 ...')
    moran_df = _aggregate(
        moran_sims,
        key_cols=['region', 'var_col', 'variable'],
        val_cols=['Morans_I', 'E_I', 'z_score', 'p_value', 'n'],
    )
    if not moran_df.empty:
        moran_df = apply_fdr_by_region(moran_df, p_col='p_value_mean')
        moran_df['sig'] = moran_df['p_value_mean'].apply(sig_stars)
        moran_df['interpretation'] = moran_df.apply(
            lambda r: (
                '正空间自相关'
                if r['Morans_I_mean'] > 0 and r['p_value_mean'] < 0.05
                else (
                    '负空间自相关'
                    if r['Morans_I_mean'] < 0 and r['p_value_mean'] < 0.05
                    else '无显著空间自相关'
                )
            ),
            axis=1,
        )
        moran_df.to_csv(
            os.path.join(OUTPUT_DIR, tagged_filename('moran_mc_mean.csv')),
            index=False,
            encoding='utf-8-sig',
        )
        print(
            f"  Moran I: {len(moran_df)} 行 → "
            f"{tagged_filename('moran_mc_mean.csv')}"
        )

    slm_df = _aggregate(
        slm_sims,
        key_cols=['region', 'dam_type'],
        val_cols=[
            'OLS_R2', 'OLS_adj_R2', 'OLS_AIC',
            'resid_Morans_I', 'resid_Morans_p',
            'SLM_R2', 'SLM_rho', 'SLM_rho_p',
            'R2_improvement', 'n',
        ],
    )
    if not slm_df.empty:
        slm_df['resid_spatial_autocorr'] = slm_df['resid_Morans_p_mean'].apply(
            lambda p: ('是' if p < 0.05 else '否') if not np.isnan(p) else '未知'
        )
        slm_df['SLM_rho_sig'] = slm_df['SLM_rho_p_mean'].apply(sig_stars)
        slm_df.to_csv(
            os.path.join(OUTPUT_DIR, tagged_filename('slm_mc_mean.csv')),
            index=False,
            encoding='utf-8-sig',
        )
        print(
            f"  OLS/SLM: {len(slm_df)} 行 → "
            f"{tagged_filename('slm_mc_mean.csv')}"
        )

    effects_df = _aggregate(
        effects_sims,
        key_cols=['region', 'dam_type', 'variable'],
        val_cols=[
            'beta', 'direct_effect', 'indirect_effect', 'total_effect',
            'z_value', 'p_value',
        ],
    )
    if not effects_df.empty:
        effects_df = apply_fdr_by_region(effects_df, p_col='p_value_mean')
        effects_df['sig'] = effects_df['p_value_mean'].apply(sig_stars)
        effects_df.to_csv(
            os.path.join(OUTPUT_DIR, tagged_filename('effects_mc_mean.csv')),
            index=False,
            encoding='utf-8-sig',
        )
        print(
            f"  Effects: {len(effects_df)} 行 → "
            f"{tagged_filename('effects_mc_mean.csv')}"
        )

    dt = _time.perf_counter() - t0
    print(f'\n完成 [{ATTR_EXP}]，耗时 {dt:.1f}s ({dt/60:.1f} 分钟)')
    print(f'输出目录: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
