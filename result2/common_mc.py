"""
common_mc.py — MC 误差分析共用模块

包含：配置常量、工具函数、数据加载、MC 采样函数
供各实验文件导入。
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from scipy import stats
from libpysal.weights import Queen, KNN, lag_spatial
from esda.moran import Moran
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════
N_SIM  = 1000
SEED   = 42
LEVELS = ['lev06']

RECALL_BY_Z    = {13: 0.8671, 14: 0.8754, 15: 0.8551, 16: 0.9011}
RECALL_DEFAULT = 0.9011

RESULT1_RUN_TAG = os.environ.get('RESULT1_RUN_TAG', 'finalv2').strip()
RESULT1_SUFFIX = f'_{RESULT1_RUN_TAG}' if RESULT1_RUN_TAG else ''
RESULT1_OUTPUT_ROOT = os.environ.get(
    'RESULT1_OUTPUT_ROOT', '/root/autodl-tmp/code/result1/output_finalv2'
).strip()

DATA_DIR = os.environ.get(
    'HLZ_DATA_DIR',
    os.path.join(RESULT1_OUTPUT_ROOT, 'hlz', f'hlz_v7{RESULT1_SUFFIX}'),
)
SHP_TEMPLATE = os.environ.get(
    'HLZ_SHP_TEMPLATE',
    f'BasinATLAS_v10_{{lev}}_with_stats_hlz_v7{RESULT1_SUFFIX}.shp',
)
CACHE_DIR = os.environ.get(
    'MC_CACHE_DIR',
    (
        os.path.join(RESULT1_OUTPUT_ROOT, f'mc_results{RESULT1_SUFFIX}')
        if RESULT1_OUTPUT_ROOT
        else '/root/autodl-tmp/code/result1/output_finalv2/mc_results_finalv2'
    ),
)
CACHE_TEMPLATE = os.environ.get(
    'MC_CACHE_TEMPLATE',
    f'precomputed_{{lev}}_{{year}}{RESULT1_SUFFIX}.pkl',
)
OUT_DIR = os.environ.get(
    'CORR_MC_OUTPUT_BASE_DIR',
    '/root/autodl-tmp/code/result2/output_finalv2',
)
os.makedirs(OUT_DIR, exist_ok=True)

EMBANKMENT_DAM_TYPE = 'embankment_dam'
_EMBANKMENT_ONLY_KEY = '_embankment_only'

# MC 影响的 delta-dam 列
MC_DAM_COLS   = ['dDam_e', 'dDam_total']
MC_DAM_LABELS = ['Embank Density', 'Total Density']

# 指标定义
LAG_IND_MAP = {
    'watot_2010': 'Total Water Use 2010',
    'gdpt_10':    'GDP Total 2010',
    'gdpp_10':    'GDP per Capita 2010',
    'elec_10':    'Electricity Consumption 2010',
    'spamI_10':   'Agri Area Irrigated 2010',
    'pop_2010':   'Population 2010',
}

DELTA_MAP = {
    'dGDPt':  ('gdpt_20',    'gdpt_10'),
    'dGDPp':  ('gdpp_20',    'gdpp_10'),
    'dPop':   ('pop_2020',   'pop_2010'),
    'dElec':  ('elec_19',    'elec_10'),
    'dWatot': ('watot_2020', 'watot_2010'),
    'dSpamI': ('spamI_20',   'spamI_10'),
}
DELTA_LABELS = {
    'dGDPt': 'ΔGDP Total', 'dGDPp': 'ΔGDP per Capita',
    'dPop': 'ΔPopulation', 'dElec': 'ΔElectricity',
    'dWatot': 'ΔTotal Water Use', 'dSpamI': 'ΔAgri Irrigated',
}

X_COLS_2010 = ['watot_2010', 'gdpt_10', 'gdpp_10', 'elec_10', 'spamI_10', 'pop_2010']
X_LABELS_2010 = {
    'watot_2010': 'Total Water Use',  'gdpt_10': 'GDP Total',
    'gdpp_10':    'GDP per Capita',   'elec_10': 'Electricity Consumption',
    'spamI_10':   'Agri Irrigated',   'pop_2010': 'Population',
}

# Exp7 所有坝型
DAM_TYPES_EXP7 = [
    ('total',    'dam_total',    'dam_total_lag'),
    ('Barrage',  'dam_b',        'dam_b_lag'),
    ('Gravity',  'dam_g',        'dam_g_lag'),
    ('Arch',     'dam_a',        'dam_a_lag'),
    ('Embank',   'dam_e',        'dam_e_lag'),
]

# Exp8 坝型
EXP8_DAM_COLS_CNT   = ['dDam_e', 'dDam_g', 'dDam_b', 'dDam_a', 'dDam_total']
EXP8_DAM_LABELS_CNT = ['Embank Density', 'Gravity Density', 'Barrage Density',
                        'Arch Density', 'Total Density']
EXP8_DAM_COLS   = EXP8_DAM_COLS_CNT
EXP8_DAM_LABELS = EXP8_DAM_LABELS_CNT


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════
def sig_stars(p):
    if pd.isna(p): return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def to_num(s):
    return pd.to_numeric(s, errors='coerce')


def make_subsets(df):
    return {
        'Global':        df,
        'Low Latitude':  df[df['lat_catego'] == 'Low Latitude'],
        'Mid Latitude':  df[df['lat_catego'] == 'Mid Latitude'],
        'High Latitude': df[df['lat_catego'] == 'High Latitude'],
    }


def within_transform(df, group_col, cols):
    df = df.copy()
    gm = df.groupby(group_col)[cols].transform('mean')
    for c in cols:
        df[f'{c}_w'] = df[c] - gm[c]
    return df


def add_latitude_classification(gdf):
    if 'lat_catego' not in gdf.columns:
        try:
            centroid_lat = gdf.geometry.centroid.y.abs()
        except Exception:
            centroid_lat = pd.Series(np.zeros(len(gdf)), index=gdf.index)
        gdf = gdf.copy()
        gdf['lat_catego'] = pd.cut(
            centroid_lat, bins=[-1, 30, 60, 91],
            labels=['Low Latitude', 'Mid Latitude', 'High Latitude'],
            right=True,
        ).astype(str)
    return gdf


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════════════
def load_gdf(lev):
    """加载 v6 shapefile，添加纬度分类、Δ 指标和 2015 插值字段（Exp7 需要）"""
    shp = os.path.join(DATA_DIR, SHP_TEMPLATE.format(lev=lev))
    gdf = gpd.read_file(shp)
    gdf = add_latitude_classification(gdf)

    for dt in ['e', 'g', 'b', 'a']:
        gdf[f'dDam_{dt}'] = to_num(gdf[f'2020_d_{dt}']) - to_num(gdf[f'2010_d_{dt}'])
    gdf['dDam_total'] = sum(to_num(gdf[f'dDam_{dt}']) for dt in ['e', 'g', 'b', 'a'])

    # Δ 社会经济指标
    for new_col, (c1, c0) in DELTA_MAP.items():
        gdf[new_col] = to_num(gdf[c1]) - to_num(gdf[c0])

    # 2015 年社会经济指标插值（固定，不受 MC 影响，Exp7 使用）
    fields_2010 = {'gdpt': 'gdpt_10', 'gdpp': 'gdpp_10', 'pop': 'pop_2010',
                   'elec': 'elec_10', 'watot': 'watot_2010'}
    fields_2020 = {'gdpt': 'gdpt_20', 'gdpp': 'gdpp_20', 'pop': 'pop_2020',
                   'elec': 'elec_19', 'watot': 'watot_2020'}
    fields_2015 = {'gdpt': 'gdpt_15', 'gdpp': 'gdpp_15', 'pop': 'pop_2015',
                   'elec': 'elec_15', 'watot': 'watot_2015'}
    for key in fields_2010:
        f15 = fields_2015[key]
        if f15 not in gdf.columns:
            gdf[f15] = (to_num(gdf[fields_2010[key]]) + to_num(gdf[fields_2020[key]])) / 2
    gdf['spamI_15'] = (to_num(gdf['spamI_10']) + to_num(gdf['spamI_20'])) / 2

    return gdf


def _embankment_arrays(cache_entry, context='cache entry'):
    """从单个缓存条目中提取土石坝的 precision 和 z_level。"""
    prec = np.asarray(cache_entry.get('prec', []), dtype=np.float32)
    z_arr = np.asarray(cache_entry.get('z_level', []), dtype=np.int16)

    if len(z_arr) not in (0, len(prec)):
        raise ValueError(
            f'{context}: z_level 长度 {len(z_arr)} 与 prec 长度 {len(prec)} 不一致'
        )

    # load_cache() 返回的紧凑缓存已经只包含土石坝，避免每次 MC 重复过滤。
    if cache_entry.get(_EMBANKMENT_ONLY_KEY, False):
        return prec, z_arr

    dam_types = cache_entry.get('dam_type')
    if dam_types is None:
        raise ValueError(
            f'{context}: 缺少 dam_type，无法安全地区分土石坝与其他坝型'
        )
    dam_types = np.asarray(dam_types, dtype=object)
    if len(dam_types) != len(prec):
        raise ValueError(
            f'{context}: dam_type 长度 {len(dam_types)} 与 prec 长度 {len(prec)} 不一致'
        )

    is_embankment = dam_types == EMBANKMENT_DAM_TYPE
    emb_z = z_arr[is_embankment] if len(z_arr) else z_arr
    return prec[is_embankment], emb_z


def _compact_embankment_cache(raw_cache, year):
    """将磁盘中的全坝型缓存转换为仅供土石坝 MC 使用的紧凑内存缓存。"""
    compact = {}
    n_embankment = 0
    n_embankment_basins = 0
    for hid, entry in raw_cache.items():
        hid_int = int(hid)
        prec, z_arr = _embankment_arrays(
            entry, context=f'year={year}, HYBAS_ID={hid_int}'
        )
        if len(prec):
            n_embankment_basins += 1
            n_embankment += len(prec)
        compact[hid_int] = {
            'prec': prec,
            'z_level': z_arr,
            _EMBANKMENT_ONLY_KEY: True,
        }
    return compact, n_embankment_basins, n_embankment


def load_cache(lev):
    """加载三年缓存，并在内存中严格过滤为土石坝记录（2015 用于 Exp7）。"""
    cache = {}
    for year in [2010, 2015, 2020]:
        pkl_path = os.path.join(
            CACHE_DIR, CACHE_TEMPLATE.format(lev=lev, year=year)
        )
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f'缓存不存在: {pkl_path}')
        with open(pkl_path, 'rb') as f:
            raw_cache = pickle.load(f)
        compact, n_emb_basins, n_emb = _compact_embankment_cache(raw_cache, year)
        cache[year] = compact
        print(
            f'  加载缓存 {year}: {len(raw_cache)} basins | '
            f'土石坝 {n_emb:,} 座 / {n_emb_basins} basins'
        )
    return cache


# ══════════════════════════════════════════════════════════════════════════════
# MC 采样
# ══════════════════════════════════════════════════════════════════════════════
def basin_weighted_recall(cache_year, hid):
    """仅按该流域土石坝的 z_level 计算 recall，缺失时返回默认值。"""
    if hid not in cache_year:
        return RECALL_DEFAULT
    _, z_arr = _embankment_arrays(
        cache_year[hid], context=f'HYBAS_ID={int(hid)}'
    )
    if len(z_arr) == 0:
        return RECALL_DEFAULT
    return float(np.array([RECALL_BY_Z.get(int(z), RECALL_DEFAULT) for z in z_arr]).mean())


def precompute_recall(cache, hybas_ids):
    """预计算各年份、各流域的 weighted recall，返回 {year: {hid: recall}}。"""
    recall = {}
    for year in [2010, 2015, 2020]:
        recall[year] = {int(hid): basin_weighted_recall(cache[year], int(hid))
                        for hid in hybas_ids}
    return recall


def mc_sample_update(gdf, cache, rng, recall=None):
    """
    对每个 basin 的 embankment dam 做 Bernoulli(precision) 采样。
    更新字段：
      - {year}_d_e  (year = 2010, 2015, 2020)
      - dDam_e, dDam_total
    固定字段不变: dDam_g/b/a
    """
    basin_area = to_num(gdf['SUB_AREA']).replace(0, np.nan).values
    hybas_ids  = gdf['HYBAS_ID'].values

    fixed_cnt_gba = gdf[['dDam_g', 'dDam_b', 'dDam_a']].sum(axis=1).values

    sim_cnt = {yr: np.zeros(len(gdf), dtype=np.float64) for yr in [2010, 2015, 2020]}

    for i, hid in enumerate(hybas_ids):
        hid_int = int(hid)
        for year in [2010, 2015, 2020]:
            if hid_int in cache[year]:
                prec, _ = _embankment_arrays(
                    cache[year][hid_int],
                    context=f'year={year}, HYBAS_ID={hid_int}',
                )
                sampled = rng.random(len(prec)) < prec
                n_tp = int(sampled.sum())
                if recall is not None:
                    r = recall[year].get(hid_int, RECALL_DEFAULT)
                else:
                    r = RECALL_DEFAULT
                n_missed = int(rng.poisson(n_tp * (1 - r) / r)) if n_tp > 0 else 0
                sim_cnt[year][i] = n_tp + n_missed

    for year in [2010, 2015, 2020]:
        gdf[f'{year}_d_e'] = sim_cnt[year] / basin_area

    gdf['dDam_e']     = (sim_cnt[2020] - sim_cnt[2010]) / basin_area
    gdf['dDam_total'] = gdf['dDam_e'] + fixed_cnt_gba

    return gdf
