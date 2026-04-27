# -*- coding: utf-8 -*-
# 06_Semisub_HeaveOperability.py — North Sea: Heave response & operability from
# "RMS response per meter Hs" vs Tp, with per‑configuration Hs/Tp limits support, A/B diff,
# and Dynamic Draft Switching (deep 17.75 m -> shallow 15.75 m when Hs/Tp exceeded),
# plus Dynamic Switch Map (deep contribution share).

import os
import re
import pathlib
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import matplotlib.pyplot as plt
import cartopy  # to ensure environment var is honored before data fetches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -----------------------------
# Environment & Page setup
# -----------------------------
# Use a writable cache on Streamlit Cloud and favor small Natural Earth sets.
os.environ.setdefault("CARTOPY_DATA_DIR", "/tmp/cartopy")
pathlib.Path(os.environ["CARTOPY_DATA_DIR"]).mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Semisub: Heave & Operability (North Sea)", layout="wide", page_icon="⚓")
st.title("Semisub — Heave Response & Operability (North Sea)")

# -----------------------------
# Constants
# -----------------------------
ZOOM_EXTENT = [-13, 35, 52, 76]
FEATURE_SCALE = "50m"        # lighter data than "10m" for faster cold starts
BASE_CMAP_CONT = "turbo"      # continuous fields (e.g., expected heave)
CMAP_OPERABILITY = "jet_r"    # operability maps (reversed jet)
CLIP_PCT = 99.6
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NS_CANDIDATES = [
    os.path.join(BASE_DIR, "..", "metocean_scatter_050deg_NS_monthclim.nc"),
    os.path.join(BASE_DIR, "metocean_scatter_050deg_NS_monthclim.nc"),
]

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_nc(paths):
    for p in paths:
        try:
            return xr.open_dataset(p), p
        except Exception:
            pass
    raise FileNotFoundError("Could not find metocean_scatter_050deg_NS_monthclim.nc")

def bin_centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])

def unwrap_lon_centers_from_edges(lon_edges):
    lon_c = bin_centers(lon_edges)
    if np.nanmax(lon_edges) > 180:
        lon_c = ((lon_c + 180) % 360) - 180
    return lon_c

def to_sorted_lon_lat(field2d, lat_c, lon_edges):
    if lat_c[0] > lat_c[-1]:
        field2d = field2d[::-1, :]
        lat_c = lat_c[::-1]
    lon_uns = unwrap_lon_centers_from_edges(lon_edges)
    j_sort = np.argsort(lon_uns)
    return field2d[:, j_sort], lat_c, lon_uns[j_sort]

def normalize_pdf(prob):
    tot = prob.sum(dim=("hs_bin","tp_bin"))
    return xr.where(tot > 0, prob/tot, 0)

# Percent helpers (static)
def pct_ticks():
    return np.arange(0, 101, 10)

def pct_levels():
    return np.linspace(0, 100, 61)

# Percent helpers respecting a lower bound
def pct_levels_from(lo):
    lo = float(lo)
    n = int(max(1, np.floor(100 - lo))) + 1  # 1%-step levels
    return np.linspace(lo, 100, n)

def pct_contours_from(lo):
    start = int(np.ceil(lo / 10.0) * 10)
    start = min(start, 100)
    return np.arange(start, 101, 10)

def pct_ticks_from(lo):
    base = np.arange(0, 101, 10)
    ticks = np.unique(np.concatenate(([float(lo)], base)))
    return ticks[ticks >= lo]

def hs_levels_zoom(arr):
    vmin = float(np.nanmin(arr)); vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0
    filled = np.arange(np.floor(vmin/0.1)*0.1, np.ceil(vmax/0.1)*0.1 + 1e-12, 0.1)
    contours = np.arange(np.floor(vmin/0.2)*0.2, np.ceil(vmax/0.2)*0.2 + 1e-12, 0.2)
    return filled, contours, contours

def plot_zoom(lon, lat, data, title, filled, contours, ticks, cmap=BASE_CMAP_CONT, show_grid=True):
    fig = plt.figure(figsize=(14, 6), dpi=160)
    ax = plt.axes(projection=ccrs.PlateCarree())
    cf = ax.contourf(lon, lat, data, levels=filled, cmap=cmap, extend="both",
                     transform=ccrs.PlateCarree(), zorder=1)
    try:
        cs = ax.contour(lon, lat, data, levels=contours, colors="black", linewidths=0.45,
                        transform=ccrs.PlateCarree(), zorder=2)
        ax.figure.canvas.draw()
        ax.clabel(cs, fontsize=6, inline=True, inline_spacing=1, fmt="%g", rightside_up=True)
    except Exception:
        pass
    ax.add_feature(cfeature.LAND.with_scale(FEATURE_SCALE), facecolor="lightgray", edgecolor="none", zorder=10)
    ax.add_feature(cfeature.COASTLINE.with_scale(FEATURE_SCALE), linewidth=0.7, zorder=11)
    ax.add_feature(cfeature.BORDERS.with_scale(FEATURE_SCALE), linewidth=0.3, zorder=12)
    ax.set_extent(ZOOM_EXTENT, crs=ccrs.PlateCarree())
    if show_grid:
        Lon2D, Lat2D = np.meshgrid(lon, lat)
        ax.scatter(Lon2D.ravel(), Lat2D.ravel(), s=6, color="gray", alpha=0.6,
                   transform=ccrs.PlateCarree(), zorder=3)
    cb = plt.colorbar(cf, ax=ax, shrink=0.75, aspect=30, pad=0.01, ticks=ticks)
    cb.set_label(title)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title)
    plt.subplots_adjust(left=0.02, right=0.97, top=0.93, bottom=0.06)
    st.pyplot(fig, width="stretch")

def interp_rows(M, x_from, x_to):
    return np.vstack([np.interp(x_to, x_from, r) for r in M])

def plot_hs_tp_curve(tp_vals, hs_limits, system_name, note_text=None, hs_limits_2=None, system_name_2=None):
    """Small line plot of the selected system’s Hs/Tp limit curve (optionally with an overlay curve)."""
    fig, ax = plt.subplots(figsize=(4.2, 2.8), dpi=160)
    ax.plot(tp_vals, hs_limits, "-o", linewidth=1.8, markersize=3, label=str(system_name))
    if hs_limits_2 is not None:
        lbl2 = str(system_name_2) if system_name_2 is not None else "Comparison"
        ax.plot(tp_vals, hs_limits_2, "--o", linewidth=1.6, markersize=3, label=lbl2, alpha=0.9)
        ax.legend(fontsize=7, frameon=False, loc="best")
    ax.set_xlabel("Tp [s]")
    ax.set_ylabel("Hs limit [m]")
    ax.grid(True, alpha=0.35)
    ax.set_title(f"{system_name} — Hs/Tp limit")
    if note_text:
        ax.text(0.01, 0.02, note_text, transform=ax.transAxes, fontsize=7, va="bottom", ha="left", alpha=0.8)
    st.pyplot(fig, width="stretch")

def default_index_for_substring(names, substr, fallback_idx):
    s = substr.lower()
    for i, n in enumerate(names):
        if s in str(n).lower():
            return i
    return fallback_idx


def _norm_label(x):
    # Robust matching for CSV headers / row labels with quotes, commas, extra spaces
    s = str(x) if x is not None else ""
    s = s.strip().strip('"').strip("'")
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def split_cfg_name(cfg):
    """Expected cfg label like: '17.75m, Original' or '15.75m, Blisters'"""
    s = str(cfg).strip().strip('"').strip("'")
    parts = [p.strip() for p in s.split(",")]
    left = parts[0] if parts else s
    rig = ", ".join(parts[1:]).strip() if len(parts) >= 2 else ""
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*m", left)
    draft = float(m.group(1)) if m else None
    return draft, rig

def unique_drafts_and_rigs(cfg_names):
    drafts = []
    rigs = []
    for c in cfg_names:
        d, r = split_cfg_name(c)
        if d is not None:
            drafts.append(d)
        if r:
            rigs.append(r)
    drafts = sorted(set(drafts))
    rigs = sorted(set(rigs), key=lambda x: x.lower())
    return drafts, rigs

def find_cfg(cfg_names, draft, rig):
    """Find exact config label in cfg_names matching draft + rig (case/space tolerant)."""
    rig_n = _norm_label(rig)
    for c in cfg_names:
        d, r = split_cfg_name(c)
        if d is None:
            continue
        if abs(d - float(draft)) < 1e-9 and _norm_label(r) == rig_n:
            return c
    return None

# -----------------------------
# Load dataset (regional)
# -----------------------------
try:
    ds, used_path = load_nc(NS_CANDIDATES)
except Exception as e:
    st.error(f"Failed to load NS dataset: {e}")
    st.stop()

for k in ["prob","hs_edges","tp_edges","lat3_edges","lon3_edges"]:
    if k not in ds:
        st.error(f"Dataset missing {k}")
        st.stop()

hs_edges = ds["hs_edges"].values
tp_edges = ds["tp_edges"].values
lat_edges = ds["lat3_edges"].values
lon_edges = ds["lon3_edges"].values

units = str(ds["hs_edges"].attrs.get("units","" )).lower()
if "cm" in units or (np.nanmax(hs_edges) > 50 and "m" not in units):
    hs_edges = hs_edges / 100.0

hs_c = bin_centers(hs_edges)  # (hs_bin)
tp_c = bin_centers(tp_edges)  # (tp_bin)
lat_c = bin_centers(lat_edges)
lon_c = unwrap_lon_centers_from_edges(lon_edges)

# -----------------------------
# Sidebar (part 1: generic)
# -----------------------------
with st.sidebar:
    st.subheader("Aggregation")
    mode = st.radio("Use", ["By month","Annual"], horizontal=True)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_vals = np.arange(1,13)
    month_label = st.selectbox("Month", months, index=0)
    month_idx = dict(zip(months, month_vals))[month_label]

    st.subheader("Wave acceptance (Hs/Tp limit)")
    limit_mode = st.radio(
        "Limit mode",
        ["Single Hcrit", "Single Hs/Tp curve (CSV)", "Per-configuration Hs/Tp limits (CSV)"],
        index=2
    )
    Hcrit = st.number_input("Hs threshold (m)", min_value=0.1, max_value=15.0, value=3.5, step=0.1)
    limit_csv_single = st.file_uploader("Upload single Hs/Tp curve CSV (2 columns: Tp, Hs_limit)", type=["csv"], key="hs_tp_limit_single")
    limit_csv_multi  = st.file_uploader("Upload per-configuration Hs/Tp limits CSV (Tp + one column per config)", type=["csv"], key="hs_tp_limit_multi")

    st.subheader("Heave per meter Hs CSV")
    heave_csv = st.file_uploader("Upload RMS response per meter Hs (by TP)", type=["csv"], key="heave_per_hs_csv")

    st.subheader("Heave acceptance")
    heave_limit = st.number_input("Heave limit (m)", min_value=0.1, max_value=20.0, value=3.0, step=0.1)

    st.subheader("Display")
    show_grid = st.checkbox("Show grid points", True)

    st.subheader("Colorbar")
    cbar_lower = st.slider(
        "Operability colorbar lower limit (%)",
        min_value=0, max_value=95, value=50, step=1,
        help="Sets the lower bound of the operability colorbar. Data are NOT clipped; colors span [lower, 100]."
    )

# -----------------------------
# Probability field
# -----------------------------
if mode == "By month":
    prob = ds["prob"].sel(month=month_idx)
    title_suffix = f" — {month_label}"
else:
    prob = ds["prob"].sum(dim="month")
    title_suffix = " — Annual"

prob = normalize_pdf(prob)  # (hs,tp,lat,lon)
HS = xr.DataArray(hs_c, dims=["hs_bin"])

# -----------------------------
# Parse heave-per-Hs CSV (rows = configurations; columns = TP1..TPn)
# -----------------------------
def parse_heave_per_hs(uploaded_csv):
    """
    Robust parser for heave-per-Hs CSV with multi-row header:

    Example:
    Hull alternative,MPM double amplitude,...
    ,TP1,TP2,TP3,...
    17.75m,0.0136,0.0136,...
    15.75m,0.0192,0.0192,...
    """

    df = pd.read_csv(uploaded_csv, header=None)

    # --- Find the TP header row (row containing TP1) ---
    header_row_candidates = df.index[
        df.apply(
            lambda r: r.astype(str).str.fullmatch(r"TP1").any(),
            axis=1
        )
    ]

    if len(header_row_candidates) == 0:
        st.error(
            "Could not find TP header row (TP1, TP2, …) in heave CSV."
        )
        st.stop()

    header_row = header_row_candidates[0]

    # --- Data starts below the TP header ---
    data = df.iloc[header_row + 1 :].reset_index(drop=True)

    # Configuration names (first column)
    cfg_names = data.iloc[:, 0].astype(str).tolist()

    # Heave per meter Hs (m/m)
    R_heave_per_Hs = data.iloc[:, 1:].astype(float).values

    # Ordinal TP indices (mapping handled later)
    tp_centers_csv = np.arange(
        1, R_heave_per_Hs.shape[1] + 1
    )

    return cfg_names, R_heave_per_Hs, tp_centers_csv

if heave_csv is None:
    st.info("Upload the 'RMS response per meter Hs' CSV to proceed.")
    st.stop()

cfg_names, R_heave_per_Hs_csv, tp_centers_csv = parse_heave_per_hs(heave_csv)
n_csv = R_heave_per_Hs_csv.shape[1]

# Map CSV TP bins to dataset TP bins
if len(tp_c) == n_csv:
    R_use = R_heave_per_Hs_csv
    mapping_note = f"TP mapping: 1:1 ({n_csv} bins)."
elif (tp_centers_csv is not None) and (np.isfinite(tp_centers_csv).sum() >= 3):
    R_use = interp_rows(R_heave_per_Hs_csv, tp_centers_csv, tp_c)
    mapping_note = "TP mapping: interpolated from CSV Tp centers."
else:
    k = np.arange(1, n_csv+1)
    tp_min, tp_max = float(tp_c.min()), float(tp_c.max())
    x_from = tp_min + ((k - 0.5)/n_csv) * (tp_max - tp_min)
    R_use = interp_rows(R_heave_per_Hs_csv, x_from, tp_c)
    mapping_note = "TP mapping: ordinal across dataset Tp range."

# -----------------------------
# Build Hs_limit(Tp) per configuration
# -----------------------------
def build_single_curve(tp_centers, csv_file, fallback_val):
    if csv_file is None:
        return np.full_like(tp_centers, fallback_val, dtype=float)
    try:
        df = pd.read_csv(csv_file)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding="latin-1")

    cols_low = [c.lower().strip() for c in df.columns]
    def col_like(options):
        for o in options:
            if o in cols_low:
                return df.columns[cols_low.index(o)]
        return None

    tp_col = col_like(["tp (s)", "tp", "tp_s", "period"])
    hs_col = col_like(["hs_limit (m)", "hs limit (m)", "hs_limit", "hs (m)", "hs"])
    if tp_col is None or hs_col is None:
        st.error("Single curve CSV must contain 'Tp (s)' and 'Hs_limit (m)'. Using Hcrit.")
        return np.full_like(tp_centers, fallback_val, dtype=float)

    tp_in = pd.to_numeric(df[tp_col], errors="coerce").astype(float).values
    hs_in = pd.to_numeric(df[hs_col], errors="coerce").astype(float).values
    m = np.isfinite(tp_in) & np.isfinite(hs_in)
    tp_in, hs_in = tp_in[m], hs_in[m]
    if tp_in.size < 2:
        st.error("Single curve CSV must have ≥2 rows. Using Hcrit.")
        return np.full_like(tp_centers, fallback_val, dtype=float)

    order = np.argsort(tp_in)
    return np.interp(tp_centers, tp_in[order], hs_in[order], left=hs_in[order][0], right=hs_in[order][-1])

def parse_limits_per_config(csv_file, cfg_names, tp_centers, fallback_val):
    """
    CSV format:
      Tp (s), A, B, C, ...
      6,     4.0, 4.5, 5.0, ...
    Returns:
      dict: cfg -> 1D array (len(tp_centers))
      missing: list of cfg names not found in CSV
    """
    if csv_file is None:
        # no CSV: same constant limits for all configs
        return {c: np.full_like(tp_centers, fallback_val, dtype=float) for c in cfg_names}, []

    try:
        df = pd.read_csv(csv_file)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding="latin-1")

    # Detect Tp column
    low = [c.lower().strip() for c in df.columns]
    tp_candidates = ["tp (s)", "tp", "tp_s", "period"]
    tp_col = None
    for cand in tp_candidates:
        if cand in low:
            tp_col = df.columns[low.index(cand)]
            break
    if tp_col is None:
        st.error("Per-configuration limits CSV must contain a 'Tp (s)' column.")
        return {c: np.full_like(tp_centers, fallback_val, dtype=float) for c in cfg_names}, cfg_names

    # Numeric Tp and per-config series
    tp_in = pd.to_numeric(df[tp_col], errors="coerce").astype(float).values
    ok_tp = np.isfinite(tp_in)
    tp_in = tp_in[ok_tp]

    limits_by_cfg = {}
    missing = []

    # Column name matching: tolerant to quotes/commas/spaces
    col_map = {_norm_label(c): c for c in df.columns if c != tp_col}
    for cfg in cfg_names:
        key = _norm_label(cfg)
        if key not in col_map:
            missing.append(cfg)
            continue
        vals = pd.to_numeric(df[col_map[key]], errors="coerce").astype(float).values
        vals = vals[ok_tp]
        if np.isfinite(vals).sum() < 2:
            missing.append(cfg)
            continue
        order = np.argsort(tp_in)
        hs_curve = np.interp(tp_centers, tp_in[order], vals[order], left=vals[order][0], right=vals[order][-1])
        limits_by_cfg[cfg] = hs_curve

    # Fill missing with fallback constant
    for cfg in missing:
        limits_by_cfg[cfg] = np.full_like(tp_centers, fallback_val, dtype=float)

    return limits_by_cfg, missing

# Build dictionary of Hs_limit(Tp) per configuration
if limit_mode == "Single Hcrit":
    Hs_limit_by_cfg = {c: np.full_like(tp_c, Hcrit, dtype=float) for c in cfg_names}
    limit_note = "Wave limit: single Hcrit for all configurations."
elif limit_mode == "Single Hs/Tp curve (CSV)":
    single_curve = build_single_curve(tp_c, limit_csv_single, Hcrit)
    Hs_limit_by_cfg = {c: single_curve for c in cfg_names}
    limit_note = "Wave limit: single Hs/Tp curve applied to all configurations."
else:
    Hs_limit_by_cfg, missing_cfgs = parse_limits_per_config(limit_csv_multi, cfg_names, tp_c, Hcrit)
    if missing_cfgs:
        st.warning("Per-configuration limit CSV missing columns for: " + ", ".join(missing_cfgs) + ". Using Hcrit fallback for those.")
    limit_note = "Wave limit: per-configuration Hs/Tp curves from CSV (fallback to Hcrit if missing)."

# -----------------------------
# Sidebar (part 2: A/B + Dynamic draft controls — now that cfg_names exist)
# -----------------------------
with st.sidebar:
    st.subheader("Two‑config comparison")

    drafts_cmp, rigs_cmp = unique_drafts_and_rigs(cfg_names)

    cmp_mode = st.radio(
        "Comparison mode",
        [
            "Rig comparison (same draft)",
            "Rig+Draft comparison",
            "Manual pick",
        ],
        index=0,
        key="cmp_mode"
    )

    def _default_rig_index(options, needle, fallback=0):
        return default_index_for_substring(options, needle, fallback)

    if (cmp_mode == "Rig comparison (same draft)") and (len(drafts_cmp) >= 1) and (len(rigs_cmp) >= 2):
        cmp_draft = st.selectbox(
            "Draft for comparison (m)",
            [f"{d:.2f}" for d in drafts_cmp],
            index=0,
            key="cmp_draft"
        )
        cmp_draft_val = float(cmp_draft)

        rigA_idx = _default_rig_index(rigs_cmp, "original", 0)
        rigB_idx = _default_rig_index(rigs_cmp, "blister", min(1, len(rigs_cmp)-1))

        rigA = st.selectbox("Rig A", rigs_cmp, index=rigA_idx, key="cmp_rigA")
        rigB = st.selectbox("Rig B", rigs_cmp, index=rigB_idx, key="cmp_rigB")

        cfgA_found = find_cfg(cfg_names, cmp_draft_val, rigA)
        cfgB_found = find_cfg(cfg_names, cmp_draft_val, rigB)

        if (cfgA_found is None) or (cfgB_found is None):
            st.warning("Could not resolve rigs at this draft from cfg list — falling back to manual pick.")
            cfgA = st.selectbox("Config A", cfg_names, index=0, key="cmpA_fallback")
            cfgB = st.selectbox("Config B", cfg_names, index=min(1, len(cfg_names)-1), key="cmpB_fallback")
        else:
            cfgA, cfgB = cfgA_found, cfgB_found
            st.caption(f"Comparing: **{cfgA}** vs **{cfgB}**")

    elif (cmp_mode == "Rig+Draft comparison") and (len(drafts_cmp) >= 1) and (len(rigs_cmp) >= 2):
        colL, colR = st.columns(2)
        with colL:
            rigA_idx = _default_rig_index(rigs_cmp, "original", 0)
            rigA = st.selectbox("Rig A", rigs_cmp, index=rigA_idx, key="cmp2_rigA")
            dA = st.selectbox("Draft A (m)", [f"{d:.2f}" for d in drafts_cmp], index=0, key="cmp2_draftA")
        with colR:
            rigB_idx = _default_rig_index(rigs_cmp, "blister", min(1, len(rigs_cmp)-1))
            rigB = st.selectbox("Rig B", rigs_cmp, index=rigB_idx, key="cmp2_rigB")
            dB = st.selectbox("Draft B (m)", [f"{d:.2f}" for d in drafts_cmp], index=min(1, len(drafts_cmp)-1), key="cmp2_draftB")

        cfgA_found = find_cfg(cfg_names, float(dA), rigA)
        cfgB_found = find_cfg(cfg_names, float(dB), rigB)

        if (cfgA_found is None) or (cfgB_found is None):
            st.warning("Could not resolve draft+rig selections from cfg list — falling back to manual pick.")
            cfgA = st.selectbox("Config A", cfg_names, index=0, key="cmpA2_fallback")
            cfgB = st.selectbox("Config B", cfg_names, index=min(1, len(cfg_names)-1), key="cmpB2_fallback")
        else:
            cfgA, cfgB = cfgA_found, cfgB_found
            st.caption(f"Comparing: **{cfgA}** vs **{cfgB}**")

    else:
        cfgA = st.selectbox("Config A", cfg_names, index=0, key="cmpA")
        cfgB = st.selectbox("Config B", cfg_names, index=min(1, len(cfg_names)-1), key="cmpB")

    cmp_metric = st.selectbox(
        "Metric for comparison",
        [
            "Operability: heave ≤ limit (%)",
            "Operability: wave ≤ Hs/Tp limit (%)",
            "Operability: ALL limits (%)",
        ],
        index=2,
        help="This metric is used for the A vs B difference map."
    )
    zero_center = st.checkbox("Zero‑center difference color scale", True,
                              help="Keeps negative and positive ranges symmetric around 0.")
    diff_absmax = st.slider("Max Δ for colorbar (pp)", 5, 50, 20, 1,
                            help="Sets ±range for the difference map in percentage points.")

    st.subheader("Draft strategy")
    draft_mode = st.radio(
        "Draft mode",
        [
            "Use selected configuration only",
            "Dynamic: deep → shallow when Hs/Tp exceeded"
        ],
        index=0
    )

    # Let the user select which CSV rows correspond to deep and shallow drafts.
    deep_default    = default_index_for_substring(cfg_names, "17.75", 0)
    shallow_default = default_index_for_substring(cfg_names, "15.75", min(1, len(cfg_names)-1))
    deep_cfg_name    = st.selectbox("Deep-draft configuration",    cfg_names, index=deep_default,    key="deep_cfg")
    shallow_cfg_name = st.selectbox("Shallow-draft configuration", cfg_names, index=shallow_default, key="shallow_cfg")

# -----------------------------
# Compute fields for SELECTED configuration (main UX)
# -----------------------------
# Prefer Draft + Rig selectors if configuration labels follow 'XX.XXm, RigName'
drafts, rigs = unique_drafts_and_rigs(cfg_names)
rig_sel = None
Draft_sel_val = None

if (len(drafts) >= 1) and (len(rigs) >= 1):
    with st.sidebar:
        st.subheader("Selected configuration")
        default_rig_idx = default_index_for_substring(rigs, "original", 0)
        rig_sel = st.selectbox("Rig alternative", rigs, index=default_rig_idx, key="rig_sel")
        draft_sel = st.selectbox("Draft (m)", [f"{d:.2f}" for d in drafts], index=0, key="draft_sel")

    Draft_sel_val = float(draft_sel)
    cfg_found = find_cfg(cfg_names, Draft_sel_val, rig_sel)
    if cfg_found is None:
        st.error(f"Could not find a configuration matching {Draft_sel_val:.2f}m, {rig_sel}. Falling back to first row.")
        cfg = cfg_names[0]
    else:
        cfg = cfg_found
else:
    cfg = st.selectbox("Hull alternative", cfg_names, index=0)

i_cfg = cfg_names.index(cfg)

# --- Small line plot of the selected system’s Hs/Tp limit curve ---
Hs_limit_tp_sel = Hs_limit_by_cfg[cfg]
curve_col, spacer = st.columns([1.2, 3.8])
with curve_col:
    # Overlay the other rig at the same draft (if available) for quick visual comparison
    hs2 = None
    name2 = None
    if (Draft_sel_val is not None) and (rig_sel is not None) and (len(rigs) >= 2):
        other_rig = None
        if "original" in _norm_label(rig_sel):
            for r in rigs:
                if "blister" in _norm_label(r):
                    other_rig = r
                    break
        elif "blister" in _norm_label(rig_sel):
            for r in rigs:
                if "original" in _norm_label(r):
                    other_rig = r
                    break
        if other_rig is None:
            for r in rigs:
                if _norm_label(r) != _norm_label(rig_sel):
                    other_rig = r
                    break
        if other_rig is not None:
            cfg2 = find_cfg(cfg_names, Draft_sel_val, other_rig)
            if cfg2 is not None:
                hs2 = Hs_limit_by_cfg[cfg2]
                name2 = cfg2

    plot_hs_tp_curve(tp_c, Hs_limit_tp_sel, cfg, note_text=limit_note, hs_limits_2=hs2, system_name_2=name2)

# Heave per meter Hs vs Tp (for selected cfg)
fTp = xr.DataArray(R_use[i_cfg], dims=["tp_bin"])  # m/m

# Sea-state heave magnitude
HS2D = xr.DataArray(hs_c, dims=["hs_bin"]).broadcast_like(prob)
TPmask = fTp.broadcast_like(prob)
M_heave = HS2D * TPmask  # m

# Wave acceptance for selected cfg
HsLim2D_sel = xr.DataArray(Hs_limit_tp_sel, dims=["tp_bin"]).broadcast_like(prob)
I_wave_sel = (HS2D <= HsLim2D_sel).astype(float)

# Heave acceptance
I_heave = (M_heave <= heave_limit).astype(float)

# Expected heave
E_heave = (prob * M_heave).sum(dim=("hs_bin","tp_bin"))  # (lat,lon)

# Operabilities for selected cfg (%)
P_heave = (prob * I_heave ).sum(dim=("hs_bin","tp_bin")) * 100.0
P_wave  = (prob * I_wave_sel).sum(dim=("hs_bin","tp_bin")) * 100.0
P_both  = (prob * I_heave * I_wave_sel).sum(dim=("hs_bin","tp_bin")) * 100.0

# -----------------------------
# Dynamic draft switching (deep -> shallow when Hs/Tp exceeded)
# -----------------------------
P_dyn = None
P_dyn_deep_share = None  # deep contribution (% of accepted dynamic operability)

if draft_mode == "Dynamic: deep → shallow when Hs/Tp exceeded":
    try:
        j_deep = cfg_names.index(deep_cfg_name)
        j_shallow = cfg_names.index(shallow_cfg_name)
    except ValueError:
        st.error("Deep/Shallow configuration names not found in CSV list.")
        j_deep = j_shallow = 0

    # Deep draft responses & limits
    fTp_deep     = xr.DataArray(R_use[j_deep], dims=["tp_bin"])
    M_heave_deep = HS2D * fTp_deep.broadcast_like(prob)
    HsLim2D_deep = xr.DataArray(Hs_limit_by_cfg[deep_cfg_name], dims=["tp_bin"]).broadcast_like(prob)
    I_wave_deep  = (HS2D <= HsLim2D_deep).astype(float)
    I_heave_deep = (M_heave_deep <= heave_limit).astype(float)

    # Shallow draft responses & limits
    fTp_shallow  = xr.DataArray(R_use[j_shallow], dims=["tp_bin"])
    M_heave_sh   = HS2D * fTp_shallow.broadcast_like(prob)
    HsLim2D_sh   = xr.DataArray(Hs_limit_by_cfg[shallow_cfg_name], dims=["tp_bin"]).broadcast_like(prob)
    I_wave_sh    = (HS2D <= HsLim2D_sh).astype(float)
    I_heave_sh   = (M_heave_sh <= heave_limit).astype(float)

    # Switching rule: stay deep if deep passes wave criterion; otherwise shallow.
    I_wave_dyn  = xr.where(I_wave_deep == 1, I_wave_deep, I_wave_sh)
    I_heave_dyn = xr.where(I_wave_deep == 1, I_heave_deep, I_heave_sh)

    # Dynamic combined operability (%)
    P_dyn = (prob * I_wave_dyn * I_heave_dyn).sum(dim=("hs_bin","tp_bin")) * 100.0

    # ---- Dynamic switch map: fraction of accepted dynamic time coming from deep branch ----
    accept_deep = prob * I_wave_deep * I_heave_deep
    accept_sh   = prob * (1.0 - I_wave_deep) * I_heave_sh
    total_acc   = accept_deep + accept_sh
    P_dyn_deep_share = xr.where(
        total_acc.sum(dim=("hs_bin","tp_bin")) > 0,
        (accept_deep.sum(dim=("hs_bin","tp_bin")) / total_acc.sum(dim=("hs_bin","tp_bin"))) * 100.0,
        np.nan
    )  # (lat,lon) in %

    # ---- Diagnostics: why dynamic may look unchanged ----
    same_cfg = (deep_cfg_name == shallow_cfg_name)
    if same_cfg:
        st.warning("Deep and shallow selections are the SAME configuration — dynamic switching will have no effect.")

    try:
        same_limits = np.allclose(Hs_limit_by_cfg[deep_cfg_name], Hs_limit_by_cfg[shallow_cfg_name], rtol=1e-6, atol=1e-6)
    except Exception:
        same_limits = False
    try:
        same_heave = np.allclose(np.asarray(R_use[j_deep]), np.asarray(R_use[j_shallow]), rtol=1e-6, atol=1e-6)
    except Exception:
        same_heave = False
    if same_limits:
        st.info("Deep and shallow Hs/Tp limit curves are identical (check your per-configuration limits CSV).")
    if same_heave:
        st.info("Deep and shallow heave-per-Hs curves are identical (check your RMS-per-Hs CSV mapping).")

    # How often would deep actually fail the wave criterion? (probability mass)
    deep_fail_prob  = (prob * (1.0 - I_wave_deep)).sum(dim=("hs_bin","tp_bin"))  # (lat,lon), 0..1
    deep_fail_share = float(deep_fail_prob.mean(dim=("lat3_bin","lon3_bin"), skipna=True))
    st.caption(f"Dynamic switch trigger (deep wave criterion fails): spatial-mean probability mass = {deep_fail_share*100:.2f}%")

# -----------------------------
# Metric selector (as in MotionOperability)
# -----------------------------
st.sidebar.subheader("Metric to show")
metric_options = [
    "Expected heave (m)",
    "Operability: heave ≤ limit (%)",
    "Operability: wave ≤ Hs/Tp limit (%)",
    "Operability: ALL limits (%)",
]
if P_dyn is not None:
    metric_options.append("Operability: Dynamic deep→shallow (%)")
    metric_options.append("Dynamic: deep contribution share (%)")
metric = st.sidebar.selectbox("Metric", metric_options)

# Prep 2D arrays (ordered lon,lat for plotting)
def prep(field):
    arr = field.transpose("lat3_bin","lon3_bin").values
    return to_sorted_lon_lat(arr, lat_c, lon_edges)

heave2, latp, lonp = prep(E_heave)
heave_hi = np.nanpercentile(heave2, CLIP_PCT)
heave2 = np.clip(heave2, None, heave_hi)

p_heave2, latp, lonp = prep(P_heave)
p_wave2,  latp, lonp = prep(P_wave)
p_both2,  latp, lonp = prep(P_both)

if P_dyn is not None:
    p_dyn2,      latp, lonp = prep(P_dyn)
    deep_share2, latp, lonp = prep(P_dyn_deep_share)

# -----------------------------
# Render map depending on metric
# -----------------------------
if metric == "Expected heave (m)":
    filled, contours, ticks = hs_levels_zoom(heave2)
    plot_zoom(
        lonp, latp, heave2,
        f"Expected heave (m) — {cfg}{title_suffix}",
        filled, contours, ticks,
        cmap=BASE_CMAP_CONT, show_grid=show_grid
    )

elif metric == "Operability: heave ≤ limit (%)":
    filled = pct_levels_from(cbar_lower)
    contours = pct_contours_from(cbar_lower)
    ticks = pct_ticks_from(cbar_lower)
    plot_zoom(
        lonp, latp, p_heave2,
        f"Operability (%) — Heave ≤ {heave_limit:.2f} m{title_suffix}",
        filled, contours, ticks,
        cmap=CMAP_OPERABILITY, show_grid=show_grid
    )

elif metric == "Operability: wave ≤ Hs/Tp limit (%)":
    filled = pct_levels_from(cbar_lower)
    contours = pct_contours_from(cbar_lower)
    ticks = pct_ticks_from(cbar_lower)
    plot_zoom(
        lonp, latp, p_wave2,
        "Operability (%) — Wave (Hs/Tp limit)" + title_suffix,
        filled, contours, ticks,
        cmap=CMAP_OPERABILITY, show_grid=show_grid
    )

elif metric == "Operability: ALL limits (%)":
    filled = pct_levels_from(cbar_lower)
    contours = pct_contours_from(cbar_lower)
    ticks = pct_ticks_from(cbar_lower)
    if (P_dyn is not None) and (draft_mode.startswith("Dynamic")):
        plot_zoom(
            lonp, latp, p_dyn2,
            f"Operability (%) — Dynamic draft switching ({deep_cfg_name} → {shallow_cfg_name}){title_suffix}",
            filled, contours, ticks,
            cmap=CMAP_OPERABILITY, show_grid=show_grid
        )
    else:
        plot_zoom(
            lonp, latp, p_both2,
            f"Operability (%) — Wave ∩ Heave — {cfg}{title_suffix}",
            filled, contours, ticks,
            cmap=CMAP_OPERABILITY, show_grid=show_grid
        )

elif metric == "Operability: Dynamic deep→shallow (%)" and P_dyn is not None:
    filled = pct_levels_from(cbar_lower)
    contours = pct_contours_from(cbar_lower)
    ticks = pct_ticks_from(cbar_lower)
    plot_zoom(
        lonp, latp, p_dyn2,
        f"Operability (%) — Dynamic draft switching ({deep_cfg_name} → {shallow_cfg_name}){title_suffix}",
        filled, contours, ticks,
        cmap=CMAP_OPERABILITY, show_grid=show_grid
    )

elif metric == "Dynamic: deep contribution share (%)" and (P_dyn_deep_share is not None):
    # Share map uses 0–100% full range
    levels_share = np.linspace(0, 100, 51)
    ticks_share = np.arange(0, 101, 10)
    plot_zoom(
        lonp, latp, deep_share2,
        f"Dynamic: deep contribution share (%) — accepted time ({deep_cfg_name} vs {shallow_cfg_name}){title_suffix}",
        levels_share, ticks_share, ticks_share,
        cmap="coolwarm", show_grid=show_grid
    )

st.caption(
    mapping_note + " "
    + limit_note +
    f" • Colorbar lower bound: {cbar_lower:.0f}%"
    f" • Dataset: {os.path.basename(used_path)}"
)

# -----------------------------
# Multi‑configuration operability comparison (table + bars)
# -----------------------------
st.markdown("## Operability comparison between all configurations")
results = []
for j, cfg_j in enumerate(cfg_names):
    fTp_j = xr.DataArray(R_use[j], dims=["tp_bin"])
    M_heave_j = HS2D * fTp_j.broadcast_like(prob)

    Hs_limit_tp_j = Hs_limit_by_cfg[cfg_j]
    I_wave_j  = (HS2D <= xr.DataArray(Hs_limit_tp_j, dims=["tp_bin"]).broadcast_like(prob)).astype(float)
    I_heave_j = (M_heave_j <= heave_limit).astype(float)

    # Maps (lat,lon) of operability %
    P_heave_map = (prob * I_heave_j).sum(dim=("hs_bin","tp_bin")) * 100.0
    P_wave_map  = (prob * I_wave_j ).sum(dim=("hs_bin","tp_bin")) * 100.0
    P_both_map  = (prob * I_heave_j * I_wave_j).sum(dim=("hs_bin","tp_bin")) * 100.0

    # Scalar means (unweighted spatial mean)
    P_heave_j = float(P_heave_map.mean(dim=("lat3_bin","lon3_bin"), skipna=True))
    P_wave_j  = float(P_wave_map.mean(dim=("lat3_bin","lon3_bin"),  skipna=True))
    P_both_j  = float(P_both_map.mean(dim=("lat3_bin","lon3_bin"),  skipna=True))

    results.append({
        "Configuration": cfg_j,
        "Heave-only (%)": P_heave_j,
        "Wave-only (%)": P_wave_j,
        "Combined (%)": P_both_j
    })

df_ops = pd.DataFrame(results).set_index("Configuration")
st.dataframe(df_ops.style.format("{:.1f}"))

st.markdown("### Operability comparison chart")
fig, ax = plt.subplots(figsize=(10,5), dpi=160)
x = np.arange(len(cfg_names)); w = 0.25
ax.bar(x - w, df_ops["Heave-only (%)"], width=w, label="Heave-only")
ax.bar(x,      df_ops["Wave-only (%)"],  width=w, label="Wave-only")
ax.bar(x + w,  df_ops["Combined (%)"],   width=w, label="Combined")
ax.set_xticks(x); ax.set_xticklabels(cfg_names)
ax.set_ylabel("Operability (%)"); ax.set_ylim(0, 100)
ax.legend(); ax.grid(True, alpha=0.3)
st.pyplot(fig, width="stretch")

# -----------------------------
# A vs B operability comparison (maps + difference)
# -----------------------------
st.markdown("---")
st.markdown("## A vs B operability difference")

def operability_maps_for_cfg(cfg_name):
    """Return the three operability maps (lat,lon) in %, for a given configuration name."""
    j = cfg_names.index(cfg_name)
    fTp_j = xr.DataArray(R_use[j], dims=["tp_bin"])  # m/m
    M_heave_j = HS2D * fTp_j.broadcast_like(prob)    # m
    Hs_limit_tp_j = Hs_limit_by_cfg[cfg_name]

    I_wave_j  = (HS2D <= xr.DataArray(Hs_limit_tp_j, dims=["tp_bin"]).broadcast_like(prob)).astype(float)
    I_heave_j = (M_heave_j <= heave_limit).astype(float)

    P_heave_map = (prob * I_heave_j).sum(dim=("hs_bin","tp_bin")) * 100.0
    P_wave_map  = (prob * I_wave_j ).sum(dim=("hs_bin","tp_bin")) * 100.0
    P_both_map  = (prob * I_heave_j * I_wave_j).sum(dim=("hs_bin","tp_bin")) * 100.0
    return P_heave_map, P_wave_map, P_both_map

def to_numpy_sorted(field_da):
    """(lat,lon) DataArray -> numpy array + sorted lat/lon for plotting with existing helpers."""
    arr = field_da.transpose("lat3_bin","lon3_bin").values
    return to_sorted_lon_lat(arr, lat_c, lon_edges)

# Compute A & B maps
P_heave_A, P_wave_A, P_both_A = operability_maps_for_cfg(cfgA)
P_heave_B, P_wave_B, P_both_B = operability_maps_for_cfg(cfgB)

# Pick metric (base)
if cmp_metric == "Operability: heave ≤ limit (%)":
    A_map = P_heave_A; B_map = P_heave_B; metric_tag = f"Heave ≤ {heave_limit:.2f} m"
elif cmp_metric == "Operability: wave ≤ Hs/Tp limit (%)":
    A_map = P_wave_A;  B_map = P_wave_B;  metric_tag = "Wave (Hs/Tp limit)"
else:
    A_map = P_both_A;  B_map = P_both_B;  metric_tag = "Wave ∩ Heave"

# --- Only the shallow draft uses dynamic in A/B when conditions are met (deep side stays static) ---
base_metric_tag = metric_tag  # keep a copy set above
use_dynamic_in_cmp = (
    draft_mode.startswith("Dynamic")
    and (P_dyn is not None)
    and (cmp_metric == "Operability: ALL limits (%)")
)

dyn_applied_to = None
if use_dynamic_in_cmp:
    # Apply override ONLY when comparing the selected deep vs shallow pair,
    # and ONLY to the shallow side (15.75 m)
    if (cfgA == shallow_cfg_name) and (cfgB == deep_cfg_name):
        A_map = P_dyn            # A = dynamic deep→shallow
        dyn_applied_to = "A"
    elif (cfgB == shallow_cfg_name) and (cfgA == deep_cfg_name):
        B_map = P_dyn            # B = dynamic deep→shallow
        dyn_applied_to = "B"
    else:
        st.info(
            "Dynamic override in A/B applies only when comparing the selected "
            f"deep vs shallow drafts ({deep_cfg_name} vs {shallow_cfg_name})."
        )

# Titles: annotate only the dynamic side
metric_tag_A = base_metric_tag
metric_tag_B = base_metric_tag
dynamic_tag  = f"Dynamic ({deep_cfg_name} → {shallow_cfg_name}) · Wave ∩ Heave"
if dyn_applied_to == "A":
    metric_tag_A = dynamic_tag
elif dyn_applied_to == "B":
    metric_tag_B = dynamic_tag

# Prepare arrays for plotting (respect grid ordering) AFTER any dynamic override
A_np,  latp_cmp, lonp_cmp = to_numpy_sorted(A_map)
B_np,  _,        _        = to_numpy_sorted(B_map)
D_np = B_np - A_np  # Δ in percentage points (pp): positive => B better than A

# --- Headline spatial means for the chosen A/B comparison metric ---
A_mean = float(A_map.mean(dim=("lat3_bin","lon3_bin"), skipna=True))
B_mean = float(B_map.mean(dim=("lat3_bin","lon3_bin"), skipna=True))
D_mean = B_mean - A_mean

st.markdown("### Headline (spatial mean)")
m1, m2, m3 = st.columns(3)
m1.metric(f"{cfgA}", f"{A_mean:.1f} %")
m2.metric(f"{cfgB}", f"{B_mean:.1f} %")
m3.metric("Δ (B − A)", f"{D_mean:+.1f} pp")


# A and B maps with operability colorbar lower bound and jet_r
filledA = pct_levels_from(cbar_lower)
contA   = pct_contours_from(cbar_lower)
ticksA  = pct_ticks_from(cbar_lower)

colA, colB = st.columns(2)
with colA:
    plot_zoom(
        lonp_cmp, latp_cmp, A_np,
        f"{cfgA} — {metric_tag_A}{title_suffix}",
        filledA, contA, ticksA,
        cmap=CMAP_OPERABILITY, show_grid=show_grid
    )
with colB:
    plot_zoom(
        lonp_cmp, latp_cmp, B_np,
        f"{cfgB} — {metric_tag_B}{title_suffix}",
        filledA, contA, ticksA,
        cmap=CMAP_OPERABILITY, show_grid=show_grid
    )

# Difference map (B - A), diverging colormap, optional zero-centering
vmax = float(diff_absmax)
vmin = -vmax if zero_center else float(np.nanmin(D_np))
n_lev = 41
levels_diff = np.linspace(vmin, vmax, n_lev)
cmap_diff = "coolwarm"

def plot_diff(lon, lat, data, title, levels):
    fig = plt.figure(figsize=(14, 6), dpi=160)
    ax = plt.axes(projection=ccrs.PlateCarree())
    cf = ax.contourf(lon, lat, data, levels=levels, cmap=cmap_diff, extend="both",
                     transform=ccrs.PlateCarree(), zorder=1)
    try:
        step = max(5, int((levels[-1] - levels[0]) / 10))
        contour_levels = np.arange(np.round(levels[0]/step)*step, np.round(levels[-1]/step)*step + step, step)
        cs = ax.contour(lon, lat, data, levels=contour_levels, colors="black", linewidths=0.45,
                        transform=ccrs.PlateCarree(), zorder=2)
        ax.figure.canvas.draw()
        ax.clabel(cs, fontsize=6, inline=True, inline_spacing=1, fmt="%g", rightside_up=True)
    except Exception:
        pass
    ax.add_feature(cfeature.LAND.with_scale(FEATURE_SCALE), facecolor="lightgray", edgecolor="none", zorder=10)
    ax.add_feature(cfeature.COASTLINE.with_scale(FEATURE_SCALE), linewidth=0.7, zorder=11)
    ax.add_feature(cfeature.BORDERS.with_scale(FEATURE_SCALE), linewidth=0.3, zorder=12)
    ax.set_extent(ZOOM_EXTENT, crs=ccrs.PlateCarree())
    if show_grid:
        Lon2D, Lat2D = np.meshgrid(lon, lat)
        ax.scatter(Lon2D.ravel(), Lat2D.ravel(), s=6, color="gray", alpha=0.6,
                   transform=ccrs.PlateCarree(), zorder=3)
    cb = plt.colorbar(cf, ax=ax, shrink=0.75, aspect=30, pad=0.01)
    cb.set_label(title + " [pp]")
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title)
    plt.subplots_adjust(left=0.02, right=0.97, top=0.93, bottom=0.06)
    st.pyplot(fig, width="stretch")

st.markdown("### Difference map (B − A)")
diff_title_tag = base_metric_tag if dyn_applied_to is None else dynamic_tag
plot_diff(
    lonp_cmp, latp_cmp, D_np,
    f"Δ Operability (pp) — {diff_title_tag} — {cfgB} minus {cfgA}{title_suffix}",
    levels_diff
)

# Spatial means (unweighted) — computed above (headline)

st.markdown("### Spatial means (unweighted)")
st.write(
    f"- **{cfgA}**: {A_mean:.1f} % \n"
    f"- **{cfgB}**: {B_mean:.1f} % \n"
    f"- **Δ (B − A)**: **{D_mean:.1f} pp**"
)

# Compact bar chart
st.markdown("### A vs B (means)")
fig2, ax2 = plt.subplots(figsize=(5.5, 3.2), dpi=160)
ax2.bar([0, 1, 2], [A_mean, B_mean, D_mean], color=["tab:blue", "tab:orange", "tab:green"])
ax2.set_xticks([0, 1, 2]); ax2.set_xticklabels([cfgA, cfgB, "Δ(B−A)"])
ax2.set_ylabel("Operability / Δ [%, pp]")
ymin = min(0, A_mean, B_mean, D_mean) - 5
ymax = max(100, A_mean, B_mean, D_mean) + 5
ax2.set_ylim(ymin, ymax)
ax2.grid(True, axis="y", alpha=0.3)
st.pyplot(fig2, width="stretch")