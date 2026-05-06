# 01_Metocean.py — Metocean Explorer (3° global + 0.5° regional zooms)
# --------------------------------------------------------------------------------
# • Zoom regions: None \ North Sea \ Mediterranean
# • Each zoom loads its own 0.5° dataset; global uses 3° dataset.
# • Strict loader validates lon/lat bounds to prevent wrong cached dataset reuse.
# • Safe color scaling fallbacks if the zoom subset is empty or all-NaN.
# • North Sea POIs included (shown only on North Sea).
# • Per‑Tp Hs limit via CSV + preview chart; table under the map.
# • Coastline cleanup: nearest‑ocean extrapolation + last‑chance fill; buffered land mask.
# --------------------------------------------------------------------------------
import math
import os
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import plotly.graph_objects as go
from matplotlib import patheffects

st.write("RUNNING FILE:", __file__)

# ---- Land mask / extrapolation dependencies ----
from cartopy.io import shapereader as shpreader
from shapely.ops import unary_union
try:
    from shapely import vectorized as shp_vec  # shapely.vectorized.contains/covers/touches
    SHAPELY_VEC = True
except Exception:
    SHAPELY_VEC = False
try:
    from scipy.interpolate import griddata
    from scipy.ndimage import distance_transform_edt
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Data sources (adjust paths if needed)
# -----------------------------
GLOBAL_DATA_PATH = os.path.join(BASE_DIR, "..", "metocean_monthclim.nc")  # 3°
REGIONAL_DATA_PATHS = {
    "North Sea": os.path.join(BASE_DIR, "..", "metocean_scatter_050deg_NS_monthclim.nc"),
    "Mediterranean": os.path.join(BASE_DIR, "..", "metocean_scatter_050deg_MED_monthclim.nc"),
}

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(layout="wide")
st.header("🌍 Global wave statistics")

# -----------------------------
# Helpers
# -----------------------------
def bin_centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])

def unwrap_lon_centers_from_edges(lon_edges):
    lon_c = bin_centers(lon_edges)
    if np.nanmax(lon_edges) > 180:
        lon_c = ((lon_c + 180) % 360) - 180
    return lon_c

def to_sorted_lon_lat(field2d, lat_c, lon_edges):
    flip_lat = False
    if lat_c[0] > lat_c[-1]:
        field2d = field2d[::-1, :]
        lat_c = lat_c[::-1]
        flip_lat = True
    lon_unsorted = unwrap_lon_centers_from_edges(lon_edges)
    lon_sort_idx = np.argsort(lon_unsorted)
    lon_sorted = lon_unsorted[lon_sort_idx]
    field2d_sorted = field2d[:, lon_sort_idx]
    lon_inv = np.argsort(lon_sort_idx)
    return field2d_sorted, lat_c, lon_sorted, flip_lat, lon_sort_idx, lon_inv

def is_hs_quantity(label: str) -> bool:
    if "%" in label:
        return False
    return "hs" in label.lower()

def hs_ticks(step=0.5, vmin=0, vmax=10):
    lo = math.floor(vmin / step) * step
    hi = math.ceil(vmax / step) * step
    return np.arange(lo, hi + step/2, step)

def hs_shading(field, n=60):
    vmin = np.nanmin(field); vmax = np.nanmax(field)
    if vmin == vmax or not np.isfinite(vmin):
        return np.linspace(0, 1, n)
    return np.linspace(vmin, vmax, n)

def tp_ticks(step=1.0, vmin=None, vmax=None):
    if vmin is None: vmin = 0
    if vmax is None: vmax = 20
    lo = math.floor(vmin / step) * step
    hi = math.ceil(vmax / step) * step
    return np.arange(lo, hi + step*0.5, step)

def tp_shading(field, n=80):
    vmin = np.nanmin(field); vmax = np.nanmax(field)
    if vmin == vmax or not np.isfinite(vmin):
        return np.linspace(0, 20, n)
    return np.linspace(vmin, vmax, n)

def pct_ticks(): return np.arange(0, 101, 10)
def pct_shading(n=61): return np.linspace(0, 100, n)

def auto_levels(arr, n=50):
    vmin = np.nanmin(arr); vmax = np.nanmax(arr)
    if vmin == vmax or not np.isfinite(vmin):
        return np.linspace(0, 1, n)
    return np.linspace(vmin, vmax, n)

def normalize_pdf(prob):
    tot = prob.sum(dim=("hs_bin","tp_bin"))
    return xr.where(tot > 0, prob/tot, 0)

def percentile_from_cdf(cdf, centers, q):
    idx_hi = (cdf >= q).argmax(dim="hs_bin")
    idx_lo = xr.where(idx_hi > 0, idx_hi - 1, 0)
    c_lo = cdf.isel(hs_bin=idx_lo); c_hi = cdf.isel(hs_bin=idx_hi)
    cen = xr.DataArray(centers, dims=["hs_bin"])
    h_lo = cen.isel(hs_bin=idx_lo); h_hi = cen.isel(hs_bin=idx_hi)
    denom = xr.where((c_hi - c_lo) > 0, c_hi - c_lo, 1)
    w = (q - c_lo)/denom
    return h_lo + w*(h_hi - h_lo)

# ---------- Natural Earth land geometry (cached) ----------
@st.cache_resource(show_spinner=False)
def _land_geom(resolution: str, include_lakes: bool):
    """
    Build a unified polygon for LAND (and optionally LAKES) at '10m'\'50m'\'110m'.
    """
    land_path = shpreader.natural_earth(resolution=resolution, category='physical', name='land')
    land_geoms = list(shpreader.Reader(land_path).geometries())
    geom = unary_union(land_geoms)
    if include_lakes:
        lakes_path = shpreader.natural_earth(resolution=resolution, category='physical', name='lakes')
        lakes_geoms = list(shpreader.Reader(lakes_path).geometries())
        geom = unary_union([geom, unary_union(lakes_geoms)])
    return geom

def land_mask_bool(lon2d, lat2d, resolution='110m', include_lakes=False, buffer_deg=-0.12):
    """
    Return boolean mask where True means ON LAND (or lakes) after applying an optional
    buffer in degrees. Negative buffer shrinks land (good for coarse grids).
    """
    if not SHAPELY_VEC:
        st.warning("Shapely vectorized ops unavailable; skipping land mask.")
        return np.zeros_like(lon2d, dtype=bool)
    geom = _land_geom(resolution, include_lakes)
    if buffer_deg and buffer_deg != 0:
        try:
            geom = geom.buffer(buffer_deg)  # shrink land slightly (e.g., -0.12°)
        except Exception:
            pass
    try:
        on_land = shp_vec.covers(geom, lon2d, lat2d)  # includes boundary
    except Exception:
        on_land = shp_vec.contains(geom, lon2d, lat2d)
    try:
        on_land \
= shp_vec.touches(geom, lon2d, lat2d)
    except Exception:
        pass
    return on_land

def mask_land_to_nan(lon2d, lat2d, arr, resolution='110m', include_lakes=False, buffer_deg=-0.12):
    on_land = land_mask_bool(lon2d, lat2d, resolution, include_lakes, buffer_deg)
    out = arr.copy()
    out[on_land] = np.nan
    return out

# -----------------------------------------------------------

# -----------------------------
# Sidebar (controls)
# -----------------------------
with st.sidebar:
    st.subheader("Data")
    zoom_region = st.selectbox(
        "Zoom region",
        ["None", "North Sea", "Mediterranean","test"],
        index=0,
        help="Select a regional zoom (0.5° grid) or None for the global 3° view."
    )
    show_grid_points = st.checkbox("Show metocean grid points (zoom/global)", value=False)
    zoom_proj_name = st.selectbox(
        "Zoom projection",
        ["PlateCarree (default)", "Mercator", "Lambert Conformal"],
        index=0,
        help="Applies only when a zoom region is selected. Global view is always PlateCarree."
    )

    st.subheader("Aggregation")
    agg = st.radio("Use:", ["By month","Annual"], horizontal=True)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_vals = np.arange(1,13)
    chosen_label = st.selectbox("Month", months, index=4)
    label_to_idx = dict(zip(months, month_vals))

    st.subheader("Statistic")
    Hcrit = st.number_input("Hs threshold (m)", 0.1, 15.0, 2.5, step=0.1)
    threshold_mode = st.radio(
        "Threshold mode",
        ["Single Hcrit", "Hs limit per Tp (CSV + graph)"],
        index=0,
        help="Use one constant Hcrit or import a CSV with one Hs limit per Tp bin."
    )
    stat = st.selectbox(
        "Statistic:",
        [
            "Mean Hs (m)",
            "Mean Tp (s)",
            "Hs P50 (m)",
            "Hs P90 (m)",
            "Hs P95 (m)",
            "P(Hs > Hcrit) (%)",
            "Operability (% time Hs ≤ Hcrit)"
        ]
    )

    # -----------------------------
    # Coastline cleanup controls
    # -----------------------------
    st.subheader("Coastline cleanup")
    fix_coast = st.checkbox(
        "Fix coast artifacts (near‑zero fill)", True,
        help="Replace 0/near‑0 coastal cells with the nearest valid ocean value before plotting."
    )
    fill_method = st.radio(
        "Extrapolation method",
        ["Nearest (griddata)", "Nearest (distance transform)"],
        index=1,
        help="Both are nearest‑ocean fills. The distance transform is often the cleanest."
    )
    zero_epsilon = st.number_input(
        "Zero threshold ε (treated invalid if value ≤ ε)",
        min_value=0.0, max_value=1.0, value=0.010, step=0.005, format="%.3f",
        help="Cells ≤ ε are considered contaminated and are replaced by nearest ocean values."
    )
    last_chance_fill = st.checkbox(
        "Force‑fill any remaining holes (global nearest)", True,
        help="If any NaNs remain anywhere, fill them from the nearest valid neighbour."
    )

    # Land‑mask behaviour
    simplified_mask = st.checkbox(
        "Use simplified land mask (110 m) — recommended for 0.5° grids", True,
        help="Keeps only large land masses; lets extrapolated ocean fill narrow straits/small islands."
    )
    mask_lakes = st.checkbox(
        "Mask lakes", False,
        help="If OFF, inland seas/lakes keep ocean‑like values after extrapolation."
    )
    coast_buffer = st.number_input(
        "Coast mask buffer (deg, negative shrinks)", value=-0.12, step=0.02, min_value=-0.50, max_value=0.50,
        help="Slightly shrink land polygons so coastal ocean pixels are not masked. Try −0.12°."
    )

    if fix_coast and not SCIPY_OK:
        st.warning("SciPy not available; coastline fix will be skipped. Install 'scipy' to enable.")

    st.subheader("Debug")
    show_debug = st.checkbox("Show debug", False)

# -----------------------------
# Region extents (deg)
# -----------------------------
REGION_EXTENTS = {
    "North Sea": [-13, 35, 52, 76],
    "Mediterranean": [-10.5, 40.5, 29.75, 46.25],
}

base_cmap = "turbo"
clip_pct_robust = 99.6

# -----------------------------
# Points of Interest (North Sea only)
# -----------------------------
POIS_NS = [
    {"name": "Ekofisk", "nr": 1, "lat": 56.5333, "lon": 3.2000},
    {"name": "Ula", "nr": 2, "lat": 57.1000, "lon": 2.8333},
    {"name": "Sleipner", "nr": 3, "lat": 58.3667, "lon": 1.9000},
    {"name": "Alvheim", "nr": 4, "lat": 59.5667, "lon": 1.9667},
    {"name": "Oseberg", "nr": 5, "lat": 60.5000, "lon": 2.8333},
    {"name": "Knarr", "nr": 6, "lat": 61.8833, "lon": 3.8333},
    {"name": "Ormen Lange", "nr": 7, "lat": 63.2500, "lon": 5.0000},
    {"name": "Skarv", "nr": 8, "lat": 65.7500, "lon": 7.6667},
    {"name": "Aasta Hansteen", "nr": 9, "lat": 67.0000, "lon": 8.0000},
    {"name": "Johan Castberg", "nr": 10, "lat": 72.0000, "lon": 22.5000},
]

# -----------------------------
# Strict, cache-safe dataset loading
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_metocean_cached(path: str, cache_key: str) -> xr.Dataset:
    _ = cache_key
    return xr.open_dataset(path)

def load_metocean_strict(path: str, expected_region: str) -> xr.Dataset:
    ds = load_metocean_cached(path, cache_key=path)

    if ("lon3_edges" not in ds) and ("lon_edges" in ds):
        ds = ds.rename({"lon_edges": "lon3_edges"})
    if ("lat3_edges" not in ds) and ("lat_edges" in ds):
        ds = ds.rename({"lat_edges": "lat3_edges"})

    lon_edges = ds.get("lon3_edges")
    lat_edges = ds.get("lat3_edges")
    if (lon_edges is None) or (lat_edges is None):
        st.error("Dataset is missing longitude/latitude edge arrays.")
        st.stop()

    lon_c = unwrap_lon_centers_from_edges(lon_edges.values)
    lat_c = bin_centers(lat_edges.values)
    got = dict(lon_min=float(np.nanmin(lon_c)), lon_max=float(np.nanmax(lon_c)),
               lat_min=float(np.nanmin(lat_c)),  lat_max=float(np.nanmax(lat_c)))
    expected = {
        "North Sea":      dict(lon_min=-13.5, lon_max= 35.5, lat_min=52.0, lat_max=76.5),
        "Mediterranean":  dict(lon_min=-10.5, lon_max= 40.5, lat_min=29.5, lat_max=46.5),
        "None": None,
    }
    ok = True
    if expected_region in expected and expected[expected_region] is not None:
        E = expected[expected_region]
        ok = (E["lon_min"] <= got["lon_min"] <= got["lon_max"] <= E["lon_max"]) and \
             (E["lat_min"] <= got["lat_min"] <= got["lat_max"] <= E["lat_max"])

    if not ok:
        st.warning(f"Loaded dataset bounds {got} do not match expected '{expected_region}'. Clearing cache and reloading…")
        st.cache_resource.clear()
        ds = xr.open_dataset(path)  # uncached reload
    return ds

# Dataset choice
use_zoom = zoom_region != "None"
path = REGIONAL_DATA_PATHS[zoom_region] if use_zoom else GLOBAL_DATA_PATH
if not os.path.exists(path):
    st.error(f"File not found: {path}")
    st.stop()

ds = load_metocean_strict(path, expected_region=zoom_region if use_zoom else "None")
st.sidebar.caption(
    "Global: **metocean_monthclim.nc (3°)** • "
    + (f"Zoom '{zoom_region}': **{os.path.basename(path)} (0.5°)**" if use_zoom else "No zoom dataset selected")
)

# Ensure required variables exist
for k in ["prob", "hs_edges", "tp_edges", "lat3_edges", "lon3_edges"]:
    if k not in ds:
        st.error(f"Dataset missing '{k}'")
        st.stop()

# -----------------------------
# Read edges / centers
# -----------------------------
hs_edges = ds["hs_edges"].values
tp_edges = ds["tp_edges"].values
lat_edges = ds["lat3_edges"].values
lon_edges = ds["lon3_edges"].values

units = str(ds["hs_edges"].attrs.get("units","")).lower()
if "cm" in units or (np.nanmax(hs_edges) > 50 and "m" not in units):
    hs_edges = hs_edges / 100.0

hs_c = bin_centers(hs_edges)
tp_c = bin_centers(tp_edges)
lat_c_unsorted = bin_centers(lat_edges)

if show_debug:
    st.write("Source file actually loaded:", os.path.basename(path))
    lon_c_dbg = unwrap_lon_centers_from_edges(lon_edges)
    lat_c_dbg = bin_centers(lat_edges)
    st.write("Lon centers range:", float(np.nanmin(lon_c_dbg)), "→", float(np.nanmax(lon_c_dbg)))
    st.write("Lat centers range:", float(np.nanmin(lat_c_dbg)), "→", float(np.nanmax(lat_c_dbg)))

# -----------------------------
# Probability field (month/annual)
# -----------------------------
if agg == "By month":
    prob = ds["prob"].sel(month=label_to_idx[chosen_label])
    title_suffix = f" — {chosen_label}"
else:
    prob = ds["prob"].sum(dim="month")
    title_suffix = " — Annual"

_tot_before = prob.sum(dim=("hs_bin","tp_bin"))
prob = normalize_pdf(prob)  # hs_bin, tp_bin, lat3_bin, lon3_bin

# -----------------------------
# Per‑Tp Hs limit: CSV import + graph
# -----------------------------
def init_per_tp_limits(default_val: float, tp_centers: np.ndarray):
    key = "hs_per_tp_limits"
    if (key not in st.session_state) or (len(st.session_state[key]) != len(tp_centers)):
        st.session_state[key] = [default_val] * len(tp_centers)
    return key

if threshold_mode == "Hs limit per Tp (CSV + graph)":
    limits_key = init_per_tp_limits(Hcrit, tp_c)
    st.subheader("Hs limit per Tp — curve")
    up = st.file_uploader("Import CSV (columns: 'Tp (s)', 'Hs_limit (m)')", type=["csv"], key="hs_csv_upload")
    if up is not None:
        try:
            df_in = pd.read_csv(up)
        except UnicodeDecodeError:
            df_in = pd.read_csv(up, encoding="latin-1")

        def find_col(candidates, cols):
            low = [c.lower().strip() for c in cols]
            for cand in candidates:
                if cand.lower().strip() in low:
                    return cols[low.index(cand.lower().strip())]
            return None

        tp_col = find_col(["tp (s)", "tp", "tp_s"], df_in.columns)
        hs_col = find_col(["hs_limit (m)", "hs limit (m)", "hs_limit", "hs (m)", "hs"], df_in.columns)

        if (tp_col is None) or (hs_col is None):
            st.error("CSV must contain columns 'Tp (s)' and 'Hs_limit (m)'.")
        else:
            tp_in = pd.to_numeric(df_in[tp_col], errors="coerce").astype(float).values
            hs_in = pd.to_numeric(df_in[hs_col], errors="coerce").astype(float).values
            mask = np.isfinite(tp_in) & np.isfinite(hs_in)
            tp_in, hs_in = tp_in[mask], hs_in[mask]
            if tp_in.size < 2:
                st.error("CSV must provide at least two valid Tp rows.")
            else:
                order = np.argsort(tp_in)
                tp_in, hs_in = tp_in[order], hs_in[order]
                hs_interp = np.interp(tp_c, tp_in, hs_in, left=hs_in[0], right=hs_in[-1])
                hs_interp = np.clip(np.round(hs_interp, 1), 0.0, 15.0)
                st.session_state[limits_key] = hs_interp.tolist()
                st.success(f"Imported {tp_in.size} rows → mapped to {len(tp_c)} Tp bins.")

    hs_limit_curve = np.array(st.session_state["hs_per_tp_limits"], dtype=float)
    hs_limit_curve = np.clip(np.round(hs_limit_curve, 1), 0.0, 15.0)
    st.session_state[limits_key] = hs_limit_curve.tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tp_c, y=hs_limit_curve, mode="lines+markers",
        line=dict(color="#1f77b4", width=2), marker=dict(size=6, color="#1f77b4")
    ))
    fig.update_layout(
        height=220, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Tp (s)", yaxis_title="Hs limit (m)",
        template="plotly_white",
        xaxis=dict(tickmode="linear", dtick=1),
        yaxis=dict(range=[0, max(3.0, float(np.nanmax(hs_limit_curve)) + 0.5)], dtick=0.5),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
else:
    hs_limit_curve = None

# -----------------------------
# Statistics
# -----------------------------
hs_w = xr.DataArray(hs_c, dims=["hs_bin"])
tp_w = xr.DataArray(tp_c, dims=["tp_bin"])

mean_hs = (prob*hs_w).sum(dim=("hs_bin","tp_bin"))
mean_tp = (prob*tp_w).sum(dim=("hs_bin","tp_bin"))

hs_pdf = prob.sum(dim="tp_bin")
hs_cdf = hs_pdf.cumsum(dim="hs_bin")
hs_p50 = percentile_from_cdf(hs_cdf, hs_c, 0.50)
hs_p90 = percentile_from_cdf(hs_cdf, hs_c, 0.90)
hs_p95 = percentile_from_cdf(hs_cdf, hs_c, 0.95)

# >>> MOD: within‑bin interpolation for Exceedance / Operability
if hs_limit_curve is None:
    # --- Precise P(Hs <= Hcrit) with within-bin interpolation (single Hcrit) ---
    # prob dims: (hs_bin, tp_bin, lat3_bin, lon3_bin), normalized to sum=1
    k = int(np.searchsorted(hs_edges, Hcrit, side="right") - 1)
    k = np.clip(k, 0, len(hs_edges) - 2)

    hs_lo = float(hs_edges[k])
    hs_hi = float(hs_edges[k + 1])
    bin_w = max(hs_hi - hs_lo, 1e-12)
    frac = float(np.clip((Hcrit - hs_lo) / bin_w, 0.0, 1.0))

    # Sum probability of all full bins entirely below Hcrit, across Tp
    p_full = prob.isel(hs_bin=slice(0, k)).sum(dim=("hs_bin", "tp_bin"))
    # Partial contribution from the crossing bin k
    p_part = prob.isel(hs_bin=k).sum(dim="tp_bin") * frac

    p_oper = (p_full + p_part) * 100.0
    p_exceed = 100.0 - p_oper
else:
    # --- Precise P(Hs <= Hs_limit(Tp)) with within-bin interpolation (per-Tp curve) ---
    Tp_limit_1D = xr.DataArray(hs_limit_curve, dims=["tp_bin"])

    HS_LO = xr.DataArray(hs_edges[:-1], dims=["hs_bin"]).broadcast_like(prob)
    HS_HI = xr.DataArray(hs_edges[1:],  dims=["hs_bin"]).broadcast_like(prob)
    HS_LIM = Tp_limit_1D.broadcast_like(prob)

    below = HS_HI <= HS_LIM
    above = HS_LO >= HS_LIM
    cross = (~below) & (~above)

    # Full contribution from bins entirely below the limit
    p_full2 = xr.where(below, prob, 0.0).sum(dim=("hs_bin","tp_bin"))

    # Partial contribution from crossing bins (uniform within-bin)
    denom = xr.where((HS_HI - HS_LO) > 0, HS_HI - HS_LO, 1e-12)
    frac2 = xr.where(cross, (HS_LIM - HS_LO) / denom, 0.0).clip(0.0, 1.0)
    p_part2 = (prob * frac2).sum(dim=("hs_bin","tp_bin"))

    p_oper = (p_full2 + p_part2) * 100.0
    p_exceed = 100.0 - p_oper
# <<< MOD end

# Select field to display
if stat == "Mean Hs (m)":
    field = mean_hs; label = "Mean Hs (m)" + title_suffix
elif stat == "Mean Tp (s)":
    field = mean_tp; label = "Mean Tp (s)" + title_suffix
elif stat == "Hs P50 (m)":
    field = hs_p50; label = "Hs P50 (m)" + title_suffix
elif stat == "Hs P90 (m)":
    field = hs_p90; label = "Hs P90 (m)" + title_suffix
elif stat == "Hs P95 (m)":
    field = hs_p95; label = "Hs P95 (m)" + title_suffix
elif stat.startswith("P(Hs"):
    field = p_exceed
    label = (f"P(Hs > {Hcrit:.1f} m) (%)" if hs_limit_curve is None else "P(Hs > Hs_limit(Tp)) (%)") + title_suffix
else:
    field = p_oper
    label = (f"Operability (% time Hs ≤ {Hcrit:.1f} m)" if hs_limit_curve is None else
             "Operability (% time Hs ≤ Hs_limit(Tp))") + title_suffix

# -----------------------------
# Prepare 2D field (sorted lon/lat)
# -----------------------------
field2d = field.transpose("lat3_bin","lon3_bin").values
field2d, latp, lonp, flip_lat, lon_sort_idx, lon_inv = to_sorted_lon_lat(
    field2d, lat_c_unsorted, lon_edges
)

# -----------------------------
# Projection factory
# -----------------------------
def get_zoom_projection(name: str):
    if name.startswith("PlateCarree"):
        return ccrs.PlateCarree()
    if name == "Mercator":
        return ccrs.Mercator(central_longitude=15, min_latitude=20, max_latitude=60)
    if name == "Lambert Conformal":
        return ccrs.LambertConformal(
            central_longitude=15,
            central_latitude=45 if (zoom_region == "Mediterranean") else 60,
            standard_parallels=(30, 45) if (zoom_region == "Mediterranean") else (50, 65)
        )
    return ccrs.PlateCarree()

# -----------------------------
# Color scaling & levels (robust)
# -----------------------------
def region_slice(arr2d, lons, lats, extent):
    lon_min, lon_max, lat_min, lat_max = extent
    j = (lons >= lon_min) & (lons <= lon_max)
    i = (lats >= lat_min) & (lats <= lat_max)
    if not np.any(i) or not np.any(j):
        return arr2d, False
    sub = arr2d[np.ix_(i, j)]
    if not np.isfinite(sub).any():
        return arr2d, False
    return sub, True

def safe_minmax(a):
    vmin = np.nanmin(a); vmax = np.nanmax(a)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return 0.0, 1.0
    return float(vmin), float(vmax)

def prep_levels(arr, label, prefer_ticks_from=None, zoom=False):
    base = prefer_ticks_from if prefer_ticks_from is not None else arr
    vmin, vmax = safe_minmax(base)

    if "P(Hs" in label or "Operability" in label:
        return pct_shading(), pct_ticks(), pct_ticks()

    if label.startswith("Mean Tp"):
        if zoom:
            contours = np.arange(math.floor(vmin/0.5)*0.5, math.ceil(vmax/0.5)*0.5 + 1e-9, 0.5)
            filled = tp_shading(base)
            return filled, contours, contours
        ticks = tp_ticks(1.0, vmin, vmax)
        return tp_shading(base), ticks, ticks

    if is_hs_quantity(label):
        if zoom:
            filled = np.arange(math.floor(vmin/0.1)*0.1, math.ceil(vmax/0.1)*0.1 + 1e-9, 0.1)
            contours = np.arange(math.floor(vmin/0.2)*0.2, math.ceil(vmax/0.2)*0.2 + 1e-9, 0.2)
            return filled, contours, contours
        ticks = hs_ticks(0.5, vmin, vmax)
        return hs_shading(base), ticks, ticks

    lev = auto_levels(base, 50)
    return lev, lev, None

is_percent_metric = ("P(Hs" in label) or ("Operability" in label)
hi_global = 100.0 if is_percent_metric else np.nanpercentile(field2d, clip_pct_robust)

if use_zoom and not is_percent_metric:
    region, ok = region_slice(field2d, lonp, latp, REGION_EXTENTS[zoom_region])
    if ok:
        hi_zoom = np.nanpercentile(region, clip_pct_robust)
        hi_use = hi_zoom
        ticks_base = np.clip(region, None, hi_zoom)
    else:
        hi_use = hi_global
        ticks_base = np.clip(field2d, None, hi_global)
else:
    hi_use = hi_global
    ticks_base = np.clip(field2d, None, hi_global)

arr_plot = np.clip(field2d, None, hi_use)
filled_levels, contour_levels, cbar_ticks = prep_levels(
    arr_plot, label, prefer_ticks_from=ticks_base, zoom=use_zoom
)
cmap_use = base_cmap + "_r" if "Operability" in label else base_cmap

# -----------------------------
# POI drawer (North Sea only)
# -----------------------------
def draw_pois(ax, pois):
    lons = [p["lon"] for p in pois]
    lats = [p["lat"] for p in pois]
    ax.scatter(lons, lats, s=28, c="black", marker="o",
               transform=ccrs.PlateCarree(), zorder=20)
    offsets = {
        1:(0.12,0.10), 2:(0.12,0.10), 3:(0.14,0.10), 4:(0.14,0.12), 5:(0.14,0.12),
        6:(0.12,0.12), 7:(0.12,0.12), 8:(0.12,0.12), 9:(0.12,0.12), 10:(0.14,0.12)
    }
    halo = [patheffects.withStroke(linewidth=2.2, foreground="white", alpha=0.9)]
    for p in pois:
        dx, dy = offsets.get(p["nr"], (0.12, 0.12))
        ax.text(p["lon"] + dx, p["lat"] + dy, f'{p["nr"]} {p["name"]}',
                transform=ccrs.PlateCarree(), fontsize=7, color="black",
                zorder=21, path_effects=halo)

# -----------------------------
# Plot function
# -----------------------------
def plot_map(lon_c, lat_c, arr2d, title, filled, contours, cmap, ticks,
             use_zoom: bool, zoom_proj, region_name: str):
    ax_proj = zoom_proj if use_zoom else ccrs.PlateCarree()
    fig = plt.figure(figsize=(15, 6), dpi=(200 if use_zoom else 150))
    ax = plt.axes(projection=ax_proj)

    cf = ax.contourf(
        lon_c, lat_c, arr2d,
        levels=filled, cmap=cmap, extend="both",
        transform=ccrs.PlateCarree(), zorder=1, antialiased=True
    )
    try:
        cs = ax.contour(
            lon_c, lat_c, arr2d, levels=contours, colors="black",
            linewidths=0.45 if use_zoom else 0.4,
            transform=ccrs.PlateCarree(), zorder=2
        )
        ax.figure.canvas.draw()
        ax.clabel(cs, fontsize=6, inline=True, inline_spacing=(1 if use_zoom else 6),
                  fmt="%g", manual=False, rightside_up=True)
    except Exception:
        pass

    feature_scale = "10m" if use_zoom else "110m"
    ax.add_feature(cfeature.LAND.with_scale(feature_scale),
                   facecolor="lightgray", edgecolor="none", zorder=10)
    ax.add_feature(cfeature.COASTLINE.with_scale(feature_scale),
                   linewidth=0.7 if use_zoom else 0.4, zorder=11)
    ax.add_feature(cfeature.BORDERS.with_scale(feature_scale),
                   linewidth=0.3 if use_zoom else 0.2, zorder=12)

    if use_zoom:
        ax.set_extent(REGION_EXTENTS[region_name], crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    if use_zoom and region_name == "North Sea":
        draw_pois(ax, POIS_NS)

    if show_grid_points:
        Lon2D, Lat2D = np.meshgrid(lon_c, lat_c)
        ax.scatter(Lon2D.ravel(), Lat2D.ravel(), s=6, color="gray", alpha=0.6,
                   transform=ccrs.PlateCarree(), zorder=3)

    cb = plt.colorbar(cf, ax=ax, shrink=0.75, aspect=30, pad=0.01, ticks=ticks)
    cb.set_label(title)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title)
    plt.subplots_adjust(left=0.02, right=0.97, top=0.93, bottom=0.06)
    st.pyplot(fig, use_container_width=True)

# -----------------------------
# Render (robust fill + buffered mask)
# -----------------------------
Lon2D, Lat2D = np.meshgrid(lonp, latp)

# 1) RELAXED water domain (110m, no lakes) — allows narrow straits to be ocean
water_relaxed = ~land_mask_bool(Lon2D, Lat2D, resolution='110m', include_lakes=False, buffer_deg=-0.12)

arr_to_plot = arr_plot.copy()
filled_cells_stageA = 0
filled_cells_stageB = 0

if fix_coast and SCIPY_OK:
    # Stage A: nearest fill inside relaxed ocean domain only
    valid   = np.isfinite(arr_to_plot) & (arr_to_plot > float(zero_epsilon)) & water_relaxed
    invalid = (~valid) & water_relaxed
    if np.any(valid) and np.any(invalid):
        if fill_method.startswith("Nearest (griddata)"):
            arr_filled = arr_to_plot.copy()
            arr_filled[invalid] = griddata(
                (Lon2D[valid], Lat2D[valid]),
                arr_to_plot[valid],
                (Lon2D[invalid], Lat2D[invalid]),
                method='nearest'
            )
        else:
            # Distance-transform: nearest VALID inside ocean domain
            edt_base = ~valid  # distance to nearest False (valid)
            _, idx = distance_transform_edt(edt_base, return_indices=True)
            arr_filled = arr_to_plot.copy()
            arr_filled[invalid] = arr_to_plot[idx[0][invalid], idx[1][invalid]]
        arr_to_plot = arr_filled
        filled_cells_stageA = int(invalid.sum())

    # Stage B (optional): fill any remaining holes ANYWHERE (global nearest)
    if last_chance_fill:
        nanmask = ~np.isfinite(arr_to_plot)
        if np.any(nanmask):
            valid2 = ~nanmask
            if np.any(valid2):
                _, idx2 = distance_transform_edt(~valid2, return_indices=True)
                arr_to_plot[nanmask] = arr_to_plot[idx2[0][nanmask], idx2[1][nanmask]]
            filled_cells_stageB = int(nanmask.sum())

# 2) FINAL LAND MASK with buffer (shrink land so coasts are preserved)
mask_res_use = '110m' if simplified_mask else ('10m' if use_zoom else '110m')
arr_plot_masked = mask_land_to_nan(
    Lon2D, Lat2D, arr_to_plot, resolution=mask_res_use,
    include_lakes=bool(mask_lakes), buffer_deg=float(coast_buffer)
)

# Plot
plot_map(
    lonp, latp, arr_plot_masked, label,
    filled_levels, contour_levels, cmap_use, cbar_ticks,
    use_zoom=use_zoom,
    zoom_proj=get_zoom_projection(zoom_proj_name),
    region_name=zoom_region if use_zoom else "None"
)

# -----------------------------
# Table under the map (reference only)
# -----------------------------
if threshold_mode == "Hs limit per Tp (CSV + graph)":
    with st.expander("Show Hs limit per Tp (table)", expanded=False):
        df_view = pd.DataFrame({"Tp (s)": tp_c, "Hs_limit (m)": st.session_state["hs_per_tp_limits"]})
        st.dataframe(df_view.style.format({"Tp (s)": "{:.1f}", "Hs_limit (m)": "{:.1f}"}), use_container_width=True)

# -----------------------------
# Debug extras
# -----------------------------
if show_debug:
    finite_pct = 100.0 * np.isfinite(arr_plot_masked).sum() / max(1, arr_plot_masked.size)
    st.write(
        "Totals BEFORE normalization:",
        float(_tot_before.min()), float(_tot_before.max())
    )
    st.write(
        "Finite values in plotted array:", f"{finite_pct:.1f}%",
        "\nZoomed:", bool(use_zoom),
        "\nZoom projection:", zoom_proj_name,
        "\nSource:", os.path.basename(path),
        "\nThreshold mode:", threshold_mode,
        "\nCoastline fix:", bool(fix_coast),
        "\nMethod:", fill_method if SCIPY_OK else "SciPy missing",
        "\nFilled (Stage A, relaxed ocean) cells:", filled_cells_stageA,
        "\nFilled (Stage B, global) cells:", filled_cells_stageB,
        "\nFinal mask resolution:", mask_res_use,
        "\nMask lakes:", bool(mask_lakes),
        "\nCoast buffer (deg):", float(coast_buffer)
    )
    # Sanity: operability + exceedance ≈ 100
    resid = (p_oper + p_exceed) - 100.0
    st.write("Operability + Exceedance residual (min/max):",
             float(resid.min().values), float(resid.max().values))