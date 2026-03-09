import math
import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import xarray as xr


# ===========================================================
# PAGE SETUP
# ===========================================================
st.set_page_config(page_title="🌍 Metocean Explorer (safe)", layout="wide")
st.header("🌍 Global wave statistics")
st.caption("Trial page with simpler annual plotting and safer figure handling.")


# ===========================================================
# DATASET LOCATION
# ===========================================================
THIS_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    THIS_DIR / "metocean_monthclim.nc",
    THIS_DIR.parent / "metocean_monthclim.nc",
    Path.cwd() / "metocean_monthclim.nc",
]


def find_dataset_path() -> Path | None:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    return None


DATA_PATH = find_dataset_path()
if DATA_PATH is None:
    st.error(
        "Could not find metocean_monthclim.nc. Put the NetCDF file either in the same folder as this page or one level above it."
    )
    st.stop()


# ===========================================================
# HELPERS
# ===========================================================
@st.cache_resource(show_spinner=False)
def load_metocean(path_str: str) -> xr.Dataset:
    return xr.open_dataset(path_str)


@st.cache_data(show_spinner=False)
def get_static_arrays(path_str: str):
    ds_local = load_metocean(path_str)

    required = ["prob", "hs_edges", "tp_edges", "lat3_edges", "lon3_edges"]
    missing = [k for k in required if k not in ds_local]
    if missing:
        raise KeyError(f"Dataset missing variables: {missing}")

    hs_edges_local = ds_local["hs_edges"].values.astype(float)
    tp_edges_local = ds_local["tp_edges"].values.astype(float)
    lat_edges_local = ds_local["lat3_edges"].values.astype(float)
    lon_edges_local = ds_local["lon3_edges"].values.astype(float)

    units = str(ds_local["hs_edges"].attrs.get("units", "")).lower()
    if "cm" in units or (np.nanmax(hs_edges_local) > 50 and "m" not in units):
        hs_edges_local = hs_edges_local / 100.0

    return hs_edges_local, tp_edges_local, lat_edges_local, lon_edges_local



def bin_centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])



def unwrap_lon_centers_from_edges(lon_edges: np.ndarray) -> np.ndarray:
    lon_c = bin_centers(lon_edges)
    if np.nanmax(lon_edges) > 180:
        lon_c = ((lon_c + 180.0) % 360.0) - 180.0
    return lon_c



def to_sorted_lon_lat(field2d: np.ndarray, lat_c: np.ndarray, lon_edges: np.ndarray):
    if lat_c[0] > lat_c[-1]:
        field2d = field2d[::-1, :]
        lat_c = lat_c[::-1]

    lon_unsorted = unwrap_lon_centers_from_edges(lon_edges)
    lon_sort_idx = np.argsort(lon_unsorted)
    lon_sorted = lon_unsorted[lon_sort_idx]
    field2d_sorted = field2d[:, lon_sort_idx]
    return field2d_sorted, lat_c, lon_sorted



def normalize_pdf(prob: xr.DataArray) -> xr.DataArray:
    tot = prob.sum(dim=("hs_bin", "tp_bin"))
    return xr.where(tot > 0, prob / tot, 0.0)



def percentile_from_cdf(cdf: xr.DataArray, centers: np.ndarray, q: float) -> xr.DataArray:
    idx_hi = (cdf >= q).argmax(dim="hs_bin")
    idx_lo = xr.where(idx_hi > 0, idx_hi - 1, 0)

    c_lo = cdf.isel(hs_bin=idx_lo)
    c_hi = cdf.isel(hs_bin=idx_hi)

    cen = xr.DataArray(centers, dims=["hs_bin"])
    h_lo = cen.isel(hs_bin=idx_lo)
    h_hi = cen.isel(hs_bin=idx_hi)

    denom = xr.where((c_hi - c_lo) > 0, c_hi - c_lo, 1.0)
    w = (q - c_lo) / denom
    return h_lo + w * (h_hi - h_lo)



def hs_ticks(vmin: float, vmax: float, step: float = 0.5) -> np.ndarray:
    lo = math.floor(vmin / step) * step
    hi = math.ceil(vmax / step) * step
    return np.arange(lo, hi + 0.5 * step, step)



def tp_ticks(vmin: float, vmax: float, step: float = 1.0) -> np.ndarray:
    lo = math.floor(vmin / step) * step
    hi = math.ceil(vmax / step) * step
    return np.arange(lo, hi + 0.5 * step, step)



def safe_minmax(arr: np.ndarray, fallback: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return fallback
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return fallback
    return vmin, vmax



def build_levels(arr: np.ndarray, stat_label: str, is_percent: bool):
    vmin, vmax = safe_minmax(arr)

    if is_percent:
        filled = np.linspace(0.0, 100.0, 31)
        contour = np.arange(0.0, 101.0, 10.0)
        ticks = contour
        cmap = "turbo_r" if "Operability" in stat_label else "turbo"
        return filled, contour, ticks, cmap

    if stat_label == "Mean Tp (s)":
        ticks = tp_ticks(vmin, vmax, step=1.0)
        filled = np.linspace(vmin, vmax, 36)
        contour = np.linspace(vmin, vmax, 10)
        return filled, contour, ticks, "turbo"

    if "Hs" in stat_label:
        ticks = hs_ticks(vmin, vmax, step=0.5)
        filled = np.linspace(vmin, vmax, 36)
        contour = np.linspace(vmin, vmax, 10)
        return filled, contour, ticks, "turbo"

    filled = np.linspace(vmin, vmax, 36)
    contour = np.linspace(vmin, vmax, 10)
    return filled, contour, None, "turbo"



def plot_global_map(
    lon_c: np.ndarray,
    lat_c: np.ndarray,
    arr2d: np.ndarray,
    title: str,
    filled: np.ndarray,
    contours: np.ndarray,
    cmap: str,
    ticks: np.ndarray | None,
    show_contour_labels: bool,
):
    fig = plt.figure(figsize=(15, 6), dpi=140)
    ax = plt.axes(projection=ccrs.PlateCarree())

    cf = ax.contourf(
        lon_c,
        lat_c,
        arr2d,
        levels=filled,
        cmap=cmap,
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    try:
        cs = ax.contour(
            lon_c,
            lat_c,
            arr2d,
            levels=contours,
            colors="black",
            linewidths=0.35,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
        if show_contour_labels:
            ax.clabel(cs, fontsize=6, inline=True, fmt="%g")
    except Exception:
        pass

    ax.add_feature(
        cfeature.LAND.with_scale("110m"),
        facecolor="lightgray",
        edgecolor="none",
        zorder=10,
    )
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.8, zorder=11)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.3, zorder=12)
    ax.set_global()

    cb = plt.colorbar(cf, ax=ax, shrink=0.75, aspect=30, pad=0.01, ticks=ticks)
    cb.set_label(title)
    cb.ax.tick_params(labelsize=8)

    ax.set_title(title)
    plt.subplots_adjust(left=0.02, right=0.97, top=0.93, bottom=0.06)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ===========================================================
# SIDEBAR
# ===========================================================
with st.sidebar:
    st.subheader("Data")
    st.caption(f"Using dataset: {DATA_PATH}")

    st.subheader("Aggregation")
    agg = st.radio("Use:", ["By month", "Annual"], horizontal=True)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_vals = np.arange(1, 13)
    chosen_label = st.selectbox("Month", month_labels, index=4, disabled=(agg == "Annual"))
    label_to_idx = dict(zip(month_labels, month_vals))

    st.subheader("Metric")
    hcrit = st.number_input("Hs threshold (m)", min_value=0.1, max_value=15.0, value=2.5, step=0.1)
    stat = st.selectbox(
        "Statistic:",
        [
            "Mean Hs (m)",
            "Mean Tp (s)",
            "Hs P50 (m)",
            "Hs P90 (m)",
            "Hs P95 (m)",
            "P(Hs > Hcrit) (%)",
            "Operability (% time Hs ≤ Hcrit)",
        ],
    )

    st.subheader("Plot options")
    label_monthly_contours = st.checkbox("Label contour lines for monthly maps", value=True)
    show_debug = st.checkbox("Show debug", value=False)


# ===========================================================
# LOAD DATA
# ===========================================================
try:
    ds = load_metocean(str(DATA_PATH))
    hs_edges, tp_edges, lat_edges, lon_edges = get_static_arrays(str(DATA_PATH))
except Exception as exc:
    st.error(f"Failed to open metocean dataset: {exc}")
    st.stop()

hs_c = bin_centers(hs_edges)
tp_c = bin_centers(tp_edges)
lat_c_unsorted = bin_centers(lat_edges)

if agg == "By month":
    prob = ds["prob"].sel(month=label_to_idx[chosen_label])
    title_suffix = f" — {chosen_label}"
else:
    prob = ds["prob"].sum(dim="month")
    title_suffix = " — Annual"

prob_total_before = prob.sum(dim=("hs_bin", "tp_bin"))
prob = normalize_pdf(prob)


# ===========================================================
# STATISTICS
# ===========================================================
hs_w = xr.DataArray(hs_c, dims=["hs_bin"])
tp_w = xr.DataArray(tp_c, dims=["tp_bin"])

mean_hs = (prob * hs_w).sum(dim=("hs_bin", "tp_bin"))
mean_tp = (prob * tp_w).sum(dim=("hs_bin", "tp_bin"))

hs_pdf = prob.sum(dim="tp_bin")
hs_cdf = hs_pdf.cumsum(dim="hs_bin")

hs_p50 = percentile_from_cdf(hs_cdf, hs_c, 0.50)
hs_p90 = percentile_from_cdf(hs_cdf, hs_c, 0.90)
hs_p95 = percentile_from_cdf(hs_cdf, hs_c, 0.95)

mask_exceed = xr.DataArray((hs_c > hcrit).astype(float), dims=["hs_bin"])
p_exceed = (hs_pdf * mask_exceed).sum(dim="hs_bin")
p_below = 1.0 - p_exceed

if stat == "Mean Hs (m)":
    field = mean_hs
    label = "Mean Hs (m)" + title_suffix
    is_percent = False
elif stat == "Mean Tp (s)":
    field = mean_tp
    label = "Mean Tp (s)" + title_suffix
    is_percent = False
elif stat == "Hs P50 (m)":
    field = hs_p50
    label = "Hs P50 (m)" + title_suffix
    is_percent = False
elif stat == "Hs P90 (m)":
    field = hs_p90
    label = "Hs P90 (m)" + title_suffix
    is_percent = False
elif stat == "Hs P95 (m)":
    field = hs_p95
    label = "Hs P95 (m)" + title_suffix
    is_percent = False
elif stat == "P(Hs > Hcrit) (%)":
    field = 100.0 * p_exceed
    label = f"P(Hs > {hcrit:.1f} m) (%)" + title_suffix
    is_percent = True
else:
    field = 100.0 * p_below
    label = f"Operability (% time Hs ≤ {hcrit:.1f} m)" + title_suffix
    is_percent = True


# ===========================================================
# 2D FIELD PREP
# ===========================================================
field2d = field.transpose("lat3_bin", "lon3_bin").values.astype(float)
field2d, latp, lonp = to_sorted_lon_lat(field2d, lat_c_unsorted, lon_edges)

if not is_percent:
    finite = field2d[np.isfinite(field2d)]
    if finite.size > 0:
        hi = float(np.nanpercentile(finite, 99.6))
        field2d = np.clip(field2d, None, hi)

filled_levels, contour_levels, cbar_ticks, cmap_use = build_levels(field2d, stat, is_percent)
show_contour_labels = agg == "By month" and label_monthly_contours


# ===========================================================
# SUMMARY METRICS
# ===========================================================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Grid cells", f"{field2d.size:,}")
with col2:
    st.metric("Field min", f"{np.nanmin(field2d):.2f}")
with col3:
    st.metric("Field max", f"{np.nanmax(field2d):.2f}")


# ===========================================================
# PLOT
# ===========================================================
plot_global_map(
    lonp,
    latp,
    field2d,
    label,
    filled_levels,
    contour_levels,
    cmap_use,
    cbar_ticks,
    show_contour_labels,
)


# ===========================================================
# DEBUG
# ===========================================================
if show_debug:
    st.markdown("---")
    st.subheader("Debug")
    st.write("Dataset path:", str(DATA_PATH))
    st.write("prob totals before normalization:", float(prob_total_before.min()), float(prob_total_before.max()))
    st.write("mean Hs global min/max:", float(np.nanmin(mean_hs)), float(np.nanmax(mean_hs)))
    st.write("mean Tp global min/max:", float(np.nanmin(mean_tp)), float(np.nanmax(mean_tp)))
    st.write("lat ascending:", bool(latp[0] < latp[-1]))
    st.write("lon min/max:", float(np.nanmin(lonp)), float(np.nanmax(lonp)))
