# 04_Comparison.py — Single Map Difference in Operability (with contour lines)
# ---------------------------------------------------------------------------
# This page creates a single global map showing:
#     ΔOperability = Operability(B) – Operability(A)
# Additions:
#   • Contour lines + labels on top of filled contours
#   • Clean, rounded symmetric color limits for nicer ticks/labels
# Parity:
#   • PlateCarree projection only
#   • turbo colormap only
#   • Tp tick handling unchanged
#   • Same land masking as other pages
# ---------------------------------------------------------------------------

import re
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ================================
# Page setup
# ================================
st.set_page_config(layout="wide")
st.header("🌊 ΔOperability — Difference Between Two Systems")

# ================================
# Sidebar
# ================================
with st.sidebar:
    st.subheader("Data")
    nc_path = "metocean_monthclim.nc"
    st.caption("Using dataset: metocean_monthclim.nc")

    resp_file = st.file_uploader("Response CSV (TP1..TPn)", type=["csv"])

    st.subheader("Aggregation")
    agg = st.radio("Use:", ["By month", "Annual"], horizontal=True)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_vals = np.arange(1,13)
    chosen_label = st.selectbox("Month", months, index=0)
    label_to_idx = dict(zip(months, month_vals))

    st.subheader("Limits")
    lim_roll = st.number_input("Roll limit (deg)", 0.5, 45.0, 3.0)
    lim_lat  = st.number_input("Lateral limit (m/s²)", 0.01, 1.50, 0.50)
    lim_vert = st.number_input("Vertical limit (m/s²)", 0.01, 2.00, 1.00)

    st.subheader("Metric to compare")
    metric = st.selectbox(
        "Metric",
        [
            "Operability: roll ≤ limit (%)",
            "Operability: lateral ≤ limit (%)",
            "Operability: vertical ≤ limit (%)",
            "Operability: ALL limits (%)",
        ],
    )

# ================================
# Helpers
# ================================
@st.cache_resource
def load_nc(path):
    return xr.open_dataset(path)

def interp_rows(R, x_from, x_to):
    """Row-wise 1D interpolation of a 2D matrix (systems × TP)."""
    return np.vstack([np.interp(x_to, x_from, row) for row in R])

def nice_symmetric_limits(v, step=5.0):
    """
    Round symmetric range [-V, +V] to a 'nice' multiple of step (default 5).
    Returns peak_rounded, levels (61), ticks (9).
    """
    if v <= 0 or not np.isfinite(v):
        v = 1.0
    peak = float(v)
    peak_rounded = float(np.ceil(peak / step) * step)
    levels = np.linspace(-peak_rounded, +peak_rounded, 61)
    ticks  = np.linspace(-peak_rounded, +peak_rounded, 9)
    return peak_rounded, levels, ticks

# ================================
# Load metocean dataset
# ================================
ds = load_nc(nc_path)

# Extract edges
hs_edges  = ds["hs_edges"].values
tp_edges  = ds["tp_edges"].values
lat_edges = ds["lat3_edges"].values
lon_edges = ds["lon3_edges"].values

# Unit fix for Hs
units = str(ds["hs_edges"].attrs.get("units","")).lower()
if "cm" in units or (np.nanmax(hs_edges) > 50 and "m" not in units):
    hs_edges = hs_edges/100.0

# Bin centers
hs_c          = 0.5*(hs_edges[:-1] + hs_edges[1:])
tp_c          = 0.5*(tp_edges[:-1] + tp_edges[1:])
lat_c_unsorted= 0.5*(lat_edges[:-1] + lat_edges[1:])

# Select/aggregate probability and normalize to a PDF
if agg == "By month":
    prob = ds["prob"].sel(month=label_to_idx[chosen_label])
else:
    prob = ds["prob"].sum(dim="month")
tot = prob.sum(dim=("hs_bin","tp_bin"))
prob = xr.where(tot > 0, prob / tot, 0)

# ================================
# Load & parse CSV
# ================================
if resp_file is None:
    st.info("Upload a valid RAO CSV file.")
    st.stop()

df_raw = pd.read_csv(resp_file, header=None)
tp_labels = df_raw.iloc[1, 2:].tolist()

def tpkey(s):
    m = re.search(r"(\d+)$", str(s))
    return int(m.group(1)) if m else 0

tp_labels = sorted(tp_labels, key=tpkey)

# Normalize into long/wide structure
records = []
for i in range(2, len(df_raw)):
    row = df_raw.iloc[i]
    alt = row.iloc[0]
    var = row.iloc[1]
    vals = row.iloc[2:2+len(tp_labels)].tolist()
    if pd.isna(alt) and pd.isna(var):
        continue
    rec = {"Hull": str(alt) if not pd.isna(alt) else None, "Var": str(var) if not pd.isna(var) else None}
    for k, v in zip(tp_labels, vals):
        rec[k] = v
    records.append(rec)

df = pd.DataFrame(records)
df["Hull"] = df["Hull"].ffill().astype(str).str.strip()
df["Var"]  = df["Var"].astype(str).str.strip()

# Extract matrices by response variable
roll_df = df[df["Var"].str.contains("roll", case=False)]
lat_df  = df[df["Var"].str.contains("lateral", case=False)]
vert_df = df[df["Var"].str.contains("vertical", case=False)]
tp_df   = df[df["Var"].str.contains("tp", case=False)]  # optional Tp centers row

# Enforce consistent order of alternatives (use CSV order but ensure uniqueness)
alt_names = roll_df["Hull"].drop_duplicates().tolist()

def mat_from(df_alt):
    mats = []
    for alt in alt_names:
        row = df_alt[df_alt["Hull"] == alt][tp_labels]
        if row.empty:
            raise ValueError(f"Missing row for hull '{alt}' in CSV for variable '{df_alt['Var'].iloc[0]}'")
        mats.append(row.iloc[0].astype(float).to_numpy())
    return np.vstack(mats)

R_roll = mat_from(roll_df)
R_lat  = mat_from(lat_df)
R_vert = mat_from(vert_df)

# Optional Tp centers from CSV
csv_tp_sec = None
if not tp_df.empty:
    try:
        csv_tp_sec = tp_df.iloc[0][tp_labels].astype(float).to_numpy()
    except Exception:
        csv_tp_sec = None

# ================================
# TP mapping to dataset grid
# ================================
n_csv = len(tp_labels)

if len(tp_c) == n_csv:
    Rr, Rl, Rv = R_roll, R_lat, R_vert
else:
    if csv_tp_sec is not None:
        # Interp from provided CSV centers
        Rr = interp_rows(R_roll, csv_tp_sec, tp_c)
        Rl = interp_rows(R_lat , csv_tp_sec, tp_c)
        Rv = interp_rows(R_vert, csv_tp_sec, tp_c)
    else:
        # Ordinal mapping across dataset Tp range
        tpmin, tpmax = float(tp_c.min()), float(tp_c.max())
        k = np.arange(1, n_csv+1)
        x_from = tpmin + ((k - 0.5)/n_csv)*(tpmax - tpmin)
        Rr = interp_rows(R_roll, x_from, tp_c)
        Rl = interp_rows(R_lat , x_from, tp_c)
        Rv = interp_rows(R_vert, x_from, tp_c)

# ================================
# Select TWO systems
# ================================
with st.sidebar:
    st.subheader("Systems")
    if len(alt_names) < 2:
        st.error("The CSV must contain at least two systems (hulls) to compare.")
        st.stop()
    A = st.selectbox("System A", alt_names, index=0)
    B = st.selectbox("System B", alt_names, index=1 if len(alt_names) > 1 else 0)

iA = alt_names.index(A)
iB = alt_names.index(B)

RAO_roll_A = xr.DataArray(Rr[iA], dims=["tp_bin"])
RAO_lat_A  = xr.DataArray(Rl[iA], dims=["tp_bin"])
RAO_vert_A = xr.DataArray(Rv[iA], dims=["tp_bin"])

RAO_roll_B = xr.DataArray(Rr[iB], dims=["tp_bin"])
RAO_lat_B  = xr.DataArray(Rl[iB], dims=["tp_bin"])
RAO_vert_B = xr.DataArray(Rv[iB], dims=["tp_bin"])

HS = xr.DataArray(hs_c, dims=["hs_bin"])

# ================================
# Compute operability fields
# ================================
def operability(prob, RAO, limit):
    M = HS * RAO
    I = xr.where(M <= limit, 1.0, 0.0)
    return (prob * I).sum(dim=("hs_bin","tp_bin")) * 100.0

if metric == "Operability: roll ≤ limit (%)":
    Aop = operability(prob, RAO_roll_A, lim_roll)
    Bop = operability(prob, RAO_roll_B, lim_roll)
elif metric == "Operability: lateral ≤ limit (%)":
    Aop = operability(prob, RAO_lat_A, lim_lat)
    Bop = operability(prob, RAO_lat_B, lim_lat)
elif metric == "Operability: vertical ≤ limit (%)":
    Aop = operability(prob, RAO_vert_A, lim_vert)
    Bop = operability(prob, RAO_vert_B, lim_vert)
else:
    IA = xr.where(
        (HS*RAO_roll_A <= lim_roll) &
        (HS*RAO_lat_A  <= lim_lat ) &
        (HS*RAO_vert_A <= lim_vert),
        1.0, 0.0
    )
    IB = xr.where(
        (HS*RAO_roll_B <= lim_roll) &
        (HS*RAO_lat_B  <= lim_lat ) &
        (HS*RAO_vert_B <= lim_vert),
        1.0, 0.0
    )
    Aop = (prob * IA).sum(dim=("hs_bin","tp_bin")) * 100.0
    Bop = (prob * IB).sum(dim=("hs_bin","tp_bin")) * 100.0

# Difference field
D = Bop - Aop
labelD = f"ΔOperability ( {B} – {A} ) [percentage points]"

# ================================
# Sorting for plotting (lat/lon)
# ================================
def prep(arr):
    """
    Prepare 2D array and sorted lon/lat centers for proper global plotting.
    - Flip latitude if descending
    - Wrap & sort longitudes to [-180, 180)
    """
    arr2d = arr.transpose("lat3_bin","lon3_bin").values

    # Latitude may be descending — flip if needed
    lat_c = lat_c_unsorted.copy()
    if lat_c[0] > lat_c[-1]:
        arr2d = arr2d[::-1, :]
        lat_c = lat_c[::-1]

    # Longitude centers and wrap to [-180,180) if needed
    lon_c_uns = 0.5*(lon_edges[:-1] + lon_edges[1:])
    if np.nanmax(lon_edges) > 180:
        lon_c_uns = ((lon_c_uns + 180) % 360) - 180
    idx = np.argsort(lon_c_uns)
    arr2d = arr2d[:, idx]
    lon_c  = lon_c_uns[idx]
    return arr2d, lat_c, lon_c

D2d, latp, lonp = prep(D)

# Symmetric levels and ticks (rounded to clean steps)
peak = max(abs(float(np.nanmin(D2d))), abs(float(np.nanmax(D2d))))
_, levs, ticks = nice_symmetric_limits(peak, step=5.0)

# ================================
# Plot with contours
# ================================
def plot_map(lon, lat, data, title, levels, ticks):
    fig = plt.figure(figsize=(15, 6), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Filled contours
    cf = ax.contourf(
        lon, lat, data,
        levels=levels,
        cmap="turbo",
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    # Contour lines + labels (use ticks if available for clean labels)
    try:
        cs = ax.contour(
            lon, lat, data,
            levels=ticks if ticks is not None else levels,
            colors="black",
            linewidths=0.4,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
        ax.clabel(cs, fontsize=6, inline=True, fmt="%g")
    except Exception:
        pass

    # Land overlay / masking
    ax.add_feature(
        cfeature.LAND.with_scale("110m"),
        facecolor="lightgray",
        edgecolor="none",
        zorder=10,
    )
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.8, zorder=11)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.3, zorder=12)
    ax.set_global()

    # Colorbar
    cb = plt.colorbar(cf, ax=ax, shrink=0.75, aspect=30, ticks=ticks)
    cb.set_label(title)
    cb.ax.tick_params(labelsize=8)

    ax.set_title(title)
    plt.subplots_adjust(left=0.02, right=0.97, top=0.93, bottom=0.06)
    st.pyplot(fig, use_container_width=True)

# ================================
# Render
# ================================
st.subheader(labelD)
plot_map(lonp, latp, D2d, labelD, levs, ticks)

# Optional caption about TP alignment (helps users understand mapping path)
if len(tp_c) == n_csv:
    st.caption(f"TP mapping: 1:1 ({n_csv} bins).")
else:
    st.caption(
        "TP mapping: interpolated from CSV Tp centers if provided; "
        "otherwise ordinal mapping across dataset Tp range."
    )