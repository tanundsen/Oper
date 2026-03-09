# 02_MotionOperability.py — Motion-based Operability (Roll & Accelerations)
# -------------------------------------------------------------------------
# Updated per user requests:
# - PlateCarree projection only
# - turbo only (reversed for operability)
# - No shading slider (fixed 50)
# - No clip slider (fixed 99.6)
# - Tp ticks improved (1-second intervals)
# - Land masking same as original
# - Removed dataset file input box

import re
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -----------------------------------------------------------
# Page setup
# -----------------------------------------------------------
st.set_page_config(layout="wide")
st.header("⚓ Motion-based Operability — Roll & Accelerations")

# -----------------------------------------------------------
# Sidebar
# -----------------------------------------------------------
with st.sidebar:
    st.subheader("Data")
    nc_path = "metocean_monthclim.nc"
    st.caption("Using dataset: metocean_monthclim.nc")

    resp_file = st.file_uploader("Response CSV (TP1..TPn)", type=["csv"])

    st.subheader("Aggregation")
    agg = st.radio("Use:", ["By month", "Annual"], horizontal=True)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_vals = np.arange(1,13)
    chosen_label = st.selectbox("Month", month_labels, index=0)
    label_to_idx = dict(zip(month_labels, month_vals))

    st.subheader("Limits")
    motion_limit_roll = st.number_input("Roll limit (deg RMS)", 0.5, 45.0, 3.0, 0.5)
    motion_limit_lat  = st.number_input("Lateral acc limit (m/s² RMS)", 0.01, 1.50, 0.50, 0.01)
    motion_limit_vert = st.number_input("Vertical acc limit (m/s² RMS)", 0.01, 2.00, 1.00, 0.01)

    st.subheader("Metric to show")
    metric = st.selectbox(
        "Metric",
        [
            "Expected roll (deg)",
            "Expected lateral acc (m/s²)",
            "Expected vertical acc (m/s²)",
            "Operability: roll ≤ limit (%)",
            "Operability: lateral ≤ limit (%)",
            "Operability: vertical ≤ limit (%)",
            "Operability: ALL limits (%)",
        ],
    )

    st.subheader("Period mapping")
    csv_tp_text = st.text_input(
        "CSV Tp centers [s] (optional)",
        value=""
    )
    use_ordinal_tp = st.checkbox(
        "Treat CSV TP1..TPn as evenly spaced across dataset Tp range",
        value=True
    )

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
@st.cache_resource
def load_metocean(path: str) -> xr.Dataset:
    return xr.open_dataset(path)

def bin_centers(edges):
    return 0.5*(edges[:-1] + edges[1:])

def unwrap_lon_centers_from_edges(lon_edges):
    lon_c = bin_centers(lon_edges)
    if np.nanmax(lon_edges) > 180:
        lon_c = ((lon_c+180) % 360) - 180
    return lon_c

def to_sorted_lon_lat(arr2d, lat_c, lon_edges):
    flip_lat = False
    if lat_c[0] > lat_c[-1]:
        arr2d = arr2d[::-1,:]
        lat_c = lat_c[::-1]
        flip_lat = True

    lon_unsorted = unwrap_lon_centers_from_edges(lon_edges)
    lon_sort_idx = np.argsort(lon_unsorted)
    lon_sorted = lon_unsorted[lon_sort_idx]
    arr_sorted = arr2d[:, lon_sort_idx]
    lon_inv = np.argsort(lon_sort_idx)
    return arr_sorted, lat_c, lon_sorted, flip_lat, lon_sort_idx, lon_inv

def normalize_pdf(prob):
    tot = prob.sum(dim=("hs_bin","tp_bin"))
    return xr.where(tot > 0, prob/tot, 0)

# --- fixed shading -->
levels_generic = 50
clip_pct = 99.6

# --- Tp tick fix
def tp_ticks(step=1.0, vmin=None, vmax=None):
    if vmin is None or vmax is None:
        return np.arange(0,21,1)
    lo = np.floor(vmin)
    hi = np.ceil(vmax)
    return np.arange(lo, hi+0.5, 1)

def pct_ticks():
    return np.arange(0,101,10)

def pct_shading():
    return np.linspace(0,100,61)

def auto_levels(arr, n=50):
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if vmin == vmax or not np.isfinite(vmin):
        return np.linspace(0,1,n)
    return np.linspace(vmin, vmax, n)

# Plotting helper: **same land masking as original**
def plot_global_map(lon_c, lat_c, arr2d, title, filled, contours, cmap, ticks):
    fig = plt.figure(figsize=(15,6), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())

    cf = ax.contourf(
        lon_c, lat_c, arr2d,
        levels=filled,
        cmap=cmap,
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=1
    )
    try:
        cs = ax.contour(
            lon_c, lat_c, arr2d,
            levels=contours,
            colors="black",
            linewidths=0.4,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
        ax.clabel(cs, fontsize=6, inline=True, fmt="%g")
    except Exception:
        pass

    # --- same masking as original ---
    ax.add_feature(
        cfeature.LAND.with_scale("110m"),
        facecolor="lightgray",
        edgecolor="none",
        zorder=10
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("110m"),
        linewidth=0.8,
        zorder=11
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("110m"),
        linewidth=0.3,
        zorder=12
    )

    ax.set_global()

    cb = plt.colorbar(
        cf, ax=ax, shrink=0.75, aspect=30,
        pad=0.01, ticks=ticks
    )
    cb.set_label(title)
    cb.ax.tick_params(labelsize=8)

    ax.set_title(title)
    plt.subplots_adjust(left=0.02, right=0.97, top=0.93, bottom=0.06)
    st.pyplot(fig, use_container_width=True)

# -----------------------------------------------------------
# Load dataset
# -----------------------------------------------------------
try:
    ds = load_metocean(nc_path)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

for k in ["prob","hs_edges","tp_edges","lat3_edges","lon3_edges"]:
    if k not in ds:
        st.error(f"Dataset missing {k}")
        st.stop()

# Unit guard
hs_edges = ds["hs_edges"].values
hs_units = str(ds["hs_edges"].attrs.get("units","")).lower()
if "cm" in hs_units or (np.nanmax(hs_edges)>50 and "m" not in hs_units):
    hs_edges = hs_edges/100.0

hs_c = bin_centers(hs_edges)
tp_edges = ds["tp_edges"].values
tp_c = bin_centers(tp_edges)

lat_c_unsorted = bin_centers(ds["lat3_edges"].values)
lon_edges = ds["lon3_edges"].values

# -----------------------------------------------------------
# Select prob
# -----------------------------------------------------------
if agg == "By month":
    prob = ds["prob"].sel(month=label_to_idx[chosen_label])
    title_suffix = f" — {chosen_label}"
else:
    prob = ds["prob"].sum(dim="month")
    title_suffix = " — Annual"

prob = normalize_pdf(prob)

# -----------------------------------------------------------
# Load response CSV
# -----------------------------------------------------------
if resp_file is None:
    st.info("Upload TP-indexed response CSV.")
    st.stop()

def parse_response_csv(uploaded):
    df_raw = pd.read_csv(uploaded, header=None)
    header_tp = df_raw.iloc[1,2:].tolist()

    # Sort TP labels like TP1,TP2...
    def tpkey(s):
        m = re.search(r"(\d+)$", str(s))
        return int(m.group(1)) if m else 0
    tp_labels = sorted(header_tp, key=tpkey)

    records=[]
    for i in range(2,len(df_raw)):
        row=df_raw.iloc[i]
        alt=row.iloc[0]
        var=row.iloc[1]
        vals=row.iloc[2:2+len(tp_labels)].tolist()
        if pd.isna(alt) and pd.isna(var):
            continue
        rec={"Hull alternative":alt, "Response variable":var}
        for k,v in zip(tp_labels, vals):
            rec[k]=v
        records.append(rec)

    df=pd.DataFrame(records)
    df["Hull alternative"]=df["Hull alternative"].ffill().astype(str).str.strip()
    df["Response variable"]=df["Response variable"].astype(str).str.strip()

    def is_roll(s): return "roll" in s.lower()
    def is_lat(s):  return "lateral" in s.lower()
    def is_vert(s): return "vertical" in s.lower()
    def is_tp_row(s): return "tp" in s.lower() and "[" in s

    roll_df=df[df["Response variable"].apply(is_roll)]
    lat_df=df[df["Response variable"].apply(is_lat)]
    vert_df=df[df["Response variable"].apply(is_vert)]
    tp_df=df[df["Response variable"].apply(is_tp_row)]

    alt_names = roll_df["Hull alternative"].drop_duplicates().tolist()

    def matrix_from(df_alt):
        mats=[]
        for alt in alt_names:
            row=df_alt[df_alt["Hull alternative"]==alt][tp_labels]
            mats.append(row.iloc[0].astype(float).to_numpy())
        return np.vstack(mats)

    R_roll = matrix_from(roll_df)
    R_lat  = matrix_from(lat_df)
    R_vert = matrix_from(vert_df)

    csv_tp_sec=None
    if not tp_df.empty:
        try:
            arr = tp_df.iloc[0][tp_labels].astype(float).to_numpy()
            csv_tp_sec = arr
        except:
            pass

    return alt_names, tp_labels, R_roll, R_lat, R_vert, csv_tp_sec

try:
    alt_names, tp_labels, R_roll, R_lat, R_vert, csv_tp_sec = parse_response_csv(resp_file)
except Exception as e:
    st.error(f"Failed parsing: {e}")
    st.stop()

# -----------------------------------------------------------
# Interpolate RAO to dataset Tp grid
# -----------------------------------------------------------
n_tp_csv = R_roll.shape[1]

tp_csv_s = None
if csv_tp_text.strip():
    try:
        arr = np.array([float(x) for x in re.split(r"[ ,]+", csv_tp_text.strip())])
        if len(arr)==n_tp_csv:
            tp_csv_s = arr
    except:
        pass

if tp_csv_s is None and (csv_tp_sec is not None):
    tp_csv_s = csv_tp_sec

def interp_matrix(R, x_from, x_to):
    return np.vstack([np.interp(x_to, x_from, row) for row in R])

if len(tp_c)==n_tp_csv:
    R_roll_use=R_roll
    R_lat_use =R_lat
    R_vert_use=R_vert
    mapping_caption=f"TP mapping: 1:1 ({n_tp_csv} bins)."
else:
    tp_min,tp_max=float(tp_c.min()), float(tp_c.max())
    if tp_csv_s is not None:
        R_roll_use=interp_matrix(R_roll, tp_csv_s, tp_c)
        R_lat_use =interp_matrix(R_lat , tp_csv_s, tp_c)
        R_vert_use=interp_matrix(R_vert, tp_csv_s, tp_c)
        mapping_caption="TP mapping: interpolated from CSV Tp[s]."
    elif use_ordinal_tp:
        k=np.arange(1,n_tp_csv+1)
        x_from = tp_min + ((k-0.5)/n_tp_csv)*(tp_max-tp_min)
        R_roll_use=interp_matrix(R_roll, x_from, tp_c)
        R_lat_use =interp_matrix(R_lat , x_from, tp_c)
        R_vert_use=interp_matrix(R_vert, x_from, tp_c)
        mapping_caption="TP mapping: ordinal."
    else:
        st.error("Tp mismatch. Add Tp row or enable ordinal mapping.")
        st.stop()

# -----------------------------------------------------------
# Pick alternative
# -----------------------------------------------------------
with st.sidebar:
    alt_idx = st.selectbox("Hull alternative", list(range(len(alt_names))),
                           format_func=lambda i: alt_names[i])

RAO_roll = xr.DataArray(R_roll_use[alt_idx], dims=["tp_bin"])
RAO_lat  = xr.DataArray(R_lat_use [alt_idx], dims=["tp_bin"])
RAO_vert = xr.DataArray(R_vert_use[alt_idx], dims=["tp_bin"])

HS = xr.DataArray(hs_c, dims=["hs_bin"])

M_roll = HS * RAO_roll
M_lat  = HS * RAO_lat
M_vert = HS * RAO_vert

E_roll = (prob*M_roll).sum(dim=("hs_bin","tp_bin"))
E_lat  = (prob*M_lat ).sum(dim=("hs_bin","tp_bin"))
E_vert = (prob*M_vert).sum(dim=("hs_bin","tp_bin"))

I_roll = xr.where(M_roll <= motion_limit_roll, 1.0, 0.0)
I_lat  = xr.where(M_lat  <= motion_limit_lat , 1.0, 0.0)
I_vert = xr.where(M_vert <= motion_limit_vert, 1.0, 0.0)

P_roll = (prob*I_roll).sum(dim=("hs_bin","tp_bin"))*100
P_lat  = (prob*I_lat ).sum(dim=("hs_bin","tp_bin"))*100
P_vert = (prob*I_vert).sum(dim=("hs_bin","tp_bin"))*100
P_all  = (prob*(I_roll*I_lat*I_vert)).sum(dim=("hs_bin","tp_bin"))*100

# -----------------------------------------------------------
# Select field
# -----------------------------------------------------------
if metric=="Expected roll (deg)":
    field=E_roll; label=f"Expected roll (deg) — {alt_names[alt_idx]}{title_suffix}"
    is_percent=False
elif metric=="Expected lateral acc (m/s²)":
    field=E_lat; label=f"Expected lateral acc (m/s²) — {alt_names[alt_idx]}{title_suffix}"
    is_percent=False
elif metric=="Expected vertical acc (m/s²)":
    field=E_vert; label=f"Expected vertical acc (m/s²) — {alt_names[alt_idx]}{title_suffix}"
    is_percent=False
elif metric=="Operability: roll ≤ limit (%)":
    field=P_roll; label=f"Operability roll ≤ {motion_limit_roll:.1f} ({alt_names[alt_idx]})"
    is_percent=True
elif metric=="Operability: lateral ≤ limit (%)":
    field=P_lat; label=f"Operability lateral ≤ {motion_limit_lat:.2f} ({alt_names[alt_idx]})"
    is_percent=True
elif metric=="Operability: vertical ≤ limit (%)":
    field=P_vert; label=f"Operability vertical ≤ {motion_limit_vert:.2f} ({alt_names[alt_idx]})"
    is_percent=True
else:
    field=P_all; label=f"Operability ALL limits ({alt_names[alt_idx]})"
    is_percent=True

# -----------------------------------------------------------
# Prepare 2D field
# -----------------------------------------------------------
field2d = field.transpose("lat3_bin","lon3_bin").values
field2d, latp, lonp, flip_lat, lon_sort_idx, lon_inv = to_sorted_lon_lat(
    field2d, lat_c_unsorted, lon_edges
)

# clipping for non-%
if is_percent:
    filled = pct_shading()
    contours = pct_ticks()
    ticks = pct_ticks()
    cmap_use = "turbo_r"
else:
    hi=np.nanpercentile(field2d, clip_pct)
    field2d=np.clip(field2d,None,hi)
    filled=auto_levels(field2d, levels_generic)
    contours=filled
    ticks=None
    cmap_use="turbo"

# -----------------------------------------------------------
# Plot
# -----------------------------------------------------------
st.subheader("🌊 Motion-based map")
plot_global_map(
    lonp, latp, field2d,
    label,
    filled, contours, cmap_use, ticks
)

st.caption(mapping_caption)

# -----------------------------------------------------------
# Local inspector (kept)
# -----------------------------------------------------------
st.markdown("---")
st.subheader("🔎 Local values (grid cell)")

colA, colB = st.columns([1,3])

with colA:
    i_lat = st.number_input("lat index", 0, len(latp)-1, value=len(latp)//2)
    i_lon = st.number_input("lon index", 0, len(lonp)-1, value=len(lonp)//2)
    st.caption(f"Approx: {latp[int(i_lat)]:.1f}°, {lonp[int(i_lon)]:.1f}°")

lat_orig = len(lat_c_unsorted)-1-int(i_lat) if lat_c_unsorted[0]>lat_c_unsorted[-1] else int(i_lat)
lon_orig = int(lon_inv[int(i_lon)])

try:
    prob_local = prob.isel(lat3_bin=lat_orig, lon3_bin=lon_orig)

    Eroll = float((prob_local*M_roll).sum())
    Elat  = float((prob_local*M_lat ).sum())
    Evert = float((prob_local*M_vert).sum())

    Proll = float((prob_local*I_roll).sum()*100)
    Plat  = float((prob_local*I_lat ).sum()*100)
    Pvert = float((prob_local*I_vert).sum()*100)
    Pall  = float((prob_local*(I_roll*I_lat*I_vert)).sum()*100)

    with colB:
        st.write(f"**Expected RMS:** roll={Eroll:.2f}°, lat={Elat:.3f}, vert={Evert:.3f}")
        st.write(f"**Operability (%):** roll={Proll:.1f}%, lat={Plat:.1f}%, vert={Pvert:.1f}%, ALL={Pall:.1f}%")

except Exception as e:
    st.warning(f"Local inspector error: {e}")