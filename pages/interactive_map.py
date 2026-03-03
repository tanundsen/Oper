# 05_Interactive_Operability_Planner.py
# -----------------------------------------------------------
# PLAN MONTH-BY-MONTH OPERATING LOCATIONS AND COMPARE SYSTEMS
# -----------------------------------------------------------
# Features:
# • Clickable world map: assign a location to each month
# • Stores all 12 coordinates in session_state
# • Computes ALL-limits operability (%) for ALL hull alternatives
# • Produces table: rows = months, columns = systems
# • Systems sorted alphabetically (as requested)
# • Includes monthly comparison plot and annual summary
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import folium
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ===========================================================
# PAGE SETUP
# ===========================================================
st.set_page_config(page_title="📍 Operability Planner", layout="wide")
st.title("📍 Monthly Location Planner & Operability Comparison")


# ===========================================================
# LOAD NETCDF (cached)
# ===========================================================
@st.cache_resource
def load_metocean(path: str) -> xr.Dataset:
    return xr.open_dataset(path)


nc_path = "metocean_monthclim.nc"
ds = load_metocean(nc_path)

# Edge arrays
hs_edges = ds["hs_edges"].values
tp_edges = ds["tp_edges"].values
lat_edges = ds["lat3_edges"].values
lon_edges = ds["lon3_edges"].values

# Hs unit fix
units = str(ds["hs_edges"].attrs.get("units", "")).lower()
if ("cm" in units) or (np.nanmax(hs_edges) > 50 and "m" not in units):
    hs_edges = hs_edges / 100.0

hs_c = 0.5 * (hs_edges[:-1] + hs_edges[1:])
tp_c = 0.5 * (tp_edges[:-1] + tp_edges[1:])
lat_c_all = 0.5 * (lat_edges[:-1] + lat_edges[1:])
lon_c_all = 0.5 * (lon_edges[:-1] + lon_edges[1:])


# ===========================================================
# LOAD RAO CSV
# ===========================================================
st.sidebar.header("Response File")
resp_file = st.sidebar.file_uploader("Upload Response CSV", type=["csv"])

if resp_file is None:
    st.info("Upload your RAO CSV to proceed.")
    st.stop()

df_raw = pd.read_csv(resp_file, header=None)
tp_labels = df_raw.iloc[1, 2:].tolist()


# Sort TP1..TPn
def tp_key(s):
    import re
    m = re.search(r"(\d+)$", str(s))
    return int(m.group(1)) if m else 0


tp_labels = sorted(tp_labels, key=tp_key)

# Normalize CSV to wide format
records = []
for i in range(2, len(df_raw)):
    row = df_raw.iloc[i]
    alt = row.iloc[0]
    var = row.iloc[1]
    vals = row.iloc[2:2 + len(tp_labels)].tolist()
    if pd.isna(alt) and pd.isna(var):
        continue
    rec = {"Hull": alt, "Var": var}
    for k, v in zip(tp_labels, vals):
        rec[k] = v
    records.append(rec)

df = pd.DataFrame(records)
df["Hull"] = df["Hull"].ffill().astype(str).str.strip()
df["Var"] = df["Var"].astype(str).str.strip()

# Extract RAO sets
roll_df = df[df["Var"].str.contains("roll", case=False)]
lat_df = df[df["Var"].str.contains("lateral", case=False)]
vert_df = df[df["Var"].str.contains("vertical", case=False)]
tp_df = df[df["Var"].str.contains("tp", case=False)]

alt_names = sorted(roll_df["Hull"].unique())   # <‑‑ alphabetical (your choice)


def mat_from(df_alt):
    mats = []
    for alt in alt_names:
        row = df_alt[df_alt["Hull"] == alt][tp_labels]
        mats.append(row.iloc[0].astype(float).to_numpy())
    return np.vstack(mats)


R_roll = mat_from(roll_df)
R_lat = mat_from(lat_df)
R_vert = mat_from(vert_df)

# Optional Tp centers in CSV
csv_tp_sec = None
if not tp_df.empty:
    try:
        arr = tp_df.iloc[0][tp_labels].astype(float).to_numpy()
        csv_tp_sec = arr
    except:
        pass


# ===========================================================
# TP MAPPING
# ===========================================================
st.sidebar.header("TP Mapping")
use_ordinal_tp = st.sidebar.checkbox("Use ordinal TP mapping", True)
csv_tp_text = st.sidebar.text_input("Optional Tp centers (comma-separated)", "")

tp_csv_s = None
if csv_tp_text.strip():
    try:
        arr = np.array([float(x) for x in csv_tp_text.replace(",", " ").split()])
        if len(arr) == len(tp_labels):
            tp_csv_s = arr
    except:
        pass

if tp_csv_s is None:
    tp_csv_s = csv_tp_sec

n_csv = len(tp_labels)


def interp_mat(M, x_from, x_to):
    return np.vstack([np.interp(x_to, x_from, r) for r in M])


if len(tp_c) == n_csv:
    Rr_use, Rl_use, Rv_use = R_roll, R_lat, R_vert
else:
    tp_min, tp_max = float(tp_c.min()), float(tp_c.max())
    if tp_csv_s is not None:
        Rr_use = interp_mat(R_roll, tp_csv_s, tp_c)
        Rl_use = interp_mat(R_lat, tp_csv_s, tp_c)
        Rv_use = interp_mat(R_vert, tp_csv_s, tp_c)
    elif use_ordinal_tp:
        k = np.arange(1, n_csv + 1)
        x_from = tp_min + ((k - 0.5) / n_csv) * (tp_max - tp_min)
        Rr_use = interp_mat(R_roll, x_from, tp_c)
        Rl_use = interp_mat(R_lat, x_from, tp_c)
        Rv_use = interp_mat(R_vert, x_from, tp_c)
    else:
        st.error("TP mismatch. Provide Tp row or enable ordinal mapping.")
        st.stop()


# ===========================================================
# OPERABILITY LIMITS
# ===========================================================
st.sidebar.header("Operability Limits")
lim_roll = st.sidebar.number_input("Roll limit (deg)", 0.1, 45.0, 3.0)
lim_lat = st.sidebar.number_input("Lateral limit (m/s²)", 0.01, 2.0, 0.50)
lim_vert = st.sidebar.number_input("Vertical limit (m/s²)", 0.01, 3.0, 1.0)

# For computing expected motions
HS = xr.DataArray(hs_c, dims=["hs_bin"])


# Location planner
# ===========================================================
months = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]
short = [m[:3].lower() for m in months]

if "coords_by_month" not in st.session_state:
    st.session_state.coords_by_month = {}

st.sidebar.header("Assign Location per Month")

# Month selection UI
labels = []
map_month_to_label = {}

for m in months:
    s = m[:3].lower()
    has = s in st.session_state.coords_by_month
    label = f"✅ {m}" if has else f"⬜ {m}"
    labels.append(label)
    map_month_to_label[label] = m

remaining = [m for m in months if m[:3].lower() not in st.session_state.coords_by_month]
default_label = f"⬜ {remaining[0]}" if remaining else labels[0]

chosen_labels = st.multiselect(
    "Choose month(s) to assign now:",
    options=labels,
    default=[default_label]
)
selected_months = [map_month_to_label[x] for x in chosen_labels]


# ===========================================================
# FOLIUM MAP
# ===========================================================
st.markdown("### 🌍 Click on the map to set the vessel’s location for the selected month(s):")

m = folium.Map(location=[20, 0], zoom_start=2)

# Add markers for saved months
for mkey, coord in st.session_state.coords_by_month.items():
    folium.Marker(
        coord,
        tooltip=mkey.title(),
        popup=mkey.title(),
        icon=folium.DivIcon(html=f"<div style='font-size:12pt'>📍 {mkey}</div>")
    ).add_to(m)

folium_state = st_folium(m, height=600, width=1200, returned_objects=["last_clicked"])

# Handle new clicks
if selected_months and folium_state and folium_state.get("last_clicked"):
    latlng = (folium_state["last_clicked"]["lat"], folium_state["last_clicked"]["lng"])
    for m in selected_months:
        st.session_state.coords_by_month[m[:3].lower()] = latlng
    st.success(f"Saved location for: {', '.join(selected_months)} → {latlng}")
    st.rerun()


# ===========================================================
# Proceed when all 12 months are assigned
# ===========================================================
if len(st.session_state.coords_by_month) < 12:
    st.warning("Assign a location for all 12 months to compute operability.")
    st.stop()

st.markdown("---")
st.subheader("📊 Monthly Total Operability per System (ALL Limits)")


# ===========================================================
# Compute Operability for each month & each system
# ===========================================================

def nearest_grid(lat_sel, lon_sel):
    i_lat = np.abs(lat_c_all - lat_sel).argmin()
    i_lon = np.abs(lon_c_all - lon_sel).argmin()
    return i_lat, i_lon


def total_operability(prob, RAO_r, RAO_l, RAO_v):
    Mroll = HS * RAO_r
    Mlat = HS * RAO_l
    Mvert = HS * RAO_v
    I = xr.where(
        (Mroll <= lim_roll) &
        (Mlat <= lim_lat) &
        (Mvert <= lim_vert),
        1.0, 0.0
    )
    return float((prob * I).sum()) * 100.0


results = []

for idx, month in enumerate(months):
    s = month[:3].lower()
    lat_sel, lon_sel = st.session_state.coords_by_month[s]

    # nearest grid cell
    ilat, ilon = nearest_grid(lat_sel, lon_sel)

    # monthly prob → normalized
    p = ds["prob"].sel(month=idx+1).isel(lat3_bin=ilat, lon3_bin=ilon)
    tot = p.sum()
    if float(tot) > 0:
        p = p / tot

    row = {"Month": month, "Lat": lat_sel, "Lon": lon_sel}

    # compute per system
    for sys_i, sys in enumerate(alt_names):
        op = total_operability(
            p, 
            xr.DataArray(Rr_use[sys_i], dims=["tp_bin"]),
            xr.DataArray(Rl_use[sys_i], dims=["tp_bin"]),
            xr.DataArray(Rv_use[sys_i], dims=["tp_bin"])
        )
        row[sys] = op

    results.append(row)

df_out = pd.DataFrame(results)
df_out_display = df_out.drop(columns=["Lat","Lon"])

st.dataframe(df_out_display.set_index("Month"), use_container_width=True)


# ===========================================================
# Plot multi-system comparison
# ===========================================================
st.markdown("### 📈 Operability Comparison (ALL Limits)")

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(12)

for sys in alt_names:
    ax.plot(x, df_out[sys], marker="o", label=sys)

ax.set_xticks(x)
ax.set_xticklabels(months, rotation=45, ha="right")
ax.set_ylabel("Operability (%)")
ax.set_title("Monthly Total Operability per System")
ax.grid(True, alpha=0.3)
ax.legend()

st.pyplot(fig)


# ===========================================================
# Annual mean summary
# ===========================================================
st.markdown("### 🏁 Annual Mean Operability per System")

annual = {sys: df_out[sys].mean() for sys in alt_names}
df_ann = pd.DataFrame({"System": list(annual.keys()), "Annual Operability (%)": list(annual.values())})
st.dataframe(df_ann.set_index("System"), use_container_width=True)

# Bar chart
fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.bar(annual.keys(), annual.values(), color="steelblue")
ax2.set_title("Annual Mean Operability")
ax2.set_ylabel("%")
st.pyplot(fig2)