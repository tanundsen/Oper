
import xarray as xr
ds = xr.open_dataset("metocean_scatter_3deg_2010_2019.nc")

print(ds["lon3_edges"].values[:20])
print(ds["lon3_edges"].values[-20:])

