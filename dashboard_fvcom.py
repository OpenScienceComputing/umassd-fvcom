"""
GlobalCoast FVCOM dashboard (Panel + STAC/GeoParquet + Icechunk + xugrid)

- Uses a geoparquet STAC index and selects FVCOM entries via rustac.
- Opens the Icechunk store from the selected asset, with dask chunks (lazy).
- Attempts to wrap result in xugrid.UgridDataset if possible (or passes through raw ds).
- Renders a TriMesh interactive map with time/level controls.
"""

import os
import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
from holoviews.operation.datashader import rasterize as hv_rasterize
import xarray as xr
import xugrid as xu
import hvplot.xugrid  # noqa: F401 — registers .hvplot accessor on UgridDataArray
import time as _time_mod
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.interpolate import LinearNDInterpolator
import rustac
import icechunk

pn.extension(sizing_mode="stretch_width")
hv.extension("bokeh")

# Use the same geoparquet path from the user request (FVCOM-specific)
CATALOG_PARQUET_URL = os.environ.get(
    "FVCOM_STAC_GEOPARQUET",
    "https://umassd-fvcom.s3.amazonaws.com/gom3/hindcast/stac/gom3-hindcast.parquet",
)

VARS = {
    "temperature": ("°C", "RdYlBu_r", True, "temp"),
    "salinity": ("PSU", "viridis", True, "salinity"),
    "surface_elevation": ("m", "seismic", False, "zeta"),
    "currents": ("m/s", "turbo", False, None),
}

CMAPS = ["RdYlBu_r", "viridis", "plasma", "seismic", "jet", "turbo",
         "Blues", "Reds", "Greens", "hot", "cool", "magma", "inferno"]
CURR_MODES = ["None", "Curly vectors", "Arrow plot"]


def fetch_fvcom_items():
    client = rustac.DuckdbClient()
    try:
        items = client.search(
            CATALOG_PARQUET_URL,
            filter={"op": "like", "args": [{"property": "title"}, "%FVCOM%"]},
        )
        if not items:
            # fallback to id search
            items = client.search(
                CATALOG_PARQUET_URL,
                filter={"op": "like", "args": [{"property": "id"}, "%FVCOM%"]},
            )
        if not items:
            raise RuntimeError("No FVCOM entries found in geoparquet catalog")
        return items
    except Exception as e:
        raise RuntimeError(f"Could not query geoparquet STAC catalog: {e}")


def pick_icechunk_href(stac_item):
    # asset could be named 'icechunk' or 'icechunk@...'
    for key, asset in stac_item.get("assets", {}).items():
        href = asset.get("href")
        if href and "icechunk" in href:
            return href
    raise RuntimeError("No icechunk asset href found in FVCOM item")


def open_icechunk(href):
    if href.startswith("s3://"):
        parsed = href.replace("s3://", "").split("/", 1)
        bucket = parsed[0]
        prefix = parsed[1] if len(parsed) > 1 else ""

        config = icechunk.RepositoryConfig.default()
        config.set_virtual_chunk_container(
            icechunk.VirtualChunkContainer(
                url_prefix=f"s3://{bucket}/",
                store=icechunk.s3_store(region="us-east-1", anonymous=True),
            )
        )
        storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, region="us-east-1", anonymous=True)
        creds = icechunk.containers_credentials({f"s3://{bucket}/": icechunk.s3_credentials(anonymous=True)})
        repo = icechunk.Repository.open(storage, config, authorize_virtual_chunk_access=creds)
        session = repo.readonly_session("main")
        return xr.open_zarr(session.store, consolidated=False, chunks="auto")
    raise ValueError("Only s3:// icechunk href is supported")


def maybe_wrap_xugrid(ds):
    if hasattr(ds, "ugrid_roles") and ds.ugrid_roles.topology:
        if "nv" in ds and ds.nv.dims == ("three", "nele"):
            ds["nv"] = ds["nv"].transpose("nele", "three")
        return xu.UgridDataset(ds)
    try:
        if "nv" in ds and ds.nv.dims == ("three", "nele"):
            ds["nv"] = ds["nv"].transpose("nele", "three")
        return xu.UgridDataset(ds)
    except Exception:
        return xu.UgridDataset(add_ugrid_metadata(ds))

def add_ugrid_metadata(ds):
    """
    Assigns UGRID convention metadata to an xarray Dataset.
    """
    mesh_name = "mesh_topology"

    mesh_attrs = {
        "cf_role": "mesh_topology",
        "topology_dimension": 2,
        "node_coordinates": "lon lat",
        "face_coordinates": "lonc latc",
        "face_node_connectivity": "nv",
        "face_dimension": "nele"
    }

    if mesh_name not in ds:
        ds = ds.assign({mesh_name: xr.DataArray(0, attrs=mesh_attrs)})
    else:
        ds[mesh_name].attrs.update(mesh_attrs)

    if "nv" in ds and ds.nv.dims == ("three", "nele"):
        ds["nv"] = ds["nv"].transpose("nele", "three")

    if "nv" in ds:
        ds.nv.attrs.update({
            "cf_role": "face_node_connectivity",
            "start_index": 1
        })

    for var in ds.data_vars:
        if "node" in ds[var].dims or "nele" in ds[var].dims:
            ds[var].attrs["mesh"] = mesh_name
            ds[var].attrs["location"] = (
                "face" if "nele" in ds[var].dims else "node"
            )

    return ds

print("Loading FVCOM dataset from STAC geoparquet...")
items = fetch_fvcom_items()
item = items[0]
asset_href = pick_icechunk_href(item)
print(f"Opening icechunk href: {asset_href}")
ds = open_icechunk(asset_href)

# Extract topology from raw xarray Dataset BEFORE wrapping — once xugrid
# consumes the dataset, nv/lonc/latc disappear as addressable variables.
LON  = ds["lon"].values.astype(float)
LAT  = ds["lat"].values.astype(float)
LONC = ds["lonc"].values.astype(float)  # face centroids — where FVCOM u/v live
LATC = ds["latc"].values.astype(float)
_nv  = ds["nv"].values
ELEM = (_nv if _nv.shape[1] == 3 else _nv.T) - 1  # ensure (nele, 3), 0-based

ds = maybe_wrap_xugrid(ds)  # now wrap for hvplot rendering

LON_MIN, LON_MAX = float(LON.min()), float(LON.max())
LAT_MIN, LAT_MAX = float(LAT.min()), float(LAT.max())
PAD = 0.02

# Node-based triangulation used only for _split_at_land land clipping
TRIANG    = mtri.Triangulation(LON, LAT, ELEM)
TRIFINDER = TRIANG.get_trifinder()

CURLY_NX, CURLY_NY = 60, 60  # interpolation grid resolution

TIMES = pd.DatetimeIndex(ds["time"].values)
N_TIMES = len(TIMES)

def parse_time_to_index(time_str):
    try:
        dt = pd.to_datetime(time_str)
        # Find the closest time index
        idx = np.argmin(np.abs(TIMES - dt))
        return idx
    except Exception as e:
        print(f"Error parsing time '{time_str}': {e}. Using first time step.")
        return 0

# Widgets
var_sel = pn.widgets.Select(name="Variable", options=list(VARS.keys()), value="temperature")
time_sel = pn.widgets.TextInput(name="Time (YYYY-MM-DD HH:MM)", value=TIMES[0].strftime('%Y-%m-%d %H:%M'))
_n_levels = ds.sizes.get("siglay", 1)
level_sel = pn.widgets.Select(
    name="Level",
    options=list(range(_n_levels)),
    value=0,
    disabled="siglay" not in ds.dims,
)

cmap_sel = pn.widgets.Select(name="Palette", options=CMAPS, value="RdYlBu_r")
curr_mode_sel = pn.widgets.Select(name="Currents overlay", options=CURR_MODES, value="None")
curr_color_sel = pn.widgets.Select(name="Currents color", options=["white", "black", "red", "blue", "green"], value="white")
vector_len_sl = pn.widgets.FloatSlider(name="Vector length", start=0.05, end=2.0,
                                       step=0.05, value=0.5)
play_btn = pn.widgets.Toggle(name="▶ Play", button_type="success")
reset_btn = pn.widgets.Button(name="⟳ Reset range", button_type="primary", width=130)
adaptive_cb = pn.widgets.Checkbox(name="Adaptive range", value=False)
vmin_input = pn.widgets.FloatInput(name="Min", value=0.0, width=100)
vmax_input = pn.widgets.FloatInput(name="Max", value=1.0, width=100)

# Time navigation buttons
prev_btn = pn.widgets.Button(name="◀ Prev", button_type="default", width=60)
next_btn = pn.widgets.Button(name="Next ▶", button_type="default", width=60)

_range_cache = {}
_curly_cache = {}
_arrow_cache = {}


def lonlat_to_merc(lon, lat):
    x = np.asarray(lon, float) * 20037508.34 / 180.0
    y = np.log(np.tan(np.pi/4 + np.radians(np.clip(lat, -85., 85.))/2)) * 20037508.34 / np.pi
    return x, y


def get_uv_faces(tidx, level):
    """Return (u, v) as float numpy arrays at face centroids (nele,)."""
    if "siglay" in ds.dims and "u" in ds and "v" in ds:
        u = ds["u"].isel(time=tidx, siglay=level).values
        v = ds["v"].isel(time=tidx, siglay=level).values
    elif "ua" in ds and "va" in ds:
        u = ds["ua"].isel(time=tidx).values
        v = ds["va"].isel(time=tidx).values
    else:
        raise RuntimeError("Currents variables (u/v or ua/va) not found")
    if hasattr(u, "compute"):
        u = u.compute()
    if hasattr(v, "compute"):
        v = v.compute()
    return u.astype(float), v.astype(float)


def _split_at_land(verts):
    """Break a streamplot path into segments that lie inside the mesh."""
    segments, cur = [], []
    for xi, yi in verts:
        if TRIFINDER(xi, yi) != -1:
            cur.append((xi, yi))
        else:
            if len(cur) >= 2:
                segments.append(cur)
            cur = []
    if len(cur) >= 2:
        segments.append(cur)
    return segments


def build_curly_paths(tidx, level, maxlength,
                      lon_min=None, lon_max=None, lat_min=None, lat_max=None):
    """
    Build curly-vector paths for FVCOM.
    Velocities are face-centered (lonc, latc) and interpolated to a regular
    grid via LinearNDInterpolator before matplotlib streamplot traces paths.
    """
    lon_min = lon_min if lon_min is not None else LON_MIN
    lon_max = lon_max if lon_max is not None else LON_MAX
    lat_min = lat_min if lat_min is not None else LAT_MIN
    lat_max = lat_max if lat_max is not None else LAT_MAX
    k = (tidx, level, maxlength,
         round(lon_min, 3), round(lon_max, 3),
         round(lat_min, 3), round(lat_max, 3))
    if k in _curly_cache:
        return _curly_cache[k]

    u_f, v_f = get_uv_faces(tidx, level)
    face_pts = np.column_stack([LONC, LATC])
    iu = LinearNDInterpolator(face_pts, u_f)
    iv = LinearNDInterpolator(face_pts, v_f)

    lons = np.linspace(lon_min, lon_max, CURLY_NX)
    lats = np.linspace(lat_min, lat_max, CURLY_NY)
    lon2d, lat2d = np.meshgrid(lons, lats)
    grid_pts = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    gu = iu(grid_pts).reshape(CURLY_NY, CURLY_NX)
    gv = iv(grid_pts).reshape(CURLY_NY, CURLY_NX)

    fig, ax = plt.subplots(figsize=(8, 8))
    sp = ax.streamplot(lons, lats,
                       np.where(np.isnan(gu), 0., gu),
                       np.where(np.isnan(gv), 0., gv),
                       density=2.5, minlength=0.05, maxlength=maxlength,
                       linewidth=1.0, arrowsize=1.0)
    raw_paths = sp.lines.get_paths()
    plt.close(fig)

    xs_ll, ys_ll, tip_lon, tip_lat, tip_ang = [], [], [], [], []
    for path in raw_paths:
        verts = path.vertices
        if len(verts) < 2:
            continue
        for seg in _split_at_land(verts):
            sx = [p[0] for p in seg]
            sy = [p[1] for p in seg]
            xs_ll.append(sx)
            ys_ll.append(sy)
            path_ang = np.arctan2(sy[-1] - sy[-2], sx[-1] - sx[-2])
            u_tip = iu(np.array([[sx[-1], sy[-1]]]))
            v_tip = iv(np.array([[sx[-1], sy[-1]]]))
            if not np.isnan(u_tip[0]) and not np.isnan(v_tip[0]):
                vel_ang = np.arctan2(float(v_tip[0]), float(u_tip[0]))
                diff = np.arctan2(np.sin(path_ang - vel_ang),
                                  np.cos(path_ang - vel_ang))
                if abs(diff) > np.pi / 2:
                    path_ang += np.pi
            tip_lon.append(sx[-1])
            tip_lat.append(sy[-1])
            tip_ang.append(float(path_ang - np.pi / 2))

    result = (xs_ll, ys_ll, tip_lon, tip_lat, tip_ang)
    _curly_cache[k] = result
    return result


def build_arrow_grid(tidx, level, lon_min=None, lon_max=None, lat_min=None, lat_max=None):
    """Interpolate face-centered u/v to a regular grid for arrow/vector-field plot."""
    lon_min = lon_min if lon_min is not None else LON_MIN
    lon_max = lon_max if lon_max is not None else LON_MAX
    lat_min = lat_min if lat_min is not None else LAT_MIN
    lat_max = lat_max if lat_max is not None else LAT_MAX
    k = (tidx, level, round(lon_min, 3), round(lon_max, 3),
         round(lat_min, 3), round(lat_max, 3))
    if k in _arrow_cache:
        return _arrow_cache[k]
    u_f, v_f = get_uv_faces(tidx, level)
    face_pts = np.column_stack([LONC, LATC])
    iu = LinearNDInterpolator(face_pts, u_f)
    iv = LinearNDInterpolator(face_pts, v_f)
    lons = np.linspace(lon_min, lon_max, CURLY_NX)
    lats = np.linspace(lat_min, lat_max, CURLY_NY)
    grid_pts = np.column_stack([g.ravel() for g in np.meshgrid(lons, lats)])
    gu = iu(grid_pts).reshape(CURLY_NY, CURLY_NX)
    gv = iv(grid_pts).reshape(CURLY_NY, CURLY_NX)
    result = (lons, lats, gu, gv)
    _arrow_cache[k] = result
    return result


def get_values(variable, tidx, level):
    if variable == "currents":
        if "siglay" in ds.dims and "u" in ds and "v" in ds:
            u = ds["u"].isel(time=tidx, siglay=level).values
            v = ds["v"].isel(time=tidx, siglay=level).values
        elif "ua" in ds and "va" in ds:
            u = ds["ua"].isel(time=tidx).values
            v = ds["va"].isel(time=tidx).values
        else:
            raise RuntimeError("Currents variables not found")
        return np.sqrt(u**2 + v**2)

    _, _, has_level, vname = VARS[variable]
    da = ds[vname]
    if has_level and "siglay" in da.dims:
        arr = da.isel(time=tidx, siglay=level).values
    elif "time" in da.dims:
        arr = da.isel(time=tidx).values
    else:
        arr = da.values
    return arr


def get_full_range(variable, level):
    key = (variable, level)
    if key not in _range_cache:
        vals = get_values(variable, 0, level)
        # If dask-backed, compute minimal chunk
        if hasattr(vals, "compute"):
            vals = vals.compute()
        vmin = float(np.nanpercentile(vals, 2))
        vmax = float(np.nanpercentile(vals, 98))
        _range_cache[key] = (round(vmin, 4), round(vmax, 4))
    return _range_cache[key]


def auto_range(event=None):
    variable = var_sel.value
    tidx = parse_time_to_index(time_sel.value)
    level = level_sel.value
    if adaptive_cb.value:
        vals = get_values(variable, tidx, level)
        if hasattr(vals, "compute"):
            vals = vals.compute()
        vmin = float(np.nanpercentile(vals, 2))
        vmax = float(np.nanpercentile(vals, 98))
    else:
        vmin, vmax = get_full_range(variable, level)
    vmin_input.value = vmin
    vmax_input.value = vmax


def on_var_change(event):
    _, cmap_default, _, _ = VARS[event.new]
    cmap_sel.value = cmap_default
    _range_cache.clear()
    auto_range()

def on_time_change(event):
    tidx = parse_time_to_index(event.new)
    formatted = TIMES[tidx].strftime('%Y-%m-%d %H:%M')
    if time_sel.value != formatted:
        time_sel.value = formatted

def prev_time(event):
    tidx = parse_time_to_index(time_sel.value)
    new_tidx = max(0, tidx - 1)
    time_sel.value = TIMES[new_tidx].strftime('%Y-%m-%d %H:%M')

def next_time(event):
    tidx = parse_time_to_index(time_sel.value)
    new_tidx = min(N_TIMES - 1, tidx + 1)
    time_sel.value = TIMES[new_tidx].strftime('%Y-%m-%d %H:%M')

var_sel.param.watch(on_var_change, "value")
level_sel.param.watch(lambda e: auto_range(), "value")
time_sel.param.watch(lambda e: auto_range() if adaptive_cb.value else None, "value")
time_sel.param.watch(on_time_change, "value")
adaptive_cb.param.watch(lambda e: auto_range(), "value")
reset_btn.on_click(auto_range)
prev_btn.on_click(prev_time)
next_btn.on_click(next_time)
auto_range()

def get_slice_uda(variable, tidx, level):
    """Return a UgridDataArray slice for hvplot rendering."""
    if variable == "currents":
        if "siglay" in ds.dims and "u" in ds and "v" in ds:
            u = ds["u"].isel(time=tidx, siglay=level)
            v = ds["v"].isel(time=tidx, siglay=level)
        elif "ua" in ds and "va" in ds:
            u = ds["ua"].isel(time=tidx)
            v = ds["va"].isel(time=tidx)
        else:
            raise RuntimeError("Currents variables not found")
        speed = np.sqrt(u**2 + v**2)
        speed.name = "currents"
        return speed

    _, _, has_level, vname = VARS[variable]
    da = ds[vname]
    if has_level and "siglay" in da.dims:
        return da.isel(time=tidx, siglay=level)
    elif "time" in da.dims:
        return da.isel(time=tidx)
    return da


# ── Streams ────────────────────────────────────────────────────────────────

PlotStream = hv.streams.Stream.define(
    "PlotStream",
    variable="temperature", time_str=TIMES[0].strftime('%Y-%m-%d %H:%M'), level=0, cmap="RdYlBu_r",
    vmin=0.0, vmax=1.0, curr_mode="None", curr_color="white", vector_len=0.5,
)
ViewportStream = hv.streams.Stream.define("ViewportStream",
    lon_min=LON_MIN, lon_max=LON_MAX, lat_min=LAT_MIN, lat_max=LAT_MAX)

_stream    = PlotStream(
    variable=var_sel.value, time_str=time_sel.value, level=level_sel.value,
    cmap=cmap_sel.value, vmin=vmin_input.value, vmax=vmax_input.value,
    curr_mode=curr_mode_sel.value, curr_color=curr_color_sel.value, vector_len=vector_len_sl.value,
)
_vp_stream = ViewportStream()
_rxy_ref   = [None]
_zoom_cb_ref  = [None]
_last_zoom_t  = [0.0]

# ── Layer callbacks ────────────────────────────────────────────────────────

_EMPTY_TIPS = {"Longitude": [], "Latitude": [], "angle": []}

def field_layer(variable, time_str, level, cmap, vmin, vmax, curr_mode, curr_color, vector_len):
    tidx = parse_time_to_index(time_str)
    da = get_slice_uda(variable, tidx, level)
    return da.hvplot.trimesh(
        geo=True, xlabel="", ylabel="",
        xlim=(LON_MIN - PAD, LON_MAX + PAD),
        ylim=(LAT_MIN - PAD, LAT_MAX + PAD),
    )

def apply_style(el, variable, time_str, level, cmap, vmin, vmax, curr_mode, curr_color, vector_len):
    tidx = parse_time_to_index(time_str)
    units, _, _, _ = VARS[variable]
    lv_str   = f" L{level}" if "siglay" in ds.dims else ""
    mode_str = f" | {curr_mode}" if curr_mode != "None" else ""
    title = f"FVCOM {variable}{lv_str} | {TIMES[tidx].strftime('%Y-%m-%d %H:%M')}{mode_str}"
    return el.opts(hv.opts.Image(
        cmap=cmap, clim=(vmin, vmax), colorbar=True,
        colorbar_opts={"title": units}, title=title,
        xlabel="Longitude", ylabel="Latitude",
        tools=["hover"], active_tools=["wheel_zoom"],
        responsive=True, min_height=650,
    ))

def curly_layer(lon_min, lon_max, lat_min, lat_max,
                variable, time_str, level, cmap, vmin, vmax, curr_mode, curr_color, vector_len):
    if curr_mode != "Curly vectors":
        return hv.Path([], kdims=["Longitude", "Latitude"]).opts(apply_ranges=False)
    tidx = parse_time_to_index(time_str)
    try:
        xs_ll, ys_ll, _, _, _ = build_curly_paths(
            tidx, level, vector_len, lon_min, lon_max, lat_min, lat_max)
        paths = []
        for xs, ys in zip(xs_ll, ys_ll):
            mx, my = lonlat_to_merc(np.array(xs), np.array(ys))
            paths.append(list(zip(mx.tolist(), my.tolist())))
        return hv.Path(paths, kdims=["Longitude", "Latitude"]).opts(
            color=curr_color, line_width=1.5, alpha=0.9, apply_ranges=False)
    except Exception:
        return hv.Path([], kdims=["Longitude", "Latitude"]).opts(apply_ranges=False)

def tips_layer(lon_min, lon_max, lat_min, lat_max,
               variable, time_str, level, cmap, vmin, vmax, curr_mode, curr_color, vector_len):
    if curr_mode != "Curly vectors":
        return hv.Points(_EMPTY_TIPS, kdims=["Longitude", "Latitude"], vdims=["angle"]).opts(apply_ranges=False)
    tidx = parse_time_to_index(time_str)
    try:
        _, _, tlons, tlats, tangs = build_curly_paths(
            tidx, level, vector_len, lon_min, lon_max, lat_min, lat_max)
        tip_mx, tip_my = lonlat_to_merc(np.array(tlons), np.array(tlats))
        return hv.Points(
            {"Longitude": tip_mx.tolist(), "Latitude": tip_my.tolist(),
             "angle": np.degrees(tangs).tolist()},
            kdims=["Longitude", "Latitude"], vdims=["angle"],
        ).opts(marker="triangle", color="white", size=8,
               angle=hv.dim("angle"), alpha=0.9, apply_ranges=False)
    except Exception:
        return hv.Points(_EMPTY_TIPS, kdims=["Longitude", "Latitude"], vdims=["angle"]).opts(apply_ranges=False)

_EMPTY_VF = (np.array([0.]), np.array([0.]), np.array([0.]), np.array([0.]))

def arrow_layer(lon_min, lon_max, lat_min, lat_max,
                variable, time_str, level, cmap, vmin, vmax, curr_mode, curr_color, vector_len):
    if curr_mode != "Arrow plot":
        return hv.VectorField(_EMPTY_VF, kdims=["Longitude", "Latitude"],
                              vdims=["Angle", "Magnitude"]).opts(alpha=0, apply_ranges=False)
    tidx = parse_time_to_index(time_str)
    try:
        lons, lats, gu, gv = build_arrow_grid(tidx, level, lon_min, lon_max, lat_min, lat_max)
        # Flatten to point-per-row, convert to Mercator, filter out-of-mesh NaNs
        lon2d, lat2d = np.meshgrid(lons, lats)
        lon_flat, lat_flat = lon2d.ravel(), lat2d.ravel()
        mx, my = lonlat_to_merc(lon_flat, lat_flat)
        gu_flat, gv_flat = gu.ravel(), gv.ravel()
        mag_flat = np.sqrt(gu_flat**2 + gv_flat**2)
        in_ocean = TRIFINDER(lon_flat, lat_flat) != -1
        valid = ~(np.isnan(gu_flat) | np.isnan(gv_flat)) & (mag_flat > 0.01) & in_ocean
        mx, my = mx[valid], my[valid]
        gu_flat, gv_flat = gu_flat[valid], gv_flat[valid]
        angle = np.arctan2(gv_flat, gu_flat)
        mag   = mag_flat[valid]
        max_mag = mag.max()
        mag_norm = mag / max_mag if max_mag > 0 else mag
        return hv.VectorField(
            (mx, my, angle, mag_norm),
            kdims=["Longitude", "Latitude"],
            vdims=["Angle", "Magnitude"],
        ).opts(color=curr_color, line_color=curr_color, alpha=0.8,
               scale=vector_len, apply_ranges=False)
    except Exception as e:
        print(f"[arrow] error: {e}")
        return hv.VectorField(_EMPTY_VF, kdims=["Longitude", "Latitude"],
                              vdims=["Angle", "Magnitude"]).opts(alpha=0, apply_ranges=False)

# ── Zoom debounce ──────────────────────────────────────────────────────────

def _on_zoom(*args, **kwargs):
    _last_zoom_t[0] = _time_mod.time()
    if _zoom_cb_ref[0] is not None:
        return
    def _check():
        if _time_mod.time() - _last_zoom_t[0] > 0.5:
            _zoom_cb_ref[0].stop()
            _zoom_cb_ref[0] = None
            rxy = _rxy_ref[0]
            if rxy is None:
                return
            xr, yr = rxy.x_range, rxy.y_range
            if xr and xr[0] is not None:
                try:
                    _vp_stream.event(
                        lon_min=max(float(xr[0]), LON_MIN - PAD),
                        lon_max=min(float(xr[1]), LON_MAX + PAD),
                        lat_min=max(float(yr[0]), LAT_MIN - PAD),
                        lat_max=min(float(yr[1]), LAT_MAX + PAD),
                    )
                except Exception as e:
                    print(f"[zoom] error: {e}", flush=True)
    _zoom_cb_ref[0] = pn.state.add_periodic_callback(_check, period=100)

# ── Wire up DynamicMaps ────────────────────────────────────────────────────

tiles      = hv.element.tiles.OSM()
field_dmap = hv.DynamicMap(field_layer, streams=[_stream])
rasterized = hv_rasterize(field_dmap)
styled     = rasterized.apply(apply_style, streams=[_stream])

_rxy_ref[0] = next((s for s in rasterized.streams if isinstance(s, hv.streams.RangeXY)), None)
if _rxy_ref[0] is not None:
    _rxy_ref[0].param.watch(_on_zoom, ["x_range", "y_range"])

curly_dmap = hv.DynamicMap(curly_layer, streams=[_vp_stream, _stream])
tips_dmap  = hv.DynamicMap(tips_layer,  streams=[_vp_stream, _stream])
arrow_dmap = hv.DynamicMap(arrow_layer, streams=[_vp_stream, _stream])

plot_obj  = (tiles * styled * curly_dmap * tips_dmap * arrow_dmap).opts(
    hv.opts.Overlay(responsive=True, min_height=650)
)
plot_pane = pn.pane.HoloViews(plot_obj, sizing_mode="stretch_both", min_height=650)

# ── Widget → stream refresh ────────────────────────────────────────────────

def refresh(*events):
    _stream.event(
        variable=var_sel.value, time_str=time_sel.value, level=level_sel.value,
        cmap=cmap_sel.value, vmin=vmin_input.value, vmax=vmax_input.value,
        curr_mode=curr_mode_sel.value, curr_color=curr_color_sel.value, vector_len=vector_len_sl.value,
    )

for _w in [var_sel, time_sel, level_sel, cmap_sel, vmin_input, vmax_input,
           curr_mode_sel, curr_color_sel, vector_len_sl]:
    _w.param.watch(refresh, "value")

sidebar = pn.Column(
    pn.pane.Markdown("## FVCOM GOM3 Explorer"),
    pn.Spacer(height=8),
    var_sel,
    pn.Row(time_sel, pn.Spacer(width=5), prev_btn, next_btn),
    level_sel,
    pn.Spacer(height=8),
    cmap_sel,
    pn.Row(vmin_input, vmax_input),
    pn.Row(reset_btn, adaptive_cb),
    pn.Spacer(height=8),
    curr_mode_sel,
    curr_color_sel,
    vector_len_sl,
    pn.Spacer(height=8),
    play_btn,
)

app = pn.template.FastListTemplate(
    title="GlobalCoast FVCOM Viewer",
    sidebar=[sidebar],
    main=[plot_pane],
    theme="dark",
    accent="#1a8cff",
)

if __name__ == "__main__":
    pn.serve(app.servable(), host="0.0.0.0", port=9002, show=False, allow_websocket_origin=["*"])
