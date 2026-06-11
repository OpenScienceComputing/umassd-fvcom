# pyright: reportAssignmentType=false
"""
FVCOM GOM3 Dashboard — panel-material-ui + param Viewer rewrite
"""

import os
import time as _time_mod

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import numpy as np
import pandas as pd
import panel as pn
import panel_material_ui as pmui
import holoviews as hv
import param
from holoviews.operation.datashader import rasterize as hv_rasterize
import xarray as xr
import xugrid as xu
import hvplot.xugrid  # noqa: F401
from scipy.interpolate import LinearNDInterpolator
import rustac
import icechunk

pn.extension(throttled=True)
hv.extension("bokeh")

CATALOG_PARQUET_URL = os.environ.get(
    "FVCOM_STAC_GEOPARQUET",
    "https://umassd-fvcom.s3.amazonaws.com/gom3/hindcast/stac/gom3-hindcast.parquet",
)

VARS = {
    "temperature":       ("°C",   "turbo", True,  "temp"),
    "salinity":          ("PSU",  "turbo", True,  "salinity"),
    "surface_elevation": ("m",    "turbo", False, "zeta"),
    "currents":          ("m/s",  "turbo", False, None),
}

CMAPS = [
    "RdYlBu_r", "viridis", "plasma", "seismic", "jet", "turbo",
    "Blues", "Reds", "Greens", "hot", "cool", "magma", "inferno",
]

CURR_MODES  = ["None", "Curly vectors", "Arrow plot"]
CURR_COLORS = ["white", "black", "red", "blue", "green"]
CURLY_NX = CURLY_NY = 60

# ── Module-level pure helpers ────────────────────────────────────────────────

def _fetch_fvcom_items():
    client = rustac.DuckdbClient()
    for filt in [
        {"op": "like", "args": [{"property": "title"}, "%FVCOM%"]},
        {"op": "like", "args": [{"property": "id"},    "%FVCOM%"]},
    ]:
        items = client.search(CATALOG_PARQUET_URL, filter=filt)
        if items:
            return items
    raise RuntimeError("No FVCOM entries found in catalog")


def _pick_icechunk_href(item):
    for asset in item.get("assets", {}).values():
        href = asset.get("href", "")
        if "icechunk" in href:
            return href
    raise RuntimeError("No icechunk asset found")


def _open_icechunk(href):
    bucket, prefix = href.replace("s3://", "").split("/", 1)
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(icechunk.VirtualChunkContainer(
        url_prefix=f"s3://{bucket}/",
        store=icechunk.s3_store(region="us-east-1", anonymous=True),
    ))
    storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, region="us-east-1", anonymous=True)
    creds = icechunk.containers_credentials(
        {f"s3://{bucket}/": icechunk.s3_credentials(anonymous=True)}
    )
    repo = icechunk.Repository.open(storage, config, authorize_virtual_chunk_access=creds)
    return xr.open_zarr(repo.readonly_session("main").store, consolidated=False, chunks="auto")


def _add_ugrid_metadata(ds):
    mesh_name = "mesh_topology"
    attrs = {
        "cf_role": "mesh_topology", "topology_dimension": 2,
        "node_coordinates": "lon lat", "face_coordinates": "lonc latc",
        "face_node_connectivity": "nv", "face_dimension": "nele",
    }
    if mesh_name not in ds:
        ds = ds.assign({mesh_name: xr.DataArray(0, attrs=attrs)})
    else:
        ds[mesh_name].attrs.update(attrs)
    if "nv" in ds:
        start_index = int(ds["nv"].attrs.get("start_index", ds["nv"].values.min()))
        ds.nv.attrs.update({
            "cf_role": "face_node_connectivity",
            "start_index": start_index,
            "face_dimension": "nele",
        })
    for var in ds.data_vars:
        if "node" in ds[var].dims or "nele" in ds[var].dims:
            ds[var].attrs.update({
                "mesh": mesh_name,
                "location": "face" if "nele" in ds[var].dims else "node",
            })
    return ds


def _wrap_xugrid(ds):
    try:
        return xu.UgridDataset(ds)
    except Exception:
        return xu.UgridDataset(_add_ugrid_metadata(ds))


def _lonlat_to_merc(lon, lat):
    x = np.asarray(lon, float) * 20037508.34 / 180.0
    y = (np.log(np.tan(np.pi / 4 + np.radians(np.clip(lat, -85.0, 85.0)) / 2))
         * 20037508.34 / np.pi)
    return x, y


# ── Load dataset (module-level, once on import) ──────────────────────────────

print("Loading FVCOM dataset from STAC geoparquet…")
_item   = _fetch_fvcom_items()[0]
_RAW_DS = _open_icechunk(_pick_icechunk_href(_item))

LON  = _RAW_DS["lon"].values.astype(float)
LAT  = _RAW_DS["lat"].values.astype(float)
LONC = _RAW_DS["lonc"].values.astype(float)
LATC = _RAW_DS["latc"].values.astype(float)
_nv         = _RAW_DS["nv"].values
_start_idx  = int(_RAW_DS["nv"].attrs.get("start_index", _nv.min()))
ELEM = (_nv if _nv.shape[1] == 3 else _nv.T) - _start_idx  # (nele, 3), 0-based

DS        = _wrap_xugrid(_RAW_DS)
LON_MIN   = float(LON.min())
LON_MAX   = float(LON.max())
LAT_MIN   = float(LAT.min())
LAT_MAX   = float(LAT.max())
PAD       = 0.02
TRIANG    = mtri.Triangulation(LON, LAT, ELEM)
TRIFINDER = TRIANG.get_trifinder()
TIMES     = pd.DatetimeIndex(_RAW_DS["time"].values)
N_TIMES   = len(TIMES)
N_LEVELS  = _RAW_DS.sizes.get("siglay", 1)
print(f"  {N_TIMES} time steps, {N_LEVELS} sigma levels.")

# ── Stream class definitions (module-level so they're created only once) ─────

_PlotStream = hv.streams.Stream.define(
    "PlotStream",
    variable="temperature", time_idx=0, level=0, cmap="turbo",
    vmin=0.0, vmax=1.0, curr_mode="None", curr_color="white", vector_len=0.5,
)
_ViewportStream = hv.streams.Stream.define(
    "ViewportStream",
    lon_min=LON_MIN, lon_max=LON_MAX, lat_min=LAT_MIN, lat_max=LAT_MAX,
)

# ── Dashboard ────────────────────────────────────────────────────────────────

class FVCOMDashboard(pn.viewable.Viewer):
    _EMPTY_TIPS = {"Longitude": [], "Latitude": [], "angle": []}
    _EMPTY_VF   = (np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))

    def __init__(self, **params):
        self._range_cache = {}
        self._curly_cache = {}
        self._arrow_cache = {}
        self._zoom_cb_ref = [None]
        self._last_zoom_t = [0.0]
        self._play_cb     = None
        self._rxy         = None

        # Streams
        self._batch_updating = False
        self._plot_stream = _PlotStream()
        self._vp_stream   = _ViewportStream()

        # Create widgets directly — pmui .from_param() does not wire widget→param
        # so we create widgets with explicit values and watch them manually.
        self._time_idx     = 0
        self._var_w        = pmui.Select(label="Variable", options=list(VARS.keys()), value="temperature")
        _t0 = TIMES[0].strftime("%Y-%m-%d")
        _t1 = TIMES[-1].strftime("%Y-%m-%d")
        self._date_range_label = pn.pane.Markdown(
            f"Range: {_t0} – {_t1}", sizing_mode="stretch_width"
        )
        self._date_input_w = pn.widgets.TextInput(
            name="Date/Time",
            placeholder="YYYY-MM-DD or YYYY-MM-DD HH:MM",
            value=TIMES[0].strftime("%Y-%m-%d %H:%M"),
            sizing_mode="stretch_width",
        )
        self._time_label   = pn.pane.Markdown("", sizing_mode="stretch_width")
        def _level_label(i):
            if i == 0:            return f"0 — surface"
            if i == N_LEVELS - 1: return f"{i} — bottom"
            return str(i)
        _level_opts = {_level_label(i): i for i in range(N_LEVELS)}
        self._level_w = pmui.Select(
            label="Sigma level", options=_level_opts, value=0,
            disabled=N_LEVELS <= 1,
        )
        self._cmap_w       = pmui.Select(label="Palette", options=CMAPS, value="turbo")
        self._curr_mode_w  = pmui.Select(label="Currents overlay", options=CURR_MODES, value="None")
        self._curr_color_w = pmui.Select(label="Currents color", options=CURR_COLORS, value="white")
        self._vector_len_w = pmui.FloatSlider(label="Vector length", start=0.05, end=2.0, step=0.05, value=0.5)
        self._vmin_w       = pn.widgets.FloatInput(name="Min", value=0.0, width=100, disabled=True)
        self._vmax_w       = pn.widgets.FloatInput(name="Max", value=1.0, width=100, disabled=True)
        self._autoscale_w  = pmui.Switch(label="Autoscale", value=True)
        self._prev_btn     = pmui.Button(label="◀", width=50, color="default")
        self._next_btn     = pmui.Button(label="▶", width=50, color="default")
        self._play_btn     = pmui.Toggle(label="▶  Play", color="success")

        super().__init__(**params)

        # Wire all plot-control widgets → _refresh (reads widget values directly)
        for _w in [self._cmap_w, self._curr_mode_w,
                   self._curr_color_w, self._vector_len_w,
                   self._vmin_w, self._vmax_w]:
            _w.param.watch(self._refresh, "value")

        # Date input: parse, snap to nearest time, refresh
        self._date_input_w.param.watch(self._on_date_input, "value")

        # Variable change: reset cmap, then refresh
        self._var_w.param.watch(self._on_var_changed, "value")

        # Level change: refresh
        self._level_w.param.watch(self._on_level_changed, "value")

        # Autoscale toggle: enable/disable vmin/vmax, compute range if switching to manual
        self._autoscale_w.param.watch(self._on_autoscale_changed, "value")

        # Buttons
        self._prev_btn.on_click(self._on_prev)
        self._next_btn.on_click(self._on_next)
        self._play_btn.param.watch(self._on_play_toggle, "value")

        # Build DynamicMaps
        _tiles  = hv.element.tiles.OSM()
        _field  = hv.DynamicMap(self._field_layer, streams=[self._plot_stream])
        _rast   = hv_rasterize(_field)
        _styled = _rast.apply(self._apply_style, streams=[self._plot_stream])

        self._rxy = next((s for s in _rast.streams if isinstance(s, hv.streams.RangeXY)), None)
        if self._rxy:
            self._rxy.param.watch(self._on_zoom, ["x_range", "y_range"])

        _curly  = hv.DynamicMap(self._curly_layer, streams=[self._vp_stream, self._plot_stream])
        _tips   = hv.DynamicMap(self._tips_layer,  streams=[self._vp_stream, self._plot_stream])
        _arrow  = hv.DynamicMap(self._arrow_layer, streams=[self._vp_stream, self._plot_stream])

        _overlay = (_tiles * _styled * _curly * _tips * _arrow).opts(
            hv.opts.Overlay(responsive=True, min_height=650)
        )
        self._plot_pane = pn.pane.HoloViews(_overlay, sizing_mode="stretch_both", min_height=650)

        # Sidebar layout
        with pn.config.set(sizing_mode="stretch_width"):
            self._sidebar = pmui.Column(
                pmui.Typography("FVCOM GOM3 Explorer", variant="h6"),
                self._var_w,
                self._date_range_label,
                self._date_input_w,
                self._time_label,
                pn.Row(self._prev_btn, self._next_btn, self._play_btn),
                self._level_w,
                pmui.Divider(),
                self._cmap_w,
                pn.Row(self._vmin_w, self._vmax_w),
                self._autoscale_w,
                pmui.Divider(),
                self._curr_mode_w,
                self._curr_color_w,
                self._vector_len_w,
            )

        # Initial state
        self._update_time_label()
        self._refresh()

    # ── Stream push (reads from widgets) ──────────────────────────────────────

    def _refresh(self, *_events):
        if self._batch_updating:
            return
        self._plot_stream.event(
            variable=self._var_w.value,
            time_idx=self._time_idx,
            level=self._level_w.value,
            cmap=self._cmap_w.value,
            vmin=self._vmin_w.value,
            vmax=self._vmax_w.value,
            curr_mode=self._curr_mode_w.value,
            curr_color=self._curr_color_w.value,
            vector_len=self._vector_len_w.value,
        )

    # ── Data helpers ──────────────────────────────────────────────────────────

    def _get_uv(self, tidx, level):
        if "siglay" in _RAW_DS.dims and "u" in _RAW_DS and "v" in _RAW_DS:
            u = _RAW_DS["u"].isel(time=tidx, siglay=level).values
            v = _RAW_DS["v"].isel(time=tidx, siglay=level).values
        elif "ua" in _RAW_DS and "va" in _RAW_DS:
            u = _RAW_DS["ua"].isel(time=tidx).values
            v = _RAW_DS["va"].isel(time=tidx).values
        else:
            raise RuntimeError("No velocity variables found")
        if hasattr(u, "compute"): u = u.compute()
        if hasattr(v, "compute"): v = v.compute()
        return u.astype(float), v.astype(float)

    def _get_values(self, variable, tidx, level):
        if variable == "currents":
            u, v = self._get_uv(tidx, level)
            return np.sqrt(u**2 + v**2)
        _, _, has_level, vname = VARS[variable]
        da = _RAW_DS[vname]
        if has_level and "siglay" in da.dims:
            arr = da.isel(time=tidx, siglay=level).values
        elif "time" in da.dims:
            arr = da.isel(time=tidx).values
        else:
            arr = da.values
        if hasattr(arr, "compute"):
            arr = arr.compute()
        return arr

    def _full_range(self, variable, level):
        key = (variable, level)
        if key not in self._range_cache:
            vals = self._get_values(variable, 0, level)
            self._range_cache[key] = (
                round(float(np.nanpercentile(vals, 2)), 4),
                round(float(np.nanpercentile(vals, 98)), 4),
            )
        return self._range_cache[key]

    def _update_time_label(self):
        self._time_label.object = f"**{TIMES[self._time_idx].strftime('%Y-%m-%d %H:%M')}**"

    def _compute_range(self):
        vmin, vmax = self._full_range(self._var_w.value, self._level_w.value)
        self._batch_updating = True
        self._vmin_w.value = vmin
        self._vmax_w.value = vmax
        self._batch_updating = False

    def _get_slice_uda(self, variable, tidx, level):
        if variable == "currents":
            if "siglay" in _RAW_DS.dims and "u" in _RAW_DS and "v" in _RAW_DS:
                u_da = DS["u"].isel(time=tidx, siglay=level)
                v_da = DS["v"].isel(time=tidx, siglay=level)
            elif "ua" in _RAW_DS and "va" in _RAW_DS:
                u_da = DS["ua"].isel(time=tidx)
                v_da = DS["va"].isel(time=tidx)
            else:
                raise RuntimeError("No velocity variables found")
            speed = np.sqrt(u_da**2 + v_da**2)
            speed.name = "currents"
            return speed
        _, _, has_level, vname = VARS[variable]
        da = DS[vname]
        if has_level and "siglay" in da.dims:
            return da.isel(time=tidx, siglay=level)
        elif "time" in da.dims:
            return da.isel(time=tidx)
        return da

    # ── DynamicMap callbacks ──────────────────────────────────────────────────

    def _field_layer(self, variable, time_idx, level, cmap, vmin, vmax,
                     curr_mode, curr_color, vector_len):
        da = self._get_slice_uda(variable, time_idx, level)
        return da.hvplot.trimesh(
            geo=True, xlabel="", ylabel="",
            xlim=(LON_MIN - PAD, LON_MAX + PAD),
            ylim=(LAT_MIN - PAD, LAT_MAX + PAD),
        )

    def _apply_style(self, el, variable, time_idx, level, cmap, vmin, vmax,
                     curr_mode, curr_color, vector_len):
        units, _, _, _ = VARS[variable]
        lv_str   = f" L{level}" if N_LEVELS > 1 else ""
        mode_str = f" | {curr_mode}" if curr_mode != "None" else ""
        title    = f"FVCOM {variable}{lv_str} | {TIMES[time_idx].strftime('%Y-%m-%d %H:%M')}{mode_str}"
        clim_kw  = {} if self._autoscale_w.value else {"clim": (vmin, vmax)}
        if el.vdims:
            el = el.redim(**{el.vdims[0].name: hv.Dimension(units)})
        return el.opts(hv.opts.Image(
            cmap=cmap, colorbar=True, title=title,
            xlabel="Longitude", ylabel="Latitude",
            tools=["hover"], active_tools=["wheel_zoom"],
            responsive=True, min_height=650,
            **clim_kw,
        ))

    def _split_at_land(self, verts):
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

    def _build_curly_paths(self, tidx, level, maxlength, lon_min, lon_max, lat_min, lat_max):
        k = (tidx, level, maxlength,
             round(lon_min, 3), round(lon_max, 3), round(lat_min, 3), round(lat_max, 3))
        if k in self._curly_cache:
            return self._curly_cache[k]
        u_f, v_f = self._get_uv(tidx, level)
        pts = np.column_stack([LONC, LATC])
        iu, iv = LinearNDInterpolator(pts, u_f), LinearNDInterpolator(pts, v_f)
        lons = np.linspace(lon_min, lon_max, CURLY_NX)
        lats = np.linspace(lat_min, lat_max, CURLY_NY)
        lon2d, lat2d = np.meshgrid(lons, lats)
        gpts = np.column_stack([lon2d.ravel(), lat2d.ravel()])
        gu = iu(gpts).reshape(CURLY_NY, CURLY_NX)
        gv = iv(gpts).reshape(CURLY_NY, CURLY_NX)
        fig, ax = plt.subplots(figsize=(8, 8))
        sp = ax.streamplot(
            lons, lats,
            np.where(np.isnan(gu), 0.0, gu),
            np.where(np.isnan(gv), 0.0, gv),
            density=2.5, minlength=0.05, maxlength=maxlength,
            linewidth=1.0, arrowsize=1.0,
        )
        raw_paths = sp.lines.get_paths()
        plt.close(fig)
        xs_ll, ys_ll, tip_lon, tip_lat, tip_ang = [], [], [], [], []
        for path in raw_paths:
            verts = path.vertices
            if len(verts) < 2:
                continue
            for seg in self._split_at_land(verts):
                sx, sy = [p[0] for p in seg], [p[1] for p in seg]
                xs_ll.append(sx)
                ys_ll.append(sy)
                ang = np.arctan2(sy[-1] - sy[-2], sx[-1] - sx[-2])
                u_tip = iu(np.array([[sx[-1], sy[-1]]]))
                v_tip = iv(np.array([[sx[-1], sy[-1]]]))
                if not np.isnan(u_tip[0]) and not np.isnan(v_tip[0]):
                    vel_ang = np.arctan2(float(v_tip[0]), float(u_tip[0]))
                    diff = np.arctan2(np.sin(ang - vel_ang), np.cos(ang - vel_ang))
                    if abs(diff) > np.pi / 2:
                        ang += np.pi
                tip_lon.append(sx[-1])
                tip_lat.append(sy[-1])
                tip_ang.append(float(ang - np.pi / 2))
        result = (xs_ll, ys_ll, tip_lon, tip_lat, tip_ang)
        self._curly_cache[k] = result
        return result

    def _build_arrow_grid(self, tidx, level, lon_min, lon_max, lat_min, lat_max):
        k = (tidx, level,
             round(lon_min, 3), round(lon_max, 3), round(lat_min, 3), round(lat_max, 3))
        if k in self._arrow_cache:
            return self._arrow_cache[k]
        u_f, v_f = self._get_uv(tidx, level)
        pts = np.column_stack([LONC, LATC])
        iu, iv = LinearNDInterpolator(pts, u_f), LinearNDInterpolator(pts, v_f)
        lons = np.linspace(lon_min, lon_max, CURLY_NX)
        lats = np.linspace(lat_min, lat_max, CURLY_NY)
        gpts = np.column_stack([g.ravel() for g in np.meshgrid(lons, lats)])
        gu = iu(gpts).reshape(CURLY_NY, CURLY_NX)
        gv = iv(gpts).reshape(CURLY_NY, CURLY_NX)
        result = (lons, lats, gu, gv)
        self._arrow_cache[k] = result
        return result

    def _curly_layer(self, lon_min, lon_max, lat_min, lat_max,
                     variable, time_idx, level, cmap, vmin, vmax,
                     curr_mode, curr_color, vector_len):
        if curr_mode != "Curly vectors":
            return hv.Path([], kdims=["Longitude", "Latitude"]).opts(apply_ranges=False)
        try:
            xs_ll, ys_ll, _, _, _ = self._build_curly_paths(
                time_idx, level, vector_len, lon_min, lon_max, lat_min, lat_max)
            paths = []
            for xs, ys in zip(xs_ll, ys_ll):
                mx, my = _lonlat_to_merc(np.array(xs), np.array(ys))
                paths.append(list(zip(mx.tolist(), my.tolist())))
            return hv.Path(paths, kdims=["Longitude", "Latitude"]).opts(
                color=curr_color, line_width=1.5, alpha=0.9, apply_ranges=False)
        except Exception:
            return hv.Path([], kdims=["Longitude", "Latitude"]).opts(apply_ranges=False)

    def _tips_layer(self, lon_min, lon_max, lat_min, lat_max,
                    variable, time_idx, level, cmap, vmin, vmax,
                    curr_mode, curr_color, vector_len):
        if curr_mode != "Curly vectors":
            return hv.Points(
                self._EMPTY_TIPS, kdims=["Longitude", "Latitude"], vdims=["angle"],
            ).opts(apply_ranges=False)
        try:
            _, _, tlons, tlats, tangs = self._build_curly_paths(
                time_idx, level, vector_len, lon_min, lon_max, lat_min, lat_max)
            tip_mx, tip_my = _lonlat_to_merc(np.array(tlons), np.array(tlats))
            return hv.Points(
                {"Longitude": tip_mx.tolist(), "Latitude": tip_my.tolist(),
                 "angle": np.degrees(tangs).tolist()},
                kdims=["Longitude", "Latitude"], vdims=["angle"],
            ).opts(marker="triangle", color=curr_color, size=8,
                   angle=hv.dim("angle"), alpha=0.9, apply_ranges=False)
        except Exception:
            return hv.Points(
                self._EMPTY_TIPS, kdims=["Longitude", "Latitude"], vdims=["angle"],
            ).opts(apply_ranges=False)

    def _arrow_layer(self, lon_min, lon_max, lat_min, lat_max,
                     variable, time_idx, level, cmap, vmin, vmax,
                     curr_mode, curr_color, vector_len):
        if curr_mode != "Arrow plot":
            return hv.VectorField(
                self._EMPTY_VF, kdims=["Longitude", "Latitude"], vdims=["Angle", "Magnitude"],
            ).opts(alpha=0, apply_ranges=False)
        try:
            lons, lats, gu, gv = self._build_arrow_grid(
                time_idx, level, lon_min, lon_max, lat_min, lat_max)
            lon2d, lat2d = np.meshgrid(lons, lats)
            lon_flat, lat_flat = lon2d.ravel(), lat2d.ravel()
            mx, my = _lonlat_to_merc(lon_flat, lat_flat)
            gu_f, gv_f = gu.ravel(), gv.ravel()
            mag_f = np.sqrt(gu_f**2 + gv_f**2)
            in_ocean = TRIFINDER(lon_flat, lat_flat) != -1
            valid = ~(np.isnan(gu_f) | np.isnan(gv_f)) & (mag_f > 0.01) & in_ocean
            mx, my   = mx[valid], my[valid]
            gu_f, gv_f = gu_f[valid], gv_f[valid]
            angle    = np.arctan2(gv_f, gu_f)
            mag      = mag_f[valid]
            max_mag  = mag.max()
            mag_norm = mag / max_mag if max_mag > 0 else mag
            return hv.VectorField(
                (mx, my, angle, mag_norm),
                kdims=["Longitude", "Latitude"], vdims=["Angle", "Magnitude"],
            ).opts(color=curr_color, line_color=curr_color, alpha=0.8,
                   magnitude=hv.dim("Magnitude"), scale=vector_len, apply_ranges=False)
        except Exception as e:
            print(f"[arrow] {e}")
            return hv.VectorField(
                self._EMPTY_VF, kdims=["Longitude", "Latitude"], vdims=["Angle", "Magnitude"],
            ).opts(alpha=0, apply_ranges=False)

    # ── Widget-specific handlers ──────────────────────────────────────────────

    def _set_time_idx(self, idx):
        self._time_idx = int(np.clip(idx, 0, N_TIMES - 1))
        self._batch_updating = True
        self._date_input_w.value = TIMES[self._time_idx].strftime("%Y-%m-%d %H:%M")
        self._batch_updating = False
        self._update_time_label()
        self._refresh()

    def _on_date_input(self, event):
        if self._batch_updating:
            return
        raw = event.new.strip()
        if not raw:
            return
        try:
            ts = pd.Timestamp(raw)
        except Exception:
            self._time_label.object = "**Invalid** — use YYYY-MM-DD or YYYY-MM-DD HH:MM"
            return
        self._set_time_idx(int(np.argmin(np.abs(TIMES - ts))))

    def _on_var_changed(self, event):
        _, cmap_default, _, _ = VARS[event.new]
        self._range_cache.clear()
        self._curly_cache.clear()
        self._arrow_cache.clear()
        self._batch_updating = True
        self._cmap_w.value = cmap_default
        self._batch_updating = False
        self._refresh()

    def _on_level_changed(self, event):
        self._refresh()

    def _on_autoscale_changed(self, event):
        if not event.new:  # switching to manual — initialise range from data
            self._compute_range()
            self._vmin_w.disabled = False
            self._vmax_w.disabled = False
        else:              # switching to autoscale — lock out the inputs
            self._vmin_w.disabled = True
            self._vmax_w.disabled = True
        self._refresh()

    # ── Button handlers ───────────────────────────────────────────────────────

    def _on_prev(self, event):
        self._set_time_idx(self._time_idx - 1)

    def _on_next(self, event):
        self._set_time_idx(self._time_idx + 1)

    def _on_play_toggle(self, event):
        if event.new:
            self._play_cb = pn.state.add_periodic_callback(self._play_step, period=500)
        else:
            if self._play_cb:
                self._play_cb.stop()
                self._play_cb = None

    def _play_step(self):
        if self._time_idx + 1 >= N_TIMES:
            self._play_btn.value = False
            return
        self._set_time_idx(self._time_idx + 1)

    # ── Zoom debounce ─────────────────────────────────────────────────────────

    def _on_zoom(self, *args, **kwargs):
        self._last_zoom_t[0] = _time_mod.time()
        if self._zoom_cb_ref[0] is not None:
            return

        def _check():
            if _time_mod.time() - self._last_zoom_t[0] > 0.5:
                self._zoom_cb_ref[0].stop()
                self._zoom_cb_ref[0] = None
                rxy = self._rxy
                if rxy is None or not rxy.x_range or rxy.x_range[0] is None:
                    return
                try:
                    self._vp_stream.event(
                        lon_min=max(float(rxy.x_range[0]), LON_MIN - PAD),
                        lon_max=min(float(rxy.x_range[1]), LON_MAX + PAD),
                        lat_min=max(float(rxy.y_range[0]), LAT_MIN - PAD),
                        lat_max=min(float(rxy.y_range[1]), LAT_MAX + PAD),
                    )
                except Exception as e:
                    print(f"[zoom] {e}")

        self._zoom_cb_ref[0] = pn.state.add_periodic_callback(_check, period=100)

    # ── Layout ────────────────────────────────────────────────────────────────

    def __panel__(self):
        return pmui.Page(
            title="FVCOM GOM3 Viewer",
            sidebar=[self._sidebar],
            main=[pmui.Column(self._plot_pane, sizing_mode="stretch_both")],
        )


FVCOMDashboard().servable()
