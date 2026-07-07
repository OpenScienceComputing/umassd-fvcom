#!/usr/bin/env python
"""
Build Icechunk store for FVCOM GOM3 hindcast (468 NetCDF3 files) — NO subchunking.

Identical to build_icechunk_batched.py except vertical variables (siglay/siglev)
are NOT subchunked. Each variable is stored as one chunk per timestep rather than
one chunk per layer per timestep. This produces far fewer virtual references and
should write significantly faster.

Store: s3://umassd-fvcom/gom3/hindcast/icechunk/gom3-whole-nochunk.icechunk

Use for benchmarking read performance vs the subchunked store (gom3-whole.icechunk).
With no subchunking, reading a single sigma layer requires fetching the full
depth stack — potentially much slower for layer-by-layer access patterns.

Run:
    export $(grep -v '^#' ~/dotenv/umassd-fvcom.env | xargs) && \\
    coiled run --software umassd-fvcom --workspace esip-lab \\
        --region us-east-1 --vm-type r5.xlarge \\
        --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \\
        --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \\
        --detach \\
        -- python -u build_icechunk_nosubchunk.py
"""
import json
import os
import struct
import sys
import time

import cftime
import coiled
import numpy as np
import s3fs
import xarray as xr
import icechunk
from dotenv import load_dotenv
from dask.distributed import Client
from kerchunk.netCDF3 import NetCDF3ToZarr

sys.stdout.reconfigure(line_buffering=True)

_dotenv_path = os.path.expanduser('~/dotenv/umassd-fvcom.env')
if os.path.exists(_dotenv_path):
    load_dotenv(_dotenv_path)
    print(f'Loaded credentials from {_dotenv_path}')

# ── Constants ──────────────────────────────────────────────────────────────────
BATCH_SIZE     = 50
MAX_BATCHES    = None
bucket, region = 'umassd-fvcom', 'us-east-1'
PROD_PREFIX    = 'gom3/hindcast/icechunk/gom3-whole-nochunk.icechunk'

# ── Coiled Dask cluster ────────────────────────────────────────────────────────
cluster = coiled.Cluster(
    n_workers=5,
    software='umassd-fvcom',
    workspace='esip-lab',
    region=region,
    worker_vm_types=['m5.large'],
    scheduler_vm_types=['m5.xlarge'],
    name='fvcom-icechunk-nochunk',
    shutdown_on_close=True,
)
client = Client(cluster)
print(f'Dask dashboard: {client.dashboard_link}')

# ── Source files ───────────────────────────────────────────────────────────────
fs = s3fs.S3FileSystem(anon=True)
flist = sorted(fs.glob('umassd-fvcom/gom3/hindcast/*.nc'))
flist = [f's3://{f}' for f in flist]
url_0 = flist[0]
print(f'{len(flist)} source files')

# ── Icechunk repo ──────────────────────────────────────────────────────────────
from icechunk import ManifestSplitCondition, ManifestSplittingConfig, ManifestSplitDimCondition, ManifestConfig
split_config = ManifestSplittingConfig.from_dict(
    {
        ManifestSplitCondition.AnyArray(): {
            ManifestSplitDimCondition.DimensionName("time"): 365 * 24
        }
    }
)
config = icechunk.RepositoryConfig(
    manifest=ManifestConfig(splitting=split_config),
)
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(
        url_prefix='s3://umassd-fvcom/',
        store=icechunk.s3_store(region=region, anonymous=True),
    ),
)
storage = icechunk.s3_storage(
    bucket=bucket, prefix=PROD_PREFIX, region=region,
    access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
)
repo = icechunk.Repository.open_or_create(storage, config)

# ── Read time metadata from file 0 on coordinator ─────────────────────────────
print('Reading time metadata from file 0 ...')
refs_0 = NetCDF3ToZarr(url_0, inline_threshold=0,
                        storage_options={'anon': True}).translate()['refs']
time_zattrs   = json.loads(refs_0['time/.zattrs'])
time_units    = time_zattrs['units']
time_calendar = time_zattrs.get('calendar', 'standard')
_be_dtype     = np.dtype(json.loads(refs_0['time/.zarray'])['dtype']).newbyteorder('>')
ref_t0, ref_t1 = refs_0['time/0'], refs_0['time/1']
with fs.open(url_0, 'rb') as _f:
    _f.seek(ref_t0[1]); t0_val = float(np.frombuffer(_f.read(ref_t0[2]), dtype=_be_dtype)[0])
    _f.seek(ref_t1[1]); t1_val = float(np.frombuffer(_f.read(ref_t1[2]), dtype=_be_dtype)[0])
dt_val = float(t1_val - t0_val)

# Snap to integer hours — avoids float64 drift (1/24 day is not exactly representable)
assert 'days since' in time_units, f'Unexpected time units: {time_units!r}'
dt_hours    = int(round(dt_val * 24))
t0_hours    = int(round(t0_val * 24))
hours_units = time_units.replace('days since', 'hours since', 1)
print(f'  units={time_units!r}  dt={dt_hours}h  '
      f't0={cftime.num2date(t0_hours, hours_units, time_calendar)}')

# ── UGRID / CF metadata ────────────────────────────────────────────────────────
CF_VAR_ATTRS = {
    'time':     {'standard_name': 'time'},
    'h':        {'standard_name': 'sea_floor_depth_below_geoid', 'units': 'm',
                 'coordinates': 'lat lon'},
    'zeta':     {'standard_name': 'sea_surface_height_above_geoid', 'units': 'meters',
                 'coordinates': 'time lat lon', 'coverage_content_type': 'modelResult'},
    'temp':     {'standard_name': 'sea_water_potential_temperature',
                 'coordinates': 'time siglay lat lon', 'coverage_content_type': 'modelResult'},
    'salinity': {'standard_name': 'sea_water_salinity', 'units': '0.001',
                 'coordinates': 'time siglay lat lon', 'coverage_content_type': 'modelResult'},
    'u':        {'standard_name': 'eastward_sea_water_velocity', 'units': 'meters s-1',
                 'coordinates': 'time siglay latc lonc', 'coverage_content_type': 'modelResult'},
    'v':        {'standard_name': 'northward_sea_water_velocity', 'units': 'meters s-1',
                 'coordinates': 'time siglay latc lonc', 'coverage_content_type': 'modelResult'},
    'ww':       {'standard_name': 'upward_sea_water_velocity', 'units': 'meters s-1',
                 'coordinates': 'time siglay latc lonc', 'coverage_content_type': 'modelResult'},
    'ua':       {'standard_name': 'barotropic_eastward_sea_water_velocity', 'units': 'meters s-1',
                 'coordinates': 'time latc lonc', 'coverage_content_type': 'modelResult'},
    'va':       {'standard_name': 'northward_barotropic_sea_water_velocity', 'units': 'meters s-1',
                 'coordinates': 'time latc lonc', 'coverage_content_type': 'modelResult'},
    'siglay':   {'standard_name': 'ocean_sigma_coordinate', 'positive': 'up',
                 'valid_min': -1.0, 'valid_max': 0.0,
                 'formula_terms': 'sigma: siglay eta: zeta depth: h'},
    'nv':       {'long_name': 'nodes surrounding element',
                 'cf_role': 'face_node_connectivity', 'start_index': 1},
}

mesh_topology_ds = xr.Dataset({'mesh_topology': xr.Variable((), np.int32(0), attrs={
    'cf_role': 'mesh_topology', 'topology_dimension': 2,
    'node_coordinates': 'lon lat', 'face_coordinates': 'lonc latc',
    'face_node_connectivity': 'nv', 'face_dimension': 'nele',
})})

def add_ugrid_metadata(ds):
    ds.attrs['Conventions'] = 'CF-1.11'
    for var, attrs in CF_VAR_ATTRS.items():
        if var in ds:
            ds[var].attrs.update(attrs)
    for var in ds.data_vars:
        dims = ds[var].dims
        if 'node' in dims or 'nele' in dims:
            ds[var].attrs.setdefault('mesh', 'mesh_topology')
            ds[var].attrs.setdefault('location', 'face' if 'nele' in dims else 'node')
    return xr.merge([ds, mesh_topology_ds], compat='override', combine_attrs='no_conflicts')

# ── Worker functions ───────────────────────────────────────────────────────────
def get_ntime_remote(url):
    """Read 8-byte NetCDF3 header for numrecs."""
    import struct, s3fs
    fs = s3fs.S3FileSystem(anon=True)
    with fs.open(url, 'rb') as f:
        hdr = f.read(8)
    assert hdr[:3] == b'CDF', f'Not a NetCDF3 file: {url}'
    return struct.unpack('>i', hdr[4:8])[0]


def make_vds_remote(url, step_count, t0_hours, dt_hours, hours_units, time_calendar):
    """
    Kerchunk + VirtualiZarr pipeline for one file — NO subchunking.
    All inputs are simple scalars — nothing large is serialized.
    Time is constructed from integer hours to avoid float64 drift.
    """
    import json
    import numpy as np
    import xarray as xr
    from kerchunk.netCDF3 import NetCDF3ToZarr
    from obspec_utils.registry import ObjectStoreRegistry
    from obstore.store import S3Store
    from pathlib import Path
    from virtualizarr.manifests import ManifestStore
    from virtualizarr.parsers.kerchunk.translator import manifestgroup_from_kerchunk_refs

    SKIP_VARS = ['Itime', 'Itime2', 'Times', 'file_date', 'iint', 'nprocs']
    bucket    = 'umassd-fvcom'
    region    = 'us-east-1'

    # No subchunking — translate refs as-is
    refs = NetCDF3ToZarr(url, inline_threshold=0,
                          storage_options={'anon': True}).translate()
    # translate() may return nested {'version':1,'refs':{...}} — unwrap if so
    if 'refs' in refs:
        refs = refs['refs']

    ntime     = json.loads(refs['time/.zarray'])['shape'][0]
    refs_data = {k: v for k, v in refs.items() if not k.startswith('time')}

    _store   = S3Store.from_url(f's3://{bucket}',
                                 config={'skip_signature': True, 'region': region})
    registry = ObjectStoreRegistry({f's3://{bucket}': _store})
    mg  = manifestgroup_from_kerchunk_refs(
        {'version': 1, 'refs': refs_data},
        skip_variables=SKIP_VARS,
        fs_root=Path.cwd().as_uri(),
    )
    ms  = ManifestStore(group=mg, registry=registry)
    vds = ms.to_virtual_dataset(loadable_variables=[], decode_times=False)

    raw_hours  = t0_hours + (step_count + np.arange(ntime, dtype=np.int64)) * dt_hours
    time_coord = xr.Variable('time', raw_hours,
                             attrs={'units': hours_units, 'calendar': time_calendar})
    return xr.Dataset(dict(vds.data_vars), coords={'time': time_coord}, attrs=vds.attrs)


# ── Read all ntimes in parallel ────────────────────────────────────────────────
print('Reading all file headers in parallel ...')
t0_hdrs   = time.perf_counter()
ntimes    = np.array(client.gather(client.map(get_ntime_remote, flist)), dtype=np.int32)
cumsteps  = np.concatenate([[0], np.cumsum(ntimes)])
total_expected = int(cumsteps[-1])
print(f'  {time.perf_counter()-t0_hdrs:.1f}s  total expected steps: {total_expected:,}')

# ── Resume: check existing committed steps ─────────────────────────────────────
try:
    credentials = icechunk.containers_credentials({
        's3://umassd-fvcom/': icechunk.s3_credentials(anonymous=True)
    })
    repo_ro     = icechunk.Repository.open(storage, config,
                                           authorize_virtual_chunk_access=credentials)
    session_ro  = repo_ro.readonly_session('main')
    ds_existing = xr.open_zarr(session_ro.store, consolidated=False, chunks=None)
    n_existing  = len(ds_existing.time)
    print(f'Store has {n_existing:,} committed steps '
          f'({n_existing / total_expected * 100:.1f}% of {total_expected:,})')
    if n_existing > 0:
        print(f'  time range: {ds_existing.time.values[0]} → {ds_existing.time.values[-1]}')
except Exception as e:
    n_existing = 0
    print(f'Store is empty or new: {e}')

start_file_idx = int(np.searchsorted(cumsteps, n_existing, side='left'))
if start_file_idx >= len(flist):
    print('All files already committed — nothing to do.')
    client.close(); cluster.close()
    raise SystemExit(0)

step_count  = int(cumsteps[start_file_idx])
first_write = (n_existing == 0)
n_batches   = (len(flist) - start_file_idx + BATCH_SIZE - 1) // BATCH_SIZE
print(f'Starting from file {start_file_idx}  step_count={step_count:,}  '
      f'({n_batches} batches remaining)')

# ── Identify time-varying variables ───────────────────────────────────────────
print('Building one vds on coordinator to identify time-varying variables ...')
_vds0 = make_vds_remote(url_0, 0, t0_hours, dt_hours, hours_units, time_calendar)
time_var_names = frozenset(
    name for name, var in _vds0.data_vars.items()
    if var.dims and var.dims[0] == 'time'
)
print(f'  {len(time_var_names)} time-varying vars, '
      f'{len(_vds0.data_vars) - len(time_var_names)} static vars')
del _vds0

# ── Main batch loop ────────────────────────────────────────────────────────────
t0_total = time.perf_counter()
session  = repo.writable_session('main')

for batch_start in range(start_file_idx, len(flist), BATCH_SIZE):
    batch_num_so_far = (batch_start - start_file_idx) // BATCH_SIZE + 1
    if MAX_BATCHES is not None and batch_num_so_far > MAX_BATCHES:
        print(f'Reached MAX_BATCHES={MAX_BATCHES}, stopping.')
        break
    t0_batch     = time.perf_counter()
    batch_files  = flist[batch_start : batch_start + BATCH_SIZE]
    batch_ntimes = ntimes[batch_start : batch_start + BATCH_SIZE]
    batch_steps  = [int(cumsteps[batch_start + i]) for i in range(len(batch_files))]
    batch_num    = (batch_start - start_file_idx) // BATCH_SIZE + 1
    batch_end    = min(batch_start + BATCH_SIZE - 1, len(flist) - 1)
    print(f'  Batch {batch_num}/{n_batches}: submitting files {batch_start}–{batch_end} ...')

    batch_vds = client.gather(
        client.map(make_vds_remote, batch_files, batch_steps,
                   t0_hours=t0_hours, dt_hours=dt_hours,
                   hours_units=hours_units, time_calendar=time_calendar)
    )
    print(f'  Batch {batch_num}/{n_batches}: gathered {len(batch_vds)} vds  '
          f'({time.perf_counter()-t0_batch:.1f}s)')

    print(f'  Batch {batch_num}/{n_batches}: concat ...')
    time_only    = [vds[list(time_var_names)] for vds in batch_vds]
    batch_concat = xr.concat(time_only, dim='time',
                             data_vars='minimal', coords='minimal',
                             compat='override')

    print(f'  Batch {batch_num}/{n_batches}: to_icechunk ...')
    if first_write:
        static_vars = {k: v for k, v in batch_vds[0].data_vars.items()
                       if k not in time_var_names}
        full_batch  = xr.merge([batch_concat, xr.Dataset(static_vars)],
                               compat='override')
        full_batch  = add_ugrid_metadata(full_batch)
        full_batch.vz.to_icechunk(session.store)
        first_write = False
    else:
        batch_concat.vz.to_icechunk(session.store, append_dim='time')

    print(f'  Batch {batch_num}/{n_batches}: committing ...')
    step_count += int(batch_ntimes.sum())
    snap        = session.commit(
        f'Batch files {batch_start}–{batch_end} | '
        f'steps {step_count - int(batch_ntimes.sum())}–{step_count}'
    )
    session = repo.writable_session('main')

    pct           = step_count / total_expected * 100
    elapsed_batch = time.perf_counter() - t0_batch
    print(f'  Batch {batch_num}/{n_batches}: done — files {batch_start}–{batch_end}  '
          f'+{int(batch_ntimes.sum()):,} steps  total={step_count:,} ({pct:.1f}%)  '
          f'{elapsed_batch:.1f}s  snap={snap[:8]}')

total_elapsed = time.perf_counter() - t0_total
print(f'\nComplete! {step_count:,} time steps, {len(flist)} files, '
      f'{total_elapsed / 60:.1f} min total')

client.close()
cluster.close()
