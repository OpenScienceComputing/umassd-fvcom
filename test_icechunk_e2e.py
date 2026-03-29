"""
End-to-end test: write 3 files to a temp Icechunk store, read back, plot one time step.
Uses s3://umassd-fvcom/gom3/hindcast/icechunk/test3.icechunk (overwritten each run).
"""
import dataclasses, json, os, struct, time
import cftime
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import s3fs
import xarray as xr
import icechunk
from kerchunk.netCDF3 import NetCDF3ToZarr
from kerchunk.utils import subchunk
from obstore.store import S3Store
from obspec_utils.registry import ObjectStoreRegistry
from pathlib import Path
from virtualizarr.manifests import ManifestStore, ChunkManifest, ManifestArray
from virtualizarr.parsers.kerchunk.translator import manifestgroup_from_kerchunk_refs
from dotenv import load_dotenv

load_dotenv(os.path.expanduser('~/dotenv/umassd-fvcom.yml'))

SKIP_VARS     = ['Itime', 'Itime2', 'Times', 'file_date', 'iint', 'nprocs']
SIGLEV_VARS   = ['kh', 'km', 'kq', 'l', 'omega', 'q2', 'q2l', 'siglev']
SIGLAY_VARS   = ['salinity', 'siglay', 'temp', 'u', 'v', 'ww']
NLEV, NLAY    = 46, 45
bucket, region = 'umassd-fvcom', 'us-east-1'

# ── File list ─────────────────────────────────────────────────────────────────
fs = s3fs.S3FileSystem(anon=True)
flist = sorted(fs.glob('umassd-fvcom/gom3/hindcast/*.nc'))
flist = [f's3://{f}' for f in flist]
url_0 = flist[0]
TEST_FILES = flist[:3]
print(f'Testing with: {[f.split("/")[-1] for f in TEST_FILES]}')

# ── Build kerchunk template ───────────────────────────────────────────────────
print('\nBuilding kerchunk template...')
refs_0 = NetCDF3ToZarr(url_0, inline_threshold=0, storage_options={'anon': True}).translate()
flat0 = refs_0
for v in SIGLEV_VARS: flat0 = subchunk(flat0, variable=v, factor=NLEV)
for v in SIGLAY_VARS: flat0 = subchunk(flat0, variable=v, factor=NLAY)
ntime_0   = json.loads(flat0['time/.zarray'])['shape'][0]
flat0_data = {k: v for k, v in flat0.items() if not k.startswith('time')}

time_zattrs   = json.loads(flat0['time/.zattrs'])
time_units    = time_zattrs['units']
time_calendar = time_zattrs.get('calendar', 'standard')
_be_dtype     = np.dtype(json.loads(flat0['time/.zarray'])['dtype']).newbyteorder('>')
ref_t0, ref_t1 = flat0['time/0'], flat0['time/1']
with fs.open(url_0, 'rb') as _f:
    _f.seek(ref_t0[1]); t0_val = np.frombuffer(_f.read(ref_t0[2]), dtype=_be_dtype)[0]
    _f.seek(ref_t1[1]); t1_val = np.frombuffer(_f.read(ref_t1[2]), dtype=_be_dtype)[0]
dt_val = t1_val - t0_val

mg       = manifestgroup_from_kerchunk_refs({'version': 1, 'refs': flat0_data},
                                             skip_variables=SKIP_VARS,
                                             fs_root=Path.cwd().as_uri())
_store   = S3Store.from_url(f's3://{bucket}', config={'skip_signature': True, 'region': region})
registry = ObjectStoreRegistry({f's3://{bucket}': _store})
ms       = ManifestStore(group=mg, registry=registry)
template_vds = ms.to_virtual_dataset(loadable_variables=[], decode_times=False)
time_var_names = {name for name, var in template_vds.data_vars.items()
                  if var.dims and var.dims[0] == 'time'}
print(f'Template ready  ntime_0={ntime_0}  dt={dt_val*24:.4f}h')

# ── Helper functions ──────────────────────────────────────────────────────────
def get_ntime_from_header(url, fs):
    with fs.open(url, 'rb') as f:
        hdr = f.read(8)
    assert hdr[:3] == b'CDF'
    return struct.unpack('>i', hdr[4:8])[0]

def clone_vds(url, ntime_n, step_count):
    new_vars = {}
    for name, var in template_vds.data_vars.items():
        if not hasattr(var.data, 'manifest'):
            new_vars[name] = var; continue
        ma = var.data
        new_manifest = ma.manifest.rename_paths(lambda p: p.replace(url_0, url))
        if name in time_var_names and ntime_n < ntime_0:
            m = new_manifest
            new_manifest = ChunkManifest.from_arrays(
                paths=m._paths[:ntime_n], offsets=m._offsets[:ntime_n], lengths=m._lengths[:ntime_n])
            new_meta = dataclasses.replace(ma.metadata, shape=(ntime_n,) + ma.metadata.shape[1:])
        else:
            new_meta = ma.metadata
        new_vars[name] = xr.Variable(var.dims,
                                     ManifestArray(metadata=new_meta, chunkmanifest=new_manifest),
                                     var.attrs)
    raw = t0_val + (step_count + np.arange(ntime_n)) * dt_val
    time_coord = xr.Variable('time', cftime.num2date(raw, time_units, time_calendar))
    return xr.Dataset(new_vars, coords={'time': time_coord}, attrs=template_vds.attrs)

# ── Create test Icechunk store (delete any previous test run first) ───────────
test_prefix = 'gom3/hindcast/icechunk/test3.icechunk'
write_fs = s3fs.S3FileSystem(
    key=os.environ['AWS_ACCESS_KEY_ID'],
    secret=os.environ['AWS_SECRET_ACCESS_KEY'],
)
test_path = f'{bucket}/{test_prefix}'
if write_fs.exists(test_path):
    write_fs.rm(test_path, recursive=True)
    print(f'Deleted existing test store at s3://{test_path}')
config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(
        url_prefix='s3://umassd-fvcom/',
        store=icechunk.s3_store(region=region, anonymous=True),
    ),
)
storage = icechunk.s3_storage(
    bucket=bucket, prefix=test_prefix, region=region,
    access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
)
repo = icechunk.Repository.open_or_create(storage, config)

# ── Write 3 files ─────────────────────────────────────────────────────────────
print('\nWriting 3 files to Icechunk...')
session    = repo.writable_session('main')
step_count = 0
t_write    = time.perf_counter()

for i, f in enumerate(TEST_FILES):
    ntime_n = ntime_0 if f == url_0 else get_ntime_from_header(f, fs)
    vds = clone_vds(f, ntime_n, step_count)
    if i == 0:
        vds.vz.to_icechunk(session.store)
    else:
        vds.vz.to_icechunk(session.store, append_dim='time')
    step_count += ntime_n
    print(f'  [{i+1}/3] {f.split("/")[-1]}  ntime={ntime_n}  step_count={step_count}')

snapshot_id = session.commit('test: 3 FVCOM files')
print(f'Written in {time.perf_counter()-t_write:.1f}s  snapshot={snapshot_id}')

# ── Read back ─────────────────────────────────────────────────────────────────
print('\nReading back...')
credentials = icechunk.containers_credentials({
    's3://umassd-fvcom/': icechunk.s3_credentials(anonymous=True)
})
repo_ro    = icechunk.Repository.open(storage, config, authorize_virtual_chunk_access=credentials)
session_ro = repo_ro.readonly_session('main')
ds = xr.open_zarr(session_ro.store, consolidated=False, chunks=None)
print(ds)
print(f'\ntime[0]  = {ds.time.values[0]}')
print(f'time[-1] = {ds.time.values[-1]}')
print(f'total time steps: {ds.sizes["time"]}  (expected {step_count})')

# Check uniformity (time decoded as datetime64[ns] by xarray)
diffs_ns = np.diff(ds.time.values.astype('datetime64[ns]').astype(np.int64))
print(f'Time spacing (ns): min={diffs_ns.min()}  max={diffs_ns.max()}  uniform={diffs_ns.min() == diffs_ns.max()}')

# ── Diagnostics ───────────────────────────────────────────────────────────────
print('\nDiagnostics:')
print(f'  nv dtype={ds["nv"].dtype}  shape={ds["nv"].shape}')
nv_vals = ds['nv'].values
print(f'  nv values: min={nv_vals.min()}  max={nv_vals.max()}  (expect 1–48451 int)')
temp_surf = ds['temp'].isel(time=-1, siglay=0).values
print(f'  temp dtype={ds["temp"].dtype}  surface range: {temp_surf.min():.2f} to {temp_surf.max():.2f} °C')

# ── Plot last time step, surface temperature ──────────────────────────────────
print('\nPlotting surface temp at last time step...')
tidx = -1
lon = ds['lon'].values
lat = ds['lat'].values
# nv has a byte-order issue when read back from Icechunk (zarr v3 endianness).
# Since nv is static mesh topology, read it directly from the source file.
import scipy.io
with fs.open(url_0, 'rb') as _f:
    nc0 = scipy.io.netcdf.netcdf_file(_f, 'r', mmap=False)
    nv  = nc0.variables['nv'].data.copy() - 1   # FVCOM 1-based → 0-based
    nc0.fp.close()
print(f'  nv from source: min={nv.min()}  max={nv.max()}  (expect 0–48450)')

triang = mtri.Triangulation(lon, lat, nv.T)
fig, ax = plt.subplots(figsize=(10, 6))
tc = ax.tricontourf(triang, temp_surf, levels=20, cmap='RdYlBu_r')
plt.colorbar(tc, ax=ax, label='Temperature (°C)')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.set_title(f'GOM3 Surface Temp — {ds.time.values[tidx]}')
plt.tight_layout()
outfile = 'test_surface_temp.png'
plt.savefig(outfile, dpi=100)
print(f'Saved {outfile}')
plt.close()
