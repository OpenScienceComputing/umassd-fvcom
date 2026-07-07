# umassd-fvcom
information and code related to the UMASSD FVCOM GOM3 data on AWS Open Data

## Virtual Dataset Construction
Construction of Parquet references for the 468 NetCDF3 64-bit offset files
```
$ coiled env create --name pangeo-notebook --workspace esip-lab --conda pangeo_notebook_env.yml 
$ coiled env create --name pangeo-worker --workspace esip-lab --conda pangeo_worker_env.yml --arm
```
Notebook launched with:
```
coiled notebook start --region us-east-1 --vm-type m5.xlarge --software pangeo-notebook --name pangeo-notebook --workspace esip-lab
```
and the notebook then launches a Coiled cluster that aggregates the references.

## Virtual Icechunk Store (Zarr v3)

On the `virtualizarr-icechunk` branch the same 468 NetCDF3 files are exposed as a
single [Icechunk](https://icechunk.io) store (Zarr v3) built with
[VirtualiZarr](https://virtualizarr.readthedocs.io) — again without copying any
data. Icechunk records byte-range references into the original NetCDF files as
*virtual chunks*, so readers stream directly from the source objects.

Published store (anonymous read):
```
s3://umassd-fvcom/gom3/hindcast/icechunk/gom3-whole-nochunk.icechunk
```

### How the references are built

`build_icechunk_nosubchunk.py` builds the store on a Coiled Dask cluster:

1. **Per-file virtual datasets.** Each NetCDF3 file is opened independently with
   kerchunk (`NetCDF3ToZarr`) and converted to a VirtualiZarr virtual dataset.
   These scans are fanned out across the cluster with `client.map`. Because every
   file is scanned on its own, the same pipeline generalizes to NetCDF4/HDF5 by
   swapping in VirtualiZarr's HDF parser.
2. **Exact time coordinates.** Time is reconstructed as integer *hours since* the
   epoch (`t0 + step*dt`) to avoid the float64 drift that comes from the original
   `days since` units.
3. **UGRID/CF metadata.** A scalar `mesh_topology` variable (with
   `face_node_connectivity`, `node_coordinates`, `face_coordinates`, …) plus CF
   `standard_name`, `units`, and `coordinates` attributes are added so the
   unstructured triangular grid is self-describing and readable by tools such as
   [`xugrid`](https://deltares.github.io/xugrid/).
4. **Batched appends.** Virtual datasets are appended to the Icechunk `main`
   branch in batches of 50 along the `time` dimension, then committed.

Run it (detached) on Coiled:
```bash
export $(grep -v '^#' ~/dotenv/umassd-fvcom.env | xargs)
coiled run --software umassd-fvcom --workspace esip-lab \
    --region us-east-1 --vm-type r5.xlarge \
    --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    --detach \
    -- python -u build_icechunk_nosubchunk.py
```

A faster NetCDF3-specific variant, `create_virtual_icechunk.ipynb`, avoids
re-scanning every file: it builds a reference template from file 0 and clones it
to the other 467 files by URL substitution, reading only an 8-byte header from
each file to get its number of time steps. This is much faster, but it relies on
all files sharing identical internal byte offsets, so it does **not** generalize
beyond uncompressed NetCDF3.

### Reading the store

The store metadata is public and the virtual chunks live in the public
`umassd-fvcom` bucket, so everything can be read anonymously:

```python
import icechunk
import xarray as xr

bucket = 'umassd-fvcom'
prefix = 'gom3/hindcast/icechunk/gom3-whole-nochunk.icechunk'
region = 'us-east-1'

# Tell Icechunk the virtual chunks in this bucket are fetched anonymously
config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(icechunk.VirtualChunkContainer(
    url_prefix=f's3://{bucket}/',
    store=icechunk.s3_store(region=region, anonymous=True),
))
credentials = icechunk.containers_credentials({
    f's3://{bucket}/': icechunk.s3_credentials(anonymous=True)
})

storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, region=region, anonymous=True)
repo = icechunk.Repository.open(storage, config,
                                authorize_virtual_chunk_access=credentials)
ds = xr.open_zarr(repo.readonly_session('main').store,
                  consolidated=False, chunks=None)
```

