# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains notebooks and code for converting UMASSD FVCOM GOM3 (Gulf of Maine 3) ocean hindcast model data (468 NetCDF3 64-bit offset files on AWS S3) into cloud-optimized Parquet reference files using Kerchunk, enabling efficient access via Zarr/Xarray without copying data.

The final data is accessible at `s3://umassd-fvcom/gom3/hindcast/parquet/combined.parq` (anonymous S3 access).

## Environments

Two Conda environments are used. Create them with Coiled:

```bash
coiled env create --name pangeo-notebook --workspace esip-lab --conda pangeo_notebook_env.yml
coiled env create --name pangeo-worker --workspace esip-lab --conda pangeo_worker_env.yml --arm
```

Launch a notebook:
```bash
coiled notebook start --region us-east-1 --vm-type m5.xlarge --software pangeo-notebook --name pangeo-notebook --workspace esip-lab
```

- `pangeo_notebook_env.yml` — full stack with visualization (holoviews, geoviews, bokeh, jupyterlab)
- `pangeo_worker_env.yml` — compute-only subset for Dask workers (no UI tools)

## Branches

- **`main`** — Kerchunk + Parquet reference workflow (Zarr v2)
- **`virtualizarr-icechunk`** — VirtualiZarr + Icechunk workflow (Zarr v3); see notebooks prefixed `1_` and `2_` and the STAC catalog output

## Data Processing Pipeline

The notebooks implement a sequential pipeline:

1. **`0_create_individual_jsons.ipynb`** — Generates individual Kerchunk JSON reference files from raw NetCDF3 files on S3. Requires AWS credentials. Uses Zarr3 environment.
2. **`create_subchunk_refs.ipynb`** — Creates optimized subchunked reference files from the individual JSONs.
3. **`individual_to_multi.ipynb`** — Consolidates individual JSON refs into a single combined Parquet file using `MultiZarrToZarr` on a Coiled cluster. Uses Zarr2.
4. **`subchunk_to_multi.ipynb`** — Same as above but for subchunked refs.

**VirtualiZarr + Icechunk pipeline** (`virtualizarr-icechunk` branch):
1. **`1_create_virtual_icechunk.ipynb`** — Opens each NetCDF3 as a virtual dataset via VirtualiZarr, rechunks vertical variables to 1 layer per chunk, appends all 468 files into an Icechunk store at `s3://umassd-fvcom/gom3/hindcast/icechunk/gom3.icechunk`. Uses Coiled for parallel opens.
2. **`2_build_stac_catalog.ipynb`** — Reads the Icechunk store, extracts metadata, builds a pystac catalog (one item, datacube extension via xstac), and uploads JSON files to `s3://umassd-fvcom/gom3/hindcast/stac/`.

Exploration/visualization notebooks:
- **`fvcom_gom3_explore.ipynb`** — Loads data via Intake catalog (`umassd-fvcom-gom3.yml`), supports both Zarr v2 and v3 via conditional logic.
- **`fvcom_picker.ipynb`** — Panel-based interactive visualization dashboard for FVCOM data.

## Key Architecture Notes

**Credentials:** AWS credentials for writing to the `umassd-fvcom` bucket are loaded via `load_dotenv(os.path.expanduser('~/dotenv/umassd-fvcom.yml'))`. Reading the source NetCDF3 files and the published Icechunk store uses anonymous S3 access.

**Zarr version duality:** The `main` branch uses Zarr v2 (Kerchunk/Parquet). The `virtualizarr-icechunk` branch uses Zarr v3 (required by Icechunk); the env files on that branch pin `zarr>=3`. Conditional branches (checking `zarr.__version__`) exist in notebooks to handle API differences. The Intake catalog (`umassd-fvcom-gom3.yml`) defines separate entries for each.

**Kerchunk + Parquet pattern:** Raw NetCDF3 files are never copied. Kerchunk generates lightweight reference files that map Zarr chunk addresses to byte ranges within the original S3 files. These references are consolidated into a single Parquet file for efficient lookup.

**Subchunking for vertical layers:** NetCDF3 files are uncompressed, so Kerchunk's subchunk feature is used to split variables that have a vertical (sigma layer) dimension so that each chunk contains only a single layer. This avoids reading the full depth stack when accessing a single layer, which is critical for performance on uncompressed data where the whole variable would otherwise be fetched as one chunk. The `create_subchunk_refs.ipynb` notebook applies this transformation to the individual JSON refs before consolidation.

**FVCOM unstructured mesh:** The model uses a triangular unstructured grid. Visualization requires special handling — use `tricontourf`/`trimesh` or holoviews `TriMesh` rather than standard rectilinear grid tools.

**Intake catalog:** `umassd-fvcom-gom3.yml` is the primary entry point for data consumers. It drops the `dstart` variable on load and configures anonymous S3 access with async support.

**Coiled for distributed processing:** The consolidation step (MultiZarrToZarr over 468 files) requires a distributed Dask cluster managed by Coiled on AWS.
