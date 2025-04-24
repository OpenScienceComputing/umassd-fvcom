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
