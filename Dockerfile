FROM condaforge/miniforge3:latest

# Install only the packages needed to run the dashboard
RUN conda install -y -c conda-forge \
        python=3.12 \
        numpy \
        pandas \
        panel \
        holoviews \
        hvplot \
        datashader \
        xarray \
        xugrid \
        matplotlib \
        scipy \
        s3fs \
        fsspec \
        aiobotocore \
        icechunk \
        zarr">=3" \
    && conda clean -afy \
    && pip install --no-cache-dir rustac

WORKDIR /app
COPY dashboard_fvcom.py .

EXPOSE 9002

CMD ["python", "dashboard_fvcom.py"]
