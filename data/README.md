# Data layout

Real CryoPPP data is intentionally not stored in this repository. Download it on the
training cluster and create a manifest with one micrograph per row:

```csv
micrograph_path,coordinates_path,split,dataset_id,particle_diameter,pixel_size
/path/to/0001.mrc,/path/to/0001.csv,train,10017,108,1.0
```

The coordinate file must contain `x,y` columns (case-insensitive). RELION STAR files
with `_rlnCoordinateX` and `_rlnCoordinateY` are also supported. Paths may be absolute
or relative to the manifest file.

Split complete micrographs, not pre-cropped patches, to avoid data leakage.

