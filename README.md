# Pre‑flight QA/QC Pipeline for the San Miguel del Vado Land Grant Project

This repository contains a Python‑based pipeline that validates and
preprocesses geospatial datasets for the San Miguel del Vado Land Grant
area in New Mexico.  It enforces a common coordinate reference system
(CRS), detects and optionally repairs geometry and raster issues,
produces cleaned output files and generates a comprehensive report.

## Background

The project CRS is **NAD83(2011) / UTM zone 13N (EPSG: 6342)**.  This
projected coordinate system uses a Transverse Mercator projection and
defines easting and northing axes measured in metres【296392665485662†L58-L67】.
All input data must ultimately be transformed into this CRS.  CRS
information is not inferred; if a dataset is missing a CRS the
pipeline treats it as a critical error and stops processing that file.

Vector geometries can become invalid through editing operations such as
self‑intersections or duplicate vertices.  GeoPandas provides a
`GeoSeries.make_valid()` method which repairs invalid geometries and
returns a new series containing valid shapes【414398350848300†L260-L270】.

For raster data, north‑up orientation is important because non‑zero
rotation terms or positive y pixel sizes can indicate that the image
has been rotated.  The GeoTIFF FAQ states that a typical north‑up
georeference matrix has a positive pixel width, a zero rotation term
and a *negative* pixel height (because image rows run downwards)【831530317450866†L594-L599】.
The pipeline checks each raster’s affine transform and, when the
`--fix` flag is supplied, reprojects rotated rasters into a north‑up
orientation.  Missing or undefined nodata values are recorded from
Rasterio’s `dataset.nodata` property【203913321545165†L88-L93】.

## Installation

1. **Create a Python environment** (recommended for isolation).  On
   Windows, use `py -m venv venv`; on macOS/Linux, use `python3 -m venv venv`.
2. **Activate the environment**:
   * Windows: `venv\Scripts\activate`
   * macOS/Linux: `source venv/bin/activate`
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   The `laspy` library is optional and only needed if you need to
   inspect LAS/LAZ files.  If it is missing the pipeline will still
   copy point cloud files but will not read their CRS.

## Usage

Run the pipeline from the command line:

```bash
python preflight.py --input path/to/raw_data --output path/to/clean_data [--fix] [--strict] [--log-level INFO]
```

Arguments:

| Flag | Description |
| --- | --- |
| `--input` | Root directory containing raw GIS data.  The scan is recursive. |
| `--output` | Directory where cleaned data and reports will be written.  It will be created if necessary. |
| `--fix` | Enable safe automated repairs: invalid vector geometries are repaired, rotated rasters are made north‑up and missing nodata values are preserved.  Without this flag the pipeline only reports issues but does not attempt repairs. |
| `--strict` | Treat warnings as errors.  If any warning is generated the run fails gating and the manifest is not produced. |
| `--log-level` | Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL). |

### Examples

1. **Basic run** – scan an input folder and write outputs under
   `clean_data` without repairing geometries:

   ```bash
   python preflight.py --input ./data/raw --output ./data/clean
   ```

2. **Repair invalid geometries and enforce north‑up rasters**:

   ```bash
   python preflight.py --input ./data/raw --output ./data/clean --fix
   ```

3. **Fail on any warning**:

   ```bash
   python preflight.py --input ./data/raw --output ./data/clean --fix --strict
   ```

## Output

After a run the output directory contains:

* A **mirrored folder structure** with cleaned GIS files.  Vectors are
  reprojected to EPSG:6342 and invalid geometries are fixed when
  possible.  Rasters are reprojected to EPSG:6342, optionally made
  north‑up and preserve nodata metadata【831530317450866†L594-L599】【203913321545165†L88-L93】.
* `qa_qc_report.csv` – a tabular summary of every processed dataset.
  Each row includes the file path, data type, original and target CRS,
  whether reprojection was applied, bounding box and centroid (in
  EPSG:6342), number of geometry issues (vectors), basic raster stats
  (resolution, nodata, bands, data type, min/max) and any warnings or
  errors.
* `qa_qc_report.md` – a human‑readable version of the report with
  bullet‑point summaries for each dataset.  Long sentences are kept
  outside of tables to maintain readability.
* `manifest.json` – a machine‑readable manifest describing the run
  (timestamp, counts of warnings and errors, gating result) and
  listing all output files.  The manifest is only written if the
  gating criteria pass (no critical errors and, when `--strict` is
  supplied, no warnings).

## Critical errors vs. warnings

The pipeline differentiates between **critical errors** and
**warnings**:

| Level | Description | Action |
| --- | --- | --- |
| **Critical error** | Missing CRS, failure to open or reproject a file, failure to write output, corrupt file. | The file is skipped, recorded in the report and the run is marked as failed.  If any critical errors occur the pipeline exits with a non‑zero status and does not create the manifest. |
| **Warning** | Issues that do not prevent processing, such as invalid geometries (when `--fix` is not used), inability to compute statistics or missing nodata values. | The warning is noted in the report and the file is still processed.  When `--strict` is specified, any warning causes gating to fail. |

## Extending to vertical datum checks

The current implementation handles only horizontal coordinate systems.  To
enforce a vertical datum such as NAVD88 in the future, you could:

1. Use `pyproj.CRS` to inspect and compare the vertical component of a
   dataset’s CRS.  Many projected CRSs include an ellipsoidal height
   by default; to enforce NAVD88 you would need to compare against
   EPSG:5703 (NAVD88) or similar.
2. For rasters, use `rasterio.warp.reproject` with a compound CRS
   consisting of the horizontal component (EPSG:6342) and the
   vertical component (EPSG:5703).  Rasterio’s `calculate_default_transform`
   will handle the vertical transformation if GDAL has access to
   appropriate geoid grids.
3. For vector and point cloud data, incorporate PDAL pipelines or
   `pyproj` transformations that include vertical operations.

Such functionality can be incorporated into the helper functions in
`gis_utils.py` without changing the overall architecture of the
pipeline.

## Support and contributions

This pipeline is designed for reproducible research and may be used on
Windows, macOS or Linux.  Contributions via pull request are welcome.
Please ensure that new features are accompanied by documentation and
tests.