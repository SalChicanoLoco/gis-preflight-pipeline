"""
Utility functions for GIS QA/QC and preprocessing
=================================================

This module encapsulates the core functionality required for the pre‑flight
quality assurance / quality control (QA/QC) pipeline described in the
user’s specification.  It isolates the file handling, CRS detection,
reprojection, geometry fixing, raster processing and statistics into
reusable functions.  `preflight.py` acts as the command‑line interface
and orchestration layer while delegating the heavy lifting to the
functions defined here.

Key design points
------------------

* Target coordinate reference system (CRS) is hard coded to
  NAD83(2011) / UTM zone 13N (EPSG:6342).  The EPSG definition
  specifies a cartesian 2D coordinate system with east and north axes
  measured in metres.  All datasets are reprojected
  into this CRS when possible.
* Vector data are handled with GeoPandas.  Invalid geometries are
  detected using the `GeoSeries.is_valid` property and can be
  repaired using `GeoSeries.make_valid`, which returns a series of
  valid geometries.  Only safe repairs are
  applied—if `--fix` is not set, invalid geometries are simply
  recorded.
* Raster data are handled with Rasterio.  The affine transform of a
  north‑up raster has zero rotation parameters and a negative
  (southwards) y pixel size; this typical arrangement is documented in
  the GeoTIFF FAQ.  When the source transform
  contains rotation or non‑negative y pixel size, the raster is
  reprojected to a new north‑up orientation.
* Point clouds (LAS/LAZ) are optionally inspected if `laspy` is
  installed.  The current implementation only reads CRS metadata and
  copies the file; full reprojection of point clouds is outside the
  project scope but can be added later.
* Functions return structured dictionaries describing the outcome of
  processing.  These dictionaries form the basis of the QA/QC report
  assembled in the CLI.

The functions herein are deliberately defensive: all file operations
are wrapped in `try/except` blocks so that errors in one dataset do
not halt the entire scan.  Each function records any warnings or
errors encountered so that the caller can decide whether the run
should be gated or allowed to continue.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    import geopandas as gpd
    from shapely.geometry.base import BaseGeometry
except ImportError:
    gpd = None  # type: ignore

try:
    import rasterio
    from rasterio import warp
    from rasterio.enums import Resampling
except ImportError:
    rasterio = None  # type: ignore

try:
    import laspy
except ImportError:
    laspy = None  # type: ignore

try:
    import fiona
except ImportError:
    fiona = None  # type: ignore

from pyproj import CRS


# -----------------------------------------------------------------------------
# Constants and simple data classes
# -----------------------------------------------------------------------------

# Define the target CRS.  NAD83(2011) / UTM zone 13N uses metres as units
# and east/north axes.  See also
# https://epsg.io/6342 for further details.
TARGET_EPSG: int = 6342
TARGET_CRS: CRS = CRS.from_epsg(TARGET_EPSG)
TARGET_CRS_WKT: str = TARGET_CRS.to_wkt()


@dataclass
class VectorReport:
    """Container for reporting on a single vector dataset."""
    file_path: str
    dataset_type: str = "vector"
    original_crs: Optional[str] = None
    target_crs: str = f"EPSG:{TARGET_EPSG}"
    reprojection_applied: bool = False
    bbox: Optional[Tuple[float, float, float, float]] = None
    centroid: Optional[Tuple[float, float]] = None
    geometry_issues: int = 0
    geometry_fixed: int = 0
    warnings: List[str] = None
    errors: List[str] = None
    output_path: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        # convert lists to comma separated strings for CSV friendliness
        data['warnings'] = '; '.join(self.warnings) if self.warnings else ''
        data['errors'] = '; '.join(self.errors) if self.errors else ''
        return data


@dataclass
class RasterReport:
    """Container for reporting on a single raster dataset."""
    file_path: str
    dataset_type: str = "raster"
    original_crs: Optional[str] = None
    target_crs: str = f"EPSG:{TARGET_EPSG}"
    reprojection_applied: bool = False
    north_up: Optional[bool] = None
    resolution: Optional[Tuple[float, float]] = None
    nodata: Optional[float] = None
    bands: Optional[int] = None
    dtype: Optional[str] = None
    data_min: Optional[float] = None
    data_max: Optional[float] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    centroid: Optional[Tuple[float, float]] = None
    warnings: List[str] = None
    errors: List[str] = None
    output_path: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data['warnings'] = '; '.join(self.warnings) if self.warnings else ''
        data['errors'] = '; '.join(self.errors) if self.errors else ''
        # Convert nodata to string only for CSV serialization
        if self.nodata is not None:
            data['nodata'] = str(self.nodata)
        # flatten dtype list into string if present
        if isinstance(self.dtype, (list, tuple)):
            data['dtype'] = ','.join(str(d) for d in self.dtype)
        return data


@dataclass
class LASReport:
    """Container for reporting on a LAS/LAZ point cloud dataset."""
    file_path: str
    dataset_type: str = "point_cloud"
    original_crs: Optional[str] = None
    target_crs: str = f"EPSG:{TARGET_EPSG}"
    reprojection_applied: bool = False
    warnings: List[str] = None
    errors: List[str] = None
    output_path: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data['warnings'] = '; '.join(self.warnings) if self.warnings else ''
        data['errors'] = '; '.join(self.errors) if self.errors else ''
        return data


# -----------------------------------------------------------------------------
# Scanning utilities
# -----------------------------------------------------------------------------

def scan_datasets(input_root: Path) -> Iterator[Dict[str, object]]:
    """Recursively scan an input directory for supported GIS files.

    Supported file extensions include:
      * Vector: `.shp`, `.gpkg`, `.geojson`
      * Raster: `.tif`, `.tiff`
      * Point cloud: `.las`, `.laz`

    Yields dictionaries with keys ``type`` and ``path``.  The caller
    should not mutate the returned objects.
    """
    for root, _, files in os.walk(str(input_root)):
        for fname in files:
            ext = Path(fname).suffix.lower()
            fpath = Path(root) / fname
            if ext in {'.shp', '.gpkg', '.geojson'}:
                yield {'type': 'vector', 'path': fpath}
            elif ext in {'.tif', '.tiff'}:
                yield {'type': 'raster', 'path': fpath}
            elif ext in {'.las', '.laz'}:
                yield {'type': 'point_cloud', 'path': fpath}
            else:
                # unsupported files are ignored
                continue


# -----------------------------------------------------------------------------
# Vector processing
# -----------------------------------------------------------------------------

def process_vector(path: Path, input_root: Path, output_root: Path,
                   fix: bool = False) -> VectorReport:
    """Validate, repair and reproject a vector dataset.

    Parameters
    ----------
    path : Path
        Full path to the vector dataset.
    input_root : Path
        Root of the input directory.  Used to compute relative paths.
    output_root : Path
        Root of the output directory.  Reprojected data are saved under
        this directory preserving the input folder structure.
    fix : bool, optional
        If true, attempt to repair invalid geometries.  If false, record
        invalid geometries but do not modify them.

    Returns
    -------
    VectorReport
        A dataclass summarising the processing outcome.
    """
    report = VectorReport(file_path=str(path), warnings=[], errors=[])
    if gpd is None:
        report.errors.append("geopandas is not installed; cannot process vector data")
        return report

    try:
        gdf = gpd.read_file(path)
    except Exception as exc:
        report.errors.append(f"Failed to read vector file: {exc}")
        return report

    # Check for multiple layers in GeoPackage
    if path.suffix.lower() == '.gpkg' and fiona is not None:
        try:
            layers = fiona.listlayers(str(path))
            if len(layers) > 1:
                report.warnings.append(
                    f"GeoPackage contains {len(layers)} layers; only processing layer '{layers[0]}'. "
                    f"Other layers: {', '.join(layers[1:])}"
                )
        except Exception as exc:
            report.warnings.append(f"Could not enumerate GeoPackage layers: {exc}")

    if gdf.crs is None:
        report.errors.append("Missing CRS; cannot process")
        return report

    report.original_crs = gdf.crs.to_string()

    # Reproject to target CRS if necessary
    try:
        needs_reprojection = False
        src_epsg = gdf.crs.to_epsg()
        if src_epsg is not None and src_epsg == TARGET_EPSG:
            # CRS matches by EPSG code, no reprojection needed
            pass
        elif not CRS.from_user_input(gdf.crs).equals(TARGET_CRS):
            needs_reprojection = True
        
        if needs_reprojection:
            gdf = gdf.to_crs(TARGET_CRS)
            report.reprojection_applied = True
    except Exception as exc:
        report.errors.append(f"Failed to reproject: {exc}")
        return report

    # Validate and optionally fix geometries
    try:
        invalid_mask = ~gdf.geometry.is_valid
    except Exception:
        # some geometry backends may not support is_valid; skip
        invalid_mask = []

    num_invalid = int(invalid_mask.sum()) if hasattr(invalid_mask, 'sum') else 0
    report.geometry_issues = num_invalid
    report.geometry_fixed = 0
    if num_invalid > 0 and fix:
        try:
            # make_valid returns valid geometries without altering valid ones
            gdf['geometry'] = gdf.geometry.make_valid()
            report.geometry_fixed = num_invalid
        except Exception as exc:
            # fallback using buffer(0) if shapely is available
            try:
                gdf['geometry'] = gdf.geometry.buffer(0)
                report.geometry_fixed = num_invalid
            except Exception as exc2:
                report.warnings.append(
                    f"Could not repair {num_invalid} invalid geometries: {exc2}"
                )

    # Calculate bounding box and centroid in target CRS
    try:
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        report.bbox = tuple(float(x) for x in bounds)
        centroid = gdf.unary_union.centroid
        report.centroid = (float(centroid.x), float(centroid.y))
    except Exception as exc:
        report.warnings.append(f"Failed to compute bounds/centroid: {exc}")

    # Determine output path
    rel_path = path.relative_to(input_root)
    out_dir = output_root.joinpath(rel_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir.joinpath(path.name)

    # Write reprojected and fixed data
    try:
        # Determine driver based on file suffix
        suffix = path.suffix.lower()
        driver = None
        if suffix == '.shp':
            driver = 'ESRI Shapefile'
        elif suffix == '.gpkg':
            driver = 'GPKG'
        elif suffix == '.geojson':
            driver = 'GeoJSON'
        # GeoPandas will infer driver if None
        gdf.to_file(out_path, driver=driver)
        report.output_path = str(out_path)
    except Exception as exc:
        report.warnings.append(f"Failed to write output vector file: {exc}")

    return report


# -----------------------------------------------------------------------------
# Raster processing
# -----------------------------------------------------------------------------

def _is_north_up(transform: "rasterio.Affine") -> bool:
    """Check if a raster affine transform describes a north‑up raster.

    According to the GeoTIFF FAQ, a typical north‑up arrangement has a
    positive pixel width and a negative pixel height with zero rotation
    terms.  Here we interpret the affine matrix as:

        \[ A  B  C \]
        \[ D  E  F \]
        \[ 0  0  1 \]

    In Rasterio, ``transform.a`` corresponds to A (x pixel size),
    ``transform.b`` to B (rotation), ``transform.d`` to D (rotation) and
    ``transform.e`` to E (y pixel size).  A north‑up raster should have
    B and D equal to zero (no rotation) and E negative.
    """
    return (transform.b == 0 and transform.d == 0 and transform.e < 0 and transform.a > 0)


def process_raster(path: Path, input_root: Path, output_root: Path,
                   fix: bool = False) -> RasterReport:
    """Validate, repair and reproject a raster dataset.

    Parameters
    ----------
    path : Path
        Full path to the raster dataset (.tif or .tiff).
    input_root : Path
        Root of the input directory.  Used to compute relative paths.
    output_root : Path
        Root of the output directory.  Reprojected data are saved under
        this directory preserving the input folder structure.
    fix : bool, optional
        If true, attempt to correct non‑north‑up rasters and propagate
        nodata values.  If false, non‑north‑up rasters are flagged but
        not modified beyond reprojection.

    Returns
    -------
    RasterReport
        A dataclass summarising the processing outcome.
    """
    report = RasterReport(file_path=str(path), warnings=[], errors=[])
    if rasterio is None:
        report.errors.append("rasterio is not installed; cannot process raster data")
        return report

    try:
        with rasterio.open(path) as src:
            # CRS check
            if src.crs is None:
                report.errors.append("Missing CRS; cannot process")
                return report
            report.original_crs = src.crs.to_string()

            # Orientation check
            transform = src.transform
            report.north_up = _is_north_up(transform)

            # Basic raster metadata
            report.bands = src.count
            # store dtypes as comma separated string later
            report.dtype = [str(dt) for dt in src.dtypes]
            # pixel resolution (absolute values)
            report.resolution = (float(abs(transform.a)), float(abs(transform.e)))
            # nodata value – if multiple bands have different nodata values,
            # rasterio exposes nodatavals; we record the first or join
            nodata_val = src.nodata
            if nodata_val is not None:
                try:
                    report.nodata = float(nodata_val)  # Keep as numeric
                except (ValueError, TypeError):
                    # If conversion fails, log warning and keep as None
                    report.warnings.append(f"Could not convert nodata value '{nodata_val}' to float")
                    report.nodata = None
            else:
                report.nodata = None

            # Compute approximate statistics for the first band.  For
            # large rasters this may be expensive; consider windowed
            # reading but fall back to full read when memory permits.
            try:
                arr = src.read(1, masked=True)
                # ignore masked values when computing min/max
                if hasattr(arr, 'mask'):
                    valid_data = arr.compressed()  # returns only unmasked values
                    if valid_data.size > 0:
                        report.data_min = float(valid_data.min())
                        report.data_max = float(valid_data.max())
                else:
                    report.data_min = float(np.nanmin(arr))
                    report.data_max = float(np.nanmax(arr))
            except Exception as exc:
                report.warnings.append(f"Failed to compute raster statistics: {exc}")

            # Determine bounding box and centroid in target CRS using
            # rasterio.warp.transform_bounds.  The densify_pts argument
            # increases sampling along edges for better accuracy.
            try:
                bounds = warp.transform_bounds(src.crs, TARGET_CRS,
                                               *src.bounds, densify_pts=21)
                report.bbox = tuple(float(x) for x in bounds)
                report.centroid = ((bounds[0] + bounds[2]) / 2.0,
                                   (bounds[1] + bounds[3]) / 2.0)
            except Exception as exc:
                report.warnings.append(f"Failed to compute bounds in target CRS: {exc}")

            # Decide whether to reproject.  Always reproject if CRS
            # differs.  If CRS equals target but the raster is rotated
            # (non‑north‑up) and fix is requested, we also reproject to
            # generate a new north‑up transform.  Rasterio's
            # calculate_default_transform can be used with identical
            # source and destination CRS to achieve this.
            need_reproject = False
            try:
                src_epsg = src.crs.to_epsg()
                if src_epsg is not None and src_epsg == TARGET_EPSG:
                    # CRS matches by EPSG code
                    need_reproject = False
                    if not report.north_up and fix:
                        need_reproject = True  # Only for orientation fix
                elif not src.crs.equals(TARGET_CRS):
                    need_reproject = True
            except Exception:
                # if comparison fails, assume need to reproject
                need_reproject = True

            # Prepare output path
            rel_path = path.relative_to(input_root)
            out_dir = output_root.joinpath(rel_path.parent)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir.joinpath(path.name)
            report.output_path = str(out_path)

            if need_reproject:
                try:
                    dst_crs = TARGET_CRS
                    # compute a destination transform for the target CRS; if
                    # the src.crs matches dst_crs but orientation is wrong,
                    # calculate_default_transform will generate a new north‑up
                    # transform.
                    dst_transform, dst_width, dst_height = warp.calculate_default_transform(
                        src.crs, dst_crs, src.width, src.height, *src.bounds
                    )
                    profile = src.profile.copy()
                    profile.update({
                        'crs': dst_crs,
                        'transform': dst_transform,
                        'width': dst_width,
                        'height': dst_height,
                    })
                    # Ensure nodata stays consistent
                    if nodata_val is not None:
                        profile['nodata'] = nodata_val
                    with rasterio.open(out_path, 'w', **profile) as dst:
                        for i in range(1, src.count + 1):
                            warp.reproject(
                                source=rasterio.band(src, i),
                                destination=rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=dst_transform,
                                dst_crs=dst_crs,
                                resampling=Resampling.nearest,
                                dst_nodata=nodata_val
                            )
                    report.reprojection_applied = True
                except Exception as exc:
                    report.errors.append(f"Failed to reproject raster: {exc}")
            else:
                # CRS matches target and raster is already north‑up
                # Simply copy the file to the output location to
                # preserve provenance.
                try:
                    shutil.copy2(path, out_path)
                    report.reprojection_applied = False
                except Exception as exc:
                    report.errors.append(f"Failed to copy raster: {exc}")
    except Exception as exc:
        report.errors.append(f"Failed to open raster: {exc}")
    return report


# -----------------------------------------------------------------------------
# LAS/LAZ processing
# -----------------------------------------------------------------------------

def process_las(path: Path, input_root: Path, output_root: Path,
                fix: bool = False) -> LASReport:
    """Inspect a LAS/LAZ point cloud.

    Because point cloud reprojection is more complex and requires a more
    specialised toolset (e.g., PDAL), this function currently only
    extracts CRS metadata if possible and copies the original file to
    the output directory.  If `laspy` is not available, the function
    records a warning.  Future extensions could use PDAL pipelines to
    apply reprojection and vertical datum corrections.
    """
    report = LASReport(file_path=str(path), warnings=[], errors=[])

    # Determine output path first
    rel_path = path.relative_to(input_root)
    out_dir = output_root.joinpath(rel_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir.joinpath(path.name)
    report.output_path = str(out_path)

    # Try to read CRS using laspy if available
    if laspy is None:
        report.warnings.append("laspy is not installed; skipping CRS inspection")
    else:
        try:
            las = laspy.read(path)
            crs = None
            try:
                crs = las.header.parse_crs()
            except Exception:
                # parse_crs may not exist on older laspy versions
                crs = None
            if crs is None:
                report.errors.append("Missing CRS in LAS/LAZ header")
            else:
                report.original_crs = str(crs)
                try:
                    las_crs = CRS.from_user_input(crs)
                    las_epsg = las_crs.to_epsg()
                    if las_epsg is not None and las_epsg != TARGET_EPSG:
                        report.warnings.append(
                            f"CRS (EPSG:{las_epsg}) differs from target (EPSG:{TARGET_EPSG}); "
                            "LAS reprojection not implemented"
                        )
                    elif not las_crs.equals(TARGET_CRS):
                        report.warnings.append(
                            "CRS differs from target; LAS reprojection not implemented"
                        )
                except Exception:
                    report.warnings.append(
                        "Could not compare CRS with target; LAS reprojection not implemented"
                    )
        except Exception as exc:
            report.errors.append(f"Failed to read LAS/LAZ file: {exc}")

    # Copy the file to the output (preserve original).  We do this even
    # when laspy isn't available so that the output folder mirrors the
    # input structure.
    try:
        shutil.copy2(path, out_path)
    except Exception as exc:
        report.errors.append(f"Failed to copy LAS/LAZ file: {exc}")

    return report