"""
Command‑line interface for the GIS pre‑flight QA/QC pipeline.
=============================================================

This script implements the entry point for scanning, validating and
preprocessing geospatial datasets in preparation for a project on the
San Miguel del Vado Land Grant area (New Mexico).  It leverages
functions from ``gis_utils.py`` to handle vector, raster and point
cloud data, enforces a common CRS of NAD83(2011) / UTM zone 13N
(EPSG:6342), collects metadata, optionally repairs geometry, and
produces a comprehensive report.

Features
--------

* **Recursive scanning** of an input folder for supported file types
  (vector, raster, LAS/LAZ).
* **CRS validation** with early exit on missing CRS.
* **Reprojection** to the project CRS (EPSG:6342) with optional
  geometry and orientation fixing via the ``--fix`` flag.
* **Reporting**: writes a CSV and a human‑readable Markdown file
  summarising each dataset’s properties, bounding boxes and any
  issues encountered.
* **Gating mode**: if any critical error occurs (missing CRS,
  failure to read or transform, etc.), the script exits with a
  non‑zero code and does not write an “approved” manifest.
  Warnings can optionally be promoted to errors via the ``--strict``
  flag.

Usage
-----

Run the script from the command line::

    python preflight.py --input <input_dir> --output <output_dir> [--fix] [--strict] [--log-level INFO]

The output directory will be created if it does not exist and will
mirror the structure of the input directory.  See ``README.md`` for
more detailed instructions and examples.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd  # type: ignore

from gis_utils import (TARGET_CRS, scan_datasets, process_vector,
                       process_raster, process_las)


def parse_args() -> argparse.Namespace:
    """Define and parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        description="Pre‑flight QA/QC and preprocessing for GIS data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', required=True,
                        help='Path to the input folder containing GIS data')
    parser.add_argument('--output', required=True,
                        help='Path to the output folder for cleaned data')
    parser.add_argument('--fix', action='store_true',
                        help='Enable safe automated repairs (geometry fixes, north‑up rasters)')
    parser.add_argument('--strict', action='store_true',
                        help='Treat warnings as errors and fail gating')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging verbosity')
    return parser.parse_args()


def configure_logging(level: str) -> None:
    """Configure root logger based on CLI argument."""
    numeric_level = getattr(logging, level.upper(), None)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def build_reports(
    input_root: Path,
    output_root: Path,
    fix: bool,
    strict: bool,
) -> Tuple[List[dict], bool, int, int]:
    """Scan datasets, process them and assemble report dictionaries.

    Returns a tuple: (results, gate_failed, critical_count, warning_count).
    ``results`` is a list of dictionaries suitable for conversion to a CSV or
    JSON file. Each dictionary will include keys common to all
    dataset types plus type‑specific fields.
    """
    results: List[dict] = []
    critical_errors = 0
    total_warnings = 0
    for item in scan_datasets(input_root):
        fpath = item['path']
        dtype = item['type']
        logging.info(f"Processing {fpath} ({dtype})")
        if dtype == 'vector':
            report = process_vector(fpath, input_root, output_root, fix)
        elif dtype == 'raster':
            report = process_raster(fpath, input_root, output_root, fix)
        elif dtype == 'point_cloud':
            report = process_las(fpath, input_root, output_root, fix)
        else:
            # Should not be reached
            logging.warning(f"Unknown dataset type for {fpath}")
            continue

        # Convert to plain dict for report; unify errors/warnings counts
        record = report.to_dict()
        results.append(record)
        if record['errors']:
            critical_errors += 1
        if record['warnings']:
            total_warnings += 1
    # Determine gating result.  Any critical errors fail the gate.
    gate_failed = critical_errors > 0 or (strict and total_warnings > 0)
    return results, gate_failed, critical_errors, total_warnings


def write_reports(results: List[dict], output_root: Path, gate_failed: bool) -> None:
    """Write CSV and Markdown reports based on processed results."""
    # CSV
    csv_path = output_root / 'qa_qc_report.csv'
    df = pd.DataFrame(results)
    # Ensure consistent ordering of columns
    # Pandas will handle missing columns automatically
    df.to_csv(csv_path, index=False)
    logging.info(f"Wrote CSV report to {csv_path}")

    # Markdown summary
    md_path = output_root / 'qa_qc_report.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# QA/QC Report\n\n")
        f.write(f"Generated on {datetime.now().isoformat()}\n\n")
        if gate_failed:
            f.write("**Status:** ❌ *FAILED* – critical errors were detected.\n\n")
        else:
            f.write("**Status:** ✅ *APPROVED* – no critical errors.\n\n")
        f.write("## Summary\n\n")
        f.write(f"Processed {len(results)} datasets.\n\n")
        f.write("## Detailed Results\n\n")
        for rec in results:
            f.write(f"### {rec['file_path']}\n")
            f.write(f"- **Type:** {rec.get('dataset_type','')}\n")
            f.write(f"- **Original CRS:** {rec.get('original_crs','NA')}\n")
            f.write(f"- **Target CRS:** {rec.get('target_crs','')}\n")
            f.write(f"- **Reprojected:** {rec.get('reprojection_applied','')}\n")
            if rec.get('bbox'):
                bbox = rec['bbox']
                f.write(f"- **Bounding box (minx, miny, maxx, maxy):** {bbox}\n")
            if rec.get('centroid'):
                centroid = rec['centroid']
                f.write(f"- **Centroid (x, y):** {centroid}\n")
            if rec.get('geometry_issues') is not None:
                f.write(f"- **Geometry issues found:** {rec['geometry_issues']}\n")
                f.write(f"- **Geometry fixed:** {rec.get('geometry_fixed', 0)}\n")
            if rec.get('bands') is not None:
                f.write(f"- **Bands:** {rec['bands']}\n")
                if rec.get('resolution'):
                    f.write(f"- **Resolution (x, y):** {rec['resolution']}\n")
                if rec.get('dtype'):
                    f.write(f"- **Data type:** {rec['dtype']}\n")
                if rec.get('nodata') is not None:
                    f.write(f"- **Nodata value:** {rec['nodata']}\n")
                if rec.get('data_min') is not None:
                    f.write(f"- **Min value:** {rec['data_min']}\n")
                if rec.get('data_max') is not None:
                    f.write(f"- **Max value:** {rec['data_max']}\n")
                north_up = rec.get('north_up')
                if north_up is not None:
                    f.write(f"- **North up:** {north_up}\n")
            if rec.get('warnings'):
                f.write(f"- **Warnings:** {rec['warnings']}\n")
            if rec.get('errors'):
                f.write(f"- **Errors:** {rec['errors']}\n")
            f.write("\n")
    logging.info(f"Wrote Markdown report to {md_path}")


def write_manifest(results: List[dict], output_root: Path,
                   gate_failed: bool, critical_count: int,
                   warning_count: int) -> None:
    """Write a manifest JSON describing the entire run.

    The manifest contains high‑level metadata about the run and each
    output file.  It is only written when gating passes (no critical
    errors).  If gating fails the function simply logs the issue.
    """
    if gate_failed:
        logging.warning("Gating failed; manifest not written")
        return
    manifest = {
        'generated_at': datetime.now().isoformat(),
        'target_crs': TARGET_CRS.to_string(),
        'files_processed': len(results),
        'critical_errors': critical_count,
        'warnings': warning_count,
        'approved': True,
        'outputs': []
    }
    for rec in results:
        # include only datasets that produced outputs
        out_path = rec.get('output_path')
        manifest['outputs'].append({
            'input': rec['file_path'],
            'output': out_path,
            'type': rec.get('dataset_type'),
            'reprojected': rec.get('reprojection_applied'),
            'errors': rec.get('errors'),
            'warnings': rec.get('warnings')
        })
    manifest_path = output_root / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    logging.info(f"Wrote manifest to {manifest_path}")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logging.info(f"Starting pre‑flight QA/QC. Input: {input_root}, Output: {output_root}")
    results, gate_failed, critical_count, warning_count = build_reports(
        input_root, output_root, fix=args.fix, strict=args.strict
    )
    write_reports(results, output_root, gate_failed)
    write_manifest(results, output_root, gate_failed, critical_count, warning_count)
    if gate_failed:
        logging.error("Pre‑flight checks failed. Exiting with error code 1.")
        raise SystemExit(1)
    else:
        logging.info("Pre‑flight checks passed. Exiting with code 0.")
        raise SystemExit(0)


if __name__ == '__main__':
    main()