"""
STAGE 1 - INGESTION
Extracts the UCI Student Performance dataset (or ERP/LMS CSV exports placed
in the same folder), keeps an untouched raw copy, and records an extraction
manifest (source, timestamp, status, row count) as required by the
assignment brief.

Usage:
    python -m scripts.ingest
    python -m scripts.ingest --source /path/to/custom_export.csv --name erp_export
"""
import argparse
import io
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from scripts.config import RAW_DATA_DIR, LOG_DIR, UCI_MAT_URL, SOURCE_FILES
from scripts.pipeline_logger import get_logger

logger = get_logger("ingest")

MANIFEST_PATH = LOG_DIR / "ingestion_manifest.csv"


def _append_manifest(record: dict) -> None:
    df = pd.DataFrame([record])
    header = not MANIFEST_PATH.exists()
    df.to_csv(MANIFEST_PATH, mode="a", header=header, index=False)


def download_uci_dataset(dest_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """Downloads and unzips the public UCI student.zip archive."""
    extracted = []
    try:
        logger.info(f"Requesting UCI dataset from {UCI_MAT_URL}")
        resp = requests.get(UCI_MAT_URL, timeout=30)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            for fname in SOURCE_FILES.values():
                if fname in z.namelist():
                    z.extract(fname, dest_dir)
                    extracted.append(dest_dir / fname)
        logger.info(f"Downloaded and extracted: {[p.name for p in extracted]}")
    except Exception as e:
        logger.error(f"UCI download failed ({e}). Falling back to local files if present.")
    return extracted


def ingest_file(source_path: Path, source_name: str) -> dict:
    """
    Copies a single source CSV into the raw layer untouched and logs an
    extraction manifest record. Any read failure is caught and logged as a
    failed extraction rather than crashing the whole pipeline run.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    status = "SUCCESS"
    row_count = 0
    error_msg = ""

    try:
        df = pd.read_csv(source_path, sep=None, engine="python")  # auto-detect ; or ,
        row_count = len(df)

        raw_target = RAW_DATA_DIR / f"{source_name}_{ts}.csv"
        shutil.copyfile(source_path, raw_target)
        logger.info(f"Ingested '{source_name}' -> {raw_target.name} ({row_count} rows)")
    except Exception as e:
        status = "FAILED"
        error_msg = str(e)
        logger.error(f"Failed to ingest {source_path}: {e}")

    record = {
        "source_name": source_name,
        "source_path": str(source_path),
        "extraction_timestamp_utc": ts,
        "status": status,
        "row_count": row_count,
        "error_message": error_msg,
    }
    _append_manifest(record)
    return record


def run(custom_source: str = None, custom_name: str = None) -> list[dict]:
    results = []

    if custom_source:
        results.append(ingest_file(Path(custom_source), custom_name or Path(custom_source).stem))
        return results

    # Try local files first (student-mat.csv / student-por.csv placed manually)
    found_local = False
    for name, fname in SOURCE_FILES.items():
        local_path = RAW_DATA_DIR / fname
        if local_path.exists():
            found_local = True
            results.append(ingest_file(local_path, name))

    if not found_local:
        downloaded = download_uci_dataset()
        for path in downloaded:
            name = [k for k, v in SOURCE_FILES.items() if v == path.name]
            results.append(ingest_file(path, name[0] if name else path.stem))

    if not results:
        logger.warning(
            "No source files found or downloaded. Place student-mat.csv / "
            "student-por.csv in data/raw/ and re-run."
        )

    summary_path = LOG_DIR / "last_ingestion_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest student data sources")
    parser.add_argument("--source", help="Path to a custom CSV export (ERP/LMS)")
    parser.add_argument("--name", help="Logical name for the custom source")
    args = parser.parse_args()
    run(custom_source=args.source, custom_name=args.name)
