"""
Convenience script to run the full Part 1 pipeline end-to-end without
Airflow (useful for local testing/demo before/alongside the DAG).

Usage:
    python run_pipeline.py
"""
from scripts import ingest, validate_clean, transform, load_to_postgres
from scripts.pipeline_logger import get_logger

logger = get_logger("run_pipeline")


def main():
    logger.info("=== STAGE 1: INGEST ===")
    ingest.run()

    logger.info("=== STAGE 2: VALIDATE & CLEAN ===")
    clean_summary = validate_clean.run()
    logger.info(f"Clean summary: {clean_summary}")

    logger.info("=== STAGE 3: TRANSFORM ===")
    transform_summary = transform.run()
    logger.info(f"Transform summary: {transform_summary}")

    logger.info("=== STAGE 4: LOAD TO POSTGRES ===")
    try:
        load_summary = load_to_postgres.run()
        logger.info(f"Load summary: {load_summary}")
    except Exception as e:
        logger.error(
            f"Postgres load skipped/failed ({e}). "
            f"Analytical CSVs are still available in data/analytical/ "
            f"for the Streamlit dashboard fallback."
        )

    logger.info("=== PIPELINE RUN COMPLETE ===")


if __name__ == "__main__":
    main()
