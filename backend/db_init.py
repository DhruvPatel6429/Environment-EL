"""Database initialization utilities.

Creates required tables if they do not already exist. This runs at API startup
so that local development (or a fresh environment) does not crash with
`relation "esg_companies" does not exist` errors.
"""
from pathlib import Path
import logging
import psycopg2
import pandas as pd
import os

from .utils import load_dataframe_to_db

logger = logging.getLogger(__name__)


def _get_simple_connection():
    """Get a simple database connection for initialization operations."""
    try:
        return psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 5432)),
            database=os.getenv("DB_NAME", "esg_db"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
        )
    except Exception as exc:
        logger.warning("PostgreSQL connection failed, falling back to SQLite in db_init: %s", exc)
        import sqlite3
        from pathlib import Path
        db_path = Path(__file__).resolve().parent.parent / "data" / "esg_db.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.isolation_level = None
        return conn


def _table_exists(conn, table_name: str) -> bool:
    is_sqlite = not hasattr(conn, "get_dsn_parameters")
    cur = conn.cursor()
    try:
        if is_sqlite:
            cur.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            row = cur.fetchone()
            return bool(row and row[0] > 0)
        else:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = %s
                );
                """,
                (table_name,),
            )
            row = cur.fetchone()
            return bool(row and row[0])
    finally:
        cur.close()


def run_schema_if_needed():
    """Create core tables if they are missing by executing `01_create_tables.sql` or SQLite equivalent.

    Safe to call multiple times; the SQL uses IF NOT EXISTS.
    """
    conn = _get_simple_connection()
    is_sqlite = not hasattr(conn, "get_dsn_parameters")
    
    if is_sqlite:
        logger.info("Using SQLite database. Schema already initialized by DatabaseService.")
        conn.close()
        return

    base_dir = Path(__file__).resolve().parent
    schema_path = base_dir / "sql" / "01_create_tables.sql"
    if not schema_path.exists():  # pragma: no cover - defensive
        logger.warning("Schema file not found at %s", schema_path)
        return

    # Enable autocommit BEFORE any statements are executed so we can run the full schema script safely
    conn.autocommit = True
    try:
        if _table_exists(conn, "esg_companies"):
            logger.debug("Table esg_companies already exists; skipping schema init.")
            return
        logger.info("Initializing database schema (esg_companies table missing)...")
        sql_text = schema_path.read_text(encoding="utf-8")
        with conn.cursor() as cur:
            cur.execute(sql_text)
        logger.info("Database schema initialized successfully.")
    except psycopg2.Error as exc:  # pragma: no cover - runtime DB errors
        logger.exception("Failed to initialize schema: %s", exc)
    finally:
        conn.close()


def ensure_database_ready():
    """Idempotent public entry point used by the FastAPI startup event."""
    try:
        run_schema_if_needed()
        _bootstrap_data_if_empty()
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error during DB init: %s", exc)


def _bootstrap_data_if_empty():
    """If esg_companies table is empty, attempt to load cleaned CSV.

    Controlled by env var AUTO_LOAD_ESG_CSV (default=1). Set to 0 to disable.
    """
    if os.getenv("AUTO_LOAD_ESG_CSV", "1") not in {"1", "true", "TRUE"}:
        logger.info("AUTO_LOAD_ESG_CSV disabled; skipping automatic data load.")
        return

    conn = _get_simple_connection()
    if hasattr(conn, "autocommit"):
        conn.autocommit = True
    try:
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM esg_companies")
            row = cur.fetchone()
            count = row[0] if row else 0
        finally:
            cur.close()
        if count > 0:
            logger.info("esg_companies already has %s rows; skipping bootstrap load.", count)
            return
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not count esg_companies rows: %s", exc)
        return
    finally:
        conn.close()

    base_dir = Path(__file__).resolve().parent.parent  # project root
    csv_path = base_dir / "data" / "processed" / "esg_data_cleaned.csv"
    if not csv_path.exists():
        logger.info("CSV bootstrap file not found at %s. Attempting to generate from raw dataset...", csv_path)
        raw_csv_path = base_dir / "data" / "raw" / "dataset.csv"
        if raw_csv_path.exists():
            try:
                import sys
                if str(base_dir) not in sys.path:
                    sys.path.append(str(base_dir))
                from scripts.data_preprocessor import load_and_preprocess_data
                load_and_preprocess_data(input_path=raw_csv_path, output_path=csv_path)
                logger.info("Processed dataset generated successfully at %s", csv_path)
            except Exception as e:
                logger.exception("Failed to run data preprocessor: %s", e)
                return
        else:
            logger.warning("Raw CSV file not found at %s. Cannot preprocess.", raw_csv_path)
            return

    try:
        logger.info("Bootstrapping esg_companies from %s...", csv_path)
        df_raw = pd.read_csv(csv_path)
        rename_map = {
            'Symbol': 'symbol',
            'Name': 'name',
            'Address': 'address',
            'Sector': 'sector',
            'Industry': 'industry',
            'Full Time Employees': 'full_time_employees',
            'Description': 'description',
            'Total ESG Risk score': 'total_esg_risk_score',
            'Environment Risk Score': 'environment_risk_score',
            'Governance Risk Score': 'governance_risk_score',
            'Social Risk Score': 'social_risk_score',
            'Controversy Level': 'controversy_level',
            'Controversy Score': 'controversy_score',
            'ESG Risk Percentile': 'esg_risk_percentile',
            'ESG Risk Level': 'esg_risk_level',
        }
        df = df_raw.rename(columns=rename_map)

        # Normalize numeric fields
        numeric_cols = [
            'full_time_employees','total_esg_risk_score','environment_risk_score',
            'social_risk_score','governance_risk_score','controversy_score',
            'esg_risk_percentile'
        ]
        if 'full_time_employees' in df:
            # Strip commas/spaces; convert to numeric; null out impossible values
            cleaned = (
                df['full_time_employees']
                .astype(str)
                .str.replace(',', '', regex=False)
                .str.strip()
                .replace({'': None, 'nan': None, 'None': None})
            )
            df['full_time_employees'] = pd.to_numeric(cleaned, errors='coerce')
            # Replace negative or absurdly large numbers (beyond signed 32-bit) with NaN
            too_large_mask = df['full_time_employees'] > 2_000_000_000
            negative_mask = df['full_time_employees'] < 0
            if too_large_mask.any() or negative_mask.any():
                logger.warning(
                    "Sanitizing %d employee counts (negative or >2B) to NULL", 
                    int(too_large_mask.sum() + negative_mask.sum())
                )
                df.loc[too_large_mask | negative_mask, 'full_time_employees'] = None
            # Use pandas nullable integer type so None stays NULL in DB
            try:
                df['full_time_employees'] = df['full_time_employees'].astype('Int64')
            except Exception:  # pragma: no cover - fallback
                pass
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Scale controversy score from 1-5 to 0-100 to match frontend UI threshold logic (>=60 is high, >=80 is severe)
        if 'controversy_score' in df.columns:
            df['controversy_score'] = df['controversy_score'] * 20

        # Reorder and select known columns only
        table_columns = [
            'symbol','name','address','sector','industry','full_time_employees','description',
            'total_esg_risk_score','environment_risk_score','social_risk_score','governance_risk_score',
            'controversy_level','controversy_score','esg_risk_percentile','esg_risk_level'
        ]
        present_cols = [c for c in table_columns if c in df.columns]
        df = df[present_cols]

        inserted = load_dataframe_to_db(df, 'esg_companies', chunk_size=1000)
        logger.info("Bootstrap load complete: inserted %s rows into esg_companies", inserted)
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed bootstrap data load: %s", exc)
