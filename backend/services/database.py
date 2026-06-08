import psycopg2
from psycopg2 import sql, pool
from psycopg2.extras import RealDictCursor
import sqlite3
import os
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import logging

from ..core.config import settings

logger = logging.getLogger(__name__)


class DatabaseService:
    _instance = None
    _connection_pool = None
    use_sqlite = False
    sqlite_db_path = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseService, cls).__new__(cls)
            cls._instance._initialize_pool()
        return cls._instance
    
    def _initialize_pool(self):
        try:
            self._connection_pool = pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=20,
                host=settings.DB_HOST,
                port=settings.DB_PORT,
                database=settings.DB_NAME,
                user=settings.DB_USER,
                password=settings.DB_PASSWORD,
                cursor_factory=RealDictCursor,
                connect_timeout=10
            )
            logger.info("Database connection pool initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize database pool: {e}. Falling back to SQLite.")
            self.use_sqlite = True
            self._connection_pool = None
            
            # Setup SQLite database path
            from pathlib import Path
            db_dir = Path(__file__).resolve().parent.parent.parent / "data"
            db_dir.mkdir(parents=True, exist_ok=True)
            self.sqlite_db_path = str(db_dir / "esg_db.sqlite")
            
            # Make sure tables exist
            self._initialize_sqlite_db()
            
    def _initialize_sqlite_db(self):
        conn = sqlite3.connect(self.sqlite_db_path)
        try:
            cursor = conn.cursor()
            
            # Create esg_companies table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS esg_companies (
                symbol TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                address TEXT,
                sector TEXT,
                industry TEXT,
                full_time_employees INTEGER,
                description TEXT,
                total_esg_risk_score REAL,
                environment_risk_score REAL,
                social_risk_score REAL,
                governance_risk_score REAL,
                controversy_level TEXT,
                controversy_score REAL,
                esg_risk_percentile REAL,
                esg_risk_level TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
            )
            """)
            
            # Create model_predictions table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_symbol TEXT NOT NULL,
                predicted_risk_level TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities TEXT NOT NULL,
                input_features TEXT NOT NULL,
                model_version TEXT DEFAULT '1.0',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                FOREIGN KEY (company_symbol) REFERENCES esg_companies(symbol) ON DELETE CASCADE
            )
            """)
            
            # Create news_cache table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_symbol TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                articles TEXT NOT NULL,
                source TEXT DEFAULT 'newsapi',
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                UNIQUE(company_symbol, query_hash)
            )
            """)
            
            # Create agent_analysis_cache table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_analysis_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_symbol TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                result TEXT NOT NULL,
                agents_involved TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                UNIQUE(company_symbol, analysis_type)
            )
            """)
            
            conn.commit()
            logger.info("SQLite database schema initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            raise
        finally:
            conn.close()
    
    @contextmanager
    def get_connection(self):
        if self.use_sqlite:
            conn = sqlite3.connect(self.sqlite_db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"SQLite database error: {e}")
                raise
            finally:
                conn.close()
        else:
            conn = self._connection_pool.getconn()
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Database error: {e}")
                raise
            finally:
                self._connection_pool.putconn(conn)
    
    def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch_one: bool = False
    ) -> Optional[List[Dict[str, Any]]]:
        if self.use_sqlite:
            # Convert %s placeholders to ?
            query = query.replace("%s", "?")
            # Replace ILIKE with LIKE
            query = query.replace("ILIKE", "LIKE")
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    if params is not None:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                    
                    if fetch_one:
                        result = cursor.fetchone()
                        return dict(result) if result else None
                    else:
                        results = cursor.fetchall()
                        return [dict(row) for row in results]
                finally:
                    cursor.close()
        else:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    
                    if fetch_one:
                        result = cursor.fetchone()
                        return dict(result) if result else None
                    else:
                        results = cursor.fetchall()
                        return [dict(row) for row in results]
    
    def get_health_status(self) -> Dict[str, Any]:
        try:
            result = self.execute_query("SELECT 1 as status", fetch_one=True)
            return {
                "status": "healthy",
                "database": "connected",
                "type": "sqlite" if self.use_sqlite else "postgres"
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "database": "disconnected", "error": str(e)}
    
    def get_top_companies(self, limit: int = 10) -> List[Dict[str, Any]]:
        query = """
            SELECT
                symbol,
                name,
                sector,
                total_esg_risk_score
            FROM esg_companies
            WHERE total_esg_risk_score IS NOT NULL
            ORDER BY total_esg_risk_score ASC
            LIMIT %s
        """
        return self.execute_query(query, (limit,))

    def get_all_companies(self, limit: int = 500) -> List[Dict[str, Any]]:
        query = """
            SELECT
                symbol,
                name,
                sector,
                total_esg_risk_score,
                esg_risk_level
            FROM esg_companies
            WHERE total_esg_risk_score IS NOT NULL
            ORDER BY total_esg_risk_score ASC
            LIMIT %s
        """
        return self.execute_query(query, (limit,))
    
    def get_sector_averages(self) -> List[Dict[str, Any]]:
        query = """
            SELECT 
                sector,
                AVG(total_esg_risk_score) as avg_esg_score,
                COUNT(*) as company_count
            FROM esg_companies
            WHERE sector IS NOT NULL 
              AND total_esg_risk_score IS NOT NULL
            GROUP BY sector
            ORDER BY avg_esg_score ASC
        """
        return self.execute_query(query)
    
    def get_high_controversy_companies(self, min_score: float = 50.0) -> List[Dict[str, Any]]:
        query = """
            SELECT 
                symbol,
                name,
                controversy_score,
                controversy_level
            FROM esg_companies
            WHERE controversy_score >= %s
            ORDER BY controversy_score DESC
        """
        return self.execute_query(query, (min_score,))
    
    def get_company_by_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        query = """
            SELECT 
                symbol,
                name,
                sector,
                industry,
                total_esg_risk_score,
                environment_risk_score,
                social_risk_score,
                governance_risk_score,
                controversy_score,
                controversy_level,
                esg_risk_level
            FROM esg_companies
            WHERE UPPER(symbol) = UPPER(%s)
            LIMIT 1
        """
        return self.execute_query(query, (symbol,), fetch_one=True)
    
    def search_companies(
        self,
        query_text: str,
        sector: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT 
                symbol,
                name,
                sector,
                total_esg_risk_score
            FROM esg_companies
            WHERE (name ILIKE %s OR symbol ILIKE %s)
        """
        params = [f"%{query_text}%", f"%{query_text}%"]
        
        if sector:
            query += " AND UPPER(sector) = UPPER(%s)"
            params.append(sector)
        
        query += " ORDER BY total_esg_risk_score ASC LIMIT %s"
        params.append(limit)
        
        return self.execute_query(query, tuple(params))
    
    def close_pool(self):
        if self._connection_pool:
            self._connection_pool.closeall()
            logger.info("Database connection pool closed")


db_service = DatabaseService()
