"""
Database Manager Module
Task 3: Store Cleaned Data in Oracle

This module handles database operations including:
- Connection management
- Table creation and schema setup
- Data insertion and management
- Query execution and data retrieval
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json

from .config import get_database_config
from .schema import DatabaseSchema

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for banking app review data"""
    
    def __init__(self, db_type: str = None):
        self.config = get_database_config()
        self.schema = DatabaseSchema()
        
        # Determine database type
        if db_type is None:
            try:
                self.db_type = self.config.get_recommended_database()
            except RuntimeError:
                logger.warning("No database available, defaulting to PostgreSQL")
                self.db_type = 'postgres'
        else:
            self.db_type = db_type.lower()
        
        # Initialize connection
        self.connection = None
        self.engine = None
        
        logger.info(f"Database manager initialized for {self.db_type}")
    
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            if self.db_type == 'oracle':
                return self._connect_oracle()
            elif self.db_type == 'postgres':
                return self._connect_postgres()
            else:
                logger.error(f"Unsupported database type: {self.db_type}")
                return False
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            return False
    
    def _connect_oracle(self) -> bool:
        """Connect to Oracle database"""
        try:
            import cx_Oracle
            from sqlalchemy import create_engine
            
            config = self.config.get_connection_params('oracle')
            
            # Create connection string
            connection_string = f"oracle+cx_oracle://{config['username']}:{config['password']}@{config['host']}:{config['port']}/?service_name={config['service_name']}"
            
            # Create SQLAlchemy engine
            self.engine = create_engine(
                connection_string,
                echo=False,
                pool_pre_ping=True
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute("SELECT 1 FROM DUAL")
            
            logger.info("✅ Oracle connection established")
            return True
            
        except ImportError:
            logger.error("cx_Oracle not available. Please install: pip install cx_Oracle")
            return False
        except Exception as e:
            logger.error(f"Oracle connection failed: {str(e)}")
            return False
    
    def _connect_postgres(self) -> bool:
        """Connect to PostgreSQL database"""
        try:
            import psycopg2
            from sqlalchemy import create_engine
            
            config = self.config.get_connection_params('postgres')
            
            # Create connection string
            connection_string = f"postgresql://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
            
            # Create SQLAlchemy engine
            self.engine = create_engine(
                connection_string,
                echo=False,
                pool_pre_ping=True
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            
            logger.info("✅ PostgreSQL connection established")
            return True
            
        except ImportError:
            logger.error("psycopg2 not available. Please install: pip install psycopg2-binary")
            return False
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {str(e)}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
    
    def create_tables(self) -> bool:
        """Create all database tables"""
        try:
            if not self.engine:
                logger.error("No database connection. Call connect() first.")
                return False
            
            # Get all CREATE TABLE statements
            sql_statements = self.schema.get_all_create_sql(self.db_type)
            
            with self.engine.connect() as conn:
                for sql in sql_statements:
                    if sql.strip():
                        logger.info(f"Executing: {sql[:100]}...")
                        conn.execute(sql)
                
                conn.commit()
            
            logger.info("✅ All tables created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Table creation failed: {str(e)}")
            return False
    
    def drop_tables(self) -> bool:
        """Drop all database tables"""
        try:
            if not self.engine:
                logger.error("No database connection. Call connect() first.")
                return False
            
            # Drop tables in reverse dependency order
            table_order = ['theme_analysis', 'sentiment_analysis', 'reviews', 'banks']
            
            with self.engine.connect() as conn:
                for table_name in table_order:
                    if table_name in self.schema.tables:
                        drop_sql = self.schema.get_drop_table_sql(table_name, self.db_type)
                        logger.info(f"Dropping table: {table_name}")
                        conn.execute(drop_sql)
                
                conn.commit()
            
            logger.info("✅ All tables dropped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Table dropping failed: {str(e)}")
            return False
    
    def insert_banks(self, banks_data: List[Dict[str, Any]]) -> bool:
        """Insert bank data into BANKS table"""
        try:
            if not self.engine:
                logger.error("No database connection. Call connect() first.")
                return False
            
            # Prepare data for insertion
            insert_data = []
            for bank in banks_data:
                insert_data.append({
                    'bank_code': bank['bank'],
                    'bank_name': bank.get('name', bank['bank']),
                    'app_id': bank['app_id'],
                    'app_name': bank.get('app_name', f"{bank['bank']} Mobile Banking"),
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                })
            
            # Convert to DataFrame and insert
            df = pd.DataFrame(insert_data)
            
            with self.engine.connect() as conn:
                df.to_sql(
                    'BANKS',
                    conn,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                conn.commit()
            
            logger.info(f"✅ Inserted {len(insert_data)} banks")
            return True
            
        except Exception as e:
            logger.error(f"Bank insertion failed: {str(e)}")
            return False
    
    def insert_reviews(self, reviews_data: pd.DataFrame) -> bool:
        """Insert review data into REVIEWS table"""
        try:
            if not self.engine:
                logger.error("No database connection. Call connect() first.")
                return False
            
            # Get bank IDs for foreign key relationships
            bank_mapping = self._get_bank_mapping()
            
            # Prepare data for insertion
            insert_data = []
            for _, row in reviews_data.iterrows():
                bank_id = bank_mapping.get(row['bank'])
                if bank_id is None:
                    logger.warning(f"Bank not found: {row['bank']}")
                    continue
                
                insert_data.append({
                    'bank_id': bank_id,
                    'review_text': row['review'],
                    'rating': row['rating'],
                    'review_date': pd.to_datetime(row['date']).date(),
                    'source': 'Google Play',
                    'scraped_at': pd.to_datetime(row.get('scraped_at', datetime.now())),
                    'created_at': datetime.now()
                })
            
            if not insert_data:
                logger.warning("No valid review data to insert")
                return False
            
            # Convert to DataFrame and insert
            df = pd.DataFrame(insert_data)
            
            with self.engine.connect() as conn:
                df.to_sql(
                    'REVIEWS',
                    conn,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                conn.commit()
            
            logger.info(f"✅ Inserted {len(insert_data)} reviews")
            return True
            
        except Exception as e:
            logger.error(f"Review insertion failed: {str(e)}")
            return False
    
    def insert_sentiment_analysis(self, sentiment_data: pd.DataFrame) -> bool:
        """Insert sentiment analysis data into SENTIMENT_ANALYSIS table"""
        try:
            if not self.engine:
                logger.error("No database connection. Call connect() first.")
                return False
            
            # Get review IDs for foreign key relationships
            review_mapping = self._get_review_mapping()
            
            # Prepare data for insertion
            insert_data = []
            for _, row in sentiment_data.iterrows():
                # Find corresponding review ID
                review_id = self._find_review_id(row, review_mapping)
                if review_id is None:
                    continue
                
                insert_data.append({
                    'review_id': review_id,
                    'sentiment_label': row.get('sentiment_label', 'neutral'),
                    'sentiment_confidence': row.get('sentiment_confidence', 0.0),
                    'sentiment_method': row.get('sentiment_method', 'consensus'),
                    'vader_compound': row.get('vader_compound'),
                    'vader_positive': row.get('vader_positive'),
                    'vader_negative': row.get('vader_negative'),
                    'vader_neutral': row.get('vader_neutral'),
                    'textblob_polarity': row.get('textblob_polarity'),
                    'textblob_subjectivity': row.get('textblob_subjectivity'),
                    'consensus_agreement': row.get('consensus_agreement'),
                    'all_sentiments': row.get('all_sentiments'),
                    'created_at': datetime.now()
                })
            
            if not insert_data:
                logger.warning("No valid sentiment data to insert")
                return False
            
            # Convert to DataFrame and insert
            df = pd.DataFrame(insert_data)
            
            with self.engine.connect() as conn:
                df.to_sql(
                    'SENTIMENT_ANALYSIS',
                    conn,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                conn.commit()
            
            logger.info(f"✅ Inserted {len(insert_data)} sentiment analysis records")
            return True
            
        except Exception as e:
            logger.error(f"Sentiment analysis insertion failed: {str(e)}")
            return False
    
    def insert_theme_analysis(self, theme_data: pd.DataFrame) -> bool:
        """Insert theme analysis data into THEME_ANALYSIS table"""
        try:
            if not self.engine:
                logger.error("No database connection. Call connect() first.")
                return False
            
            # Get review IDs for foreign key relationships
            review_mapping = self._get_review_mapping()
            
            # Prepare data for insertion
            insert_data = []
            for _, row in theme_data.iterrows():
                # Find corresponding review ID
                review_id = self._find_review_id(row, review_mapping)
                if review_id is None:
                    continue
                
                insert_data.append({
                    'review_id': review_id,
                    'primary_theme': row.get('primary_theme', 'general'),
                    'theme_scores': row.get('theme_scores'),
                    'keywords': row.get('keywords'),
                    'identified_themes': row.get('identified_themes'),
                    'created_at': datetime.now()
                })
            
            if not insert_data:
                logger.warning("No valid theme data to insert")
                return False
            
            # Convert to DataFrame and insert
            df = pd.DataFrame(insert_data)
            
            with self.engine.connect() as conn:
                df.to_sql(
                    'THEME_ANALYSIS',
                    conn,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                conn.commit()
            
            logger.info(f"✅ Inserted {len(insert_data)} theme analysis records")
            return True
            
        except Exception as e:
            logger.error(f"Theme analysis insertion failed: {str(e)}")
            return False
    
    def _get_bank_mapping(self) -> Dict[str, int]:
        """Get mapping of bank codes to bank IDs"""
        try:
            query = "SELECT bank_code, bank_id FROM BANKS"
            with self.engine.connect() as conn:
                result = conn.execute(query)
                return {row[0]: row[1] for row in result}
        except Exception as e:
            logger.error(f"Failed to get bank mapping: {str(e)}")
            return {}
    
    def _get_review_mapping(self) -> Dict[str, int]:
        """Get mapping of review text to review IDs"""
        try:
            query = "SELECT review_text, review_id FROM REVIEWS"
            with self.engine.connect() as conn:
                result = conn.execute(query)
                return {row[0]: row[1] for row in result}
        except Exception as e:
            logger.error(f"Failed to get review mapping: {str(e)}")
            return {}
    
    def _find_review_id(self, row: pd.Series, review_mapping: Dict[str, int]) -> Optional[int]:
        """Find review ID based on review text"""
        review_text = str(row.get('review', ''))
        return review_mapping.get(review_text)
    
    def get_table_counts(self) -> Dict[str, int]:
        """Get record counts for all tables"""
        try:
            if not self.engine:
                logger.error("No database connection. Call connect() first.")
                return {}
            
            counts = {}
            table_names = ['BANKS', 'REVIEWS', 'SENTIMENT_ANALYSIS', 'THEME_ANALYSIS']
            
            with self.engine.connect() as conn:
                for table_name in table_names:
                    try:
                        result = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = result.fetchone()[0]
                        counts[table_name] = count
                    except Exception as e:
                        logger.warning(f"Could not count {table_name}: {str(e)}")
                        counts[table_name] = 0
            
            return counts
            
        except Exception as e:
            logger.error(f"Failed to get table counts: {str(e)}")
            return {}
    
    def export_schema_sql(self, output_file: str = None) -> str:
        """Export complete schema SQL to file"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"database_schema_{self.db_type}_{timestamp}.sql"
        
        try:
            sql_statements = self.schema.get_all_create_sql(self.db_type)
            
            with open(output_file, 'w') as f:
                f.write(f"-- Database Schema for {self.db_type.upper()}\n")
                f.write(f"-- Generated on: {datetime.now()}\n")
                f.write(f"-- Tables: {', '.join(self.schema.tables.keys())}\n\n")
                
                for sql in sql_statements:
                    f.write(sql + "\n\n")
            
            logger.info(f"✅ Schema exported to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Schema export failed: {str(e)}")
            return ""

def main():
    """Test the database manager"""
    print("🗄️  Database Manager Test")
    print("=" * 40)
    
    # Test configuration
    config = get_database_config()
    print(f"Available databases: {config.get_available_databases()}")
    
    # Test schema
    schema = DatabaseSchema()
    print(f"Schema tables: {list(schema.tables.keys())}")
    
    # Test manager initialization
    try:
        manager = DatabaseManager()
        print(f"Manager initialized for: {manager.db_type}")
        
        # Test connection
        if manager.connect():
            print("✅ Connection successful")
            
            # Test table creation
            if manager.create_tables():
                print("✅ Tables created")
                
                # Get table counts
                counts = manager.get_table_counts()
                print(f"Table counts: {counts}")
                
                # Export schema
                schema_file = manager.export_schema_sql()
                print(f"Schema exported to: {schema_file}")
            
            manager.disconnect()
        else:
            print("❌ Connection failed")
    
    except Exception as e:
        print(f"❌ Manager test failed: {str(e)}")

if __name__ == "__main__":
    main()
