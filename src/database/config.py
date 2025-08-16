"""
Database Configuration Module
Task 3: Store Cleaned Data in Oracle

This module handles database configuration for both Oracle and PostgreSQL
as fallback options for storing the banking app review data.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration for Oracle and PostgreSQL"""
    
    def __init__(self):
        self.config_dir = Path(__file__).parent
        self.env_file = self.config_dir.parent.parent.parent / '.env'
        
        # Load environment variables if .env file exists
        self._load_env_vars()
        
        # Database configurations
        self.oracle_config = self._get_oracle_config()
        self.postgres_config = self._get_postgres_config()
        
        # Default to Oracle, fallback to PostgreSQL
        self.primary_db = 'oracle'
        self.fallback_db = 'postgres'
    
    def _load_env_vars(self):
        """Load environment variables from .env file"""
        if self.env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(self.env_file)
                logger.info("Environment variables loaded from .env file")
            except ImportError:
                logger.warning("python-dotenv not available, using system environment variables")
        else:
            logger.info("No .env file found, using system environment variables")
    
    def _get_oracle_config(self) -> Dict[str, Any]:
        """Get Oracle database configuration"""
        return {
            'host': os.getenv('ORACLE_HOST', 'localhost'),
            'port': int(os.getenv('ORACLE_PORT', '1521')),
            'service_name': os.getenv('ORACLE_SERVICE', 'XE'),
            'username': os.getenv('ORACLE_USER', 'system'),
            'password': os.getenv('ORACLE_PASSWORD', 'oracle'),
            'dsn': os.getenv('ORACLE_DSN', 'localhost:1521/XE'),
            'encoding': 'UTF-8',
            'nencoding': 'UTF-8',
            'autocommit': True
        }
    
    def _get_postgres_config(self) -> Dict[str, Any]:
        """Get PostgreSQL database configuration"""
        return {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'bank_reviews'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
            'encoding': 'utf8'
        }
    
    def get_database_url(self, db_type: str = 'oracle') -> str:
        """Get database connection URL"""
        if db_type.lower() == 'oracle':
            config = self.oracle_config
            return f"oracle+cx_oracle://{config['username']}:{config['password']}@{config['host']}:{config['port']}/?service_name={config['service_name']}"
        elif db_type.lower() == 'postgres':
            config = self.postgres_config
            return f"postgresql://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def get_connection_params(self, db_type: str = 'oracle') -> Dict[str, Any]:
        """Get database connection parameters"""
        if db_type.lower() == 'oracle':
            return self.oracle_config
        elif db_type.lower() == 'postgres':
            return self.postgres_config
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def test_connection(self, db_type: str = 'oracle') -> bool:
        """Test database connection"""
        try:
            if db_type.lower() == 'oracle':
                return self._test_oracle_connection()
            elif db_type.lower() == 'postgres':
                return self._test_postgres_connection()
            else:
                logger.error(f"Unsupported database type: {db_type}")
                return False
        except Exception as e:
            logger.error(f"Connection test failed for {db_type}: {str(e)}")
            return False
    
    def _test_oracle_connection(self) -> bool:
        """Test Oracle connection"""
        try:
            import cx_Oracle
            config = self.oracle_config
            
            # Test connection
            connection = cx_Oracle.connect(
                user=config['username'],
                password=config['password'],
                dsn=config['dsn'],
                encoding=config['encoding'],
                nencoding=config['nencoding']
            )
            
            # Test query
            cursor = connection.cursor()
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()
            
            cursor.close()
            connection.close()
            
            logger.info("✅ Oracle connection test successful")
            return True
            
        except ImportError:
            logger.warning("cx_Oracle not available for Oracle connection test")
            return False
        except Exception as e:
            logger.error(f"Oracle connection test failed: {str(e)}")
            return False
    
    def _test_postgres_connection(self) -> bool:
        """Test PostgreSQL connection"""
        try:
            import psycopg2
            config = self.postgres_config
            
            # Test connection
            connection = psycopg2.connect(
                host=config['host'],
                port=config['port'],
                database=config['database'],
                user=config['username'],
                password=config['password']
            )
            
            # Test query
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            cursor.close()
            connection.close()
            
            logger.info("✅ PostgreSQL connection test successful")
            return True
            
        except ImportError:
            logger.warning("psycopg2 not available for PostgreSQL connection test")
            return False
        except Exception as e:
            logger.error(f"PostgreSQL connection test failed: {str(e)}")
            return False
    
    def get_available_databases(self) -> Dict[str, bool]:
        """Get list of available databases and their connection status"""
        return {
            'oracle': self.test_connection('oracle'),
            'postgres': self.test_connection('postgres')
        }
    
    def get_recommended_database(self) -> str:
        """Get the recommended database to use"""
        available = self.get_available_databases()
        
        if available['oracle']:
            logger.info("Oracle is available and recommended")
            return 'oracle'
        elif available['postgres']:
            logger.info("Oracle not available, using PostgreSQL as fallback")
            return 'postgres'
        else:
            raise RuntimeError("No database connections available. Please check your configuration.")

# Global database configuration instance
db_config = DatabaseConfig()

def get_database_config() -> DatabaseConfig:
    """Get the global database configuration instance"""
    return db_config

if __name__ == "__main__":
    # Test configuration
    config = DatabaseConfig()
    
    print("🔧 Database Configuration Test")
    print("=" * 40)
    
    print(f"Oracle config: {config.oracle_config}")
    print(f"PostgreSQL config: {config.postgres_config}")
    
    print("\n🔍 Testing connections...")
    available = config.get_available_databases()
    
    for db_type, is_available in available.items():
        status = "✅ Available" if is_available else "❌ Not available"
        print(f"{db_type.capitalize()}: {status}")
    
    try:
        recommended = config.get_recommended_database()
        print(f"\n🎯 Recommended database: {recommended}")
    except RuntimeError as e:
        print(f"\n❌ Error: {e}")
