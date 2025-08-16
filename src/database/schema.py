"""
Database Schema Module
Task 3: Store Cleaned Data in Oracle

This module defines the database schema for storing banking app review data.
It includes table creation scripts for both Oracle and PostgreSQL.
"""

import logging
from typing import Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseSchema:
    """Database schema definitions for banking app reviews"""
    
    def __init__(self):
        self.schema_name = 'BANK_REVIEWS'
        self.tables = {
            'banks': self._get_banks_table_schema(),
            'reviews': self._get_reviews_table_schema(),
            'sentiment_analysis': self._get_sentiment_analysis_table_schema(),
            'theme_analysis': self._get_theme_analysis_table_schema()
        }
    
    def _get_banks_table_schema(self) -> Dict[str, Any]:
        """Get banks table schema"""
        return {
            'table_name': 'BANKS',
            'columns': {
                'bank_id': {
                    'oracle': 'NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY',
                    'postgres': 'SERIAL PRIMARY KEY',
                    'description': 'Unique identifier for each bank'
                },
                'bank_code': {
                    'oracle': 'VARCHAR2(10) NOT NULL UNIQUE',
                    'postgres': 'VARCHAR(10) NOT NULL UNIQUE',
                    'description': 'Bank code (e.g., CBE, BOA, Dashen)'
                },
                'bank_name': {
                    'oracle': 'VARCHAR2(100) NOT NULL',
                    'postgres': 'VARCHAR(100) NOT NULL',
                    'description': 'Full bank name'
                },
                'app_id': {
                    'oracle': 'VARCHAR2(100) NOT NULL',
                    'postgres': 'VARCHAR(100) NOT NULL',
                    'description': 'Google Play Store app ID'
                },
                'app_name': {
                    'oracle': 'VARCHAR2(200) NOT NULL',
                    'postgres': 'VARCHAR(200) NOT NULL',
                    'description': 'Mobile app name'
                },
                'created_at': {
                    'oracle': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                    'postgres': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                    'description': 'Record creation timestamp'
                },
                'updated_at': {
                    'oracle': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                    'postgres': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                    'description': 'Record update timestamp'
                }
            },
            'constraints': [
                'CONSTRAINT pk_banks PRIMARY KEY (bank_id)',
                'CONSTRAINT uk_banks_code UNIQUE (bank_code)',
                'CONSTRAINT uk_banks_app_id UNIQUE (app_id)'
            ]
        }
    
    def _get_reviews_table_schema(self) -> Dict[str, Any]:
        """Get reviews table schema"""
        return {
            'table_name': 'REVIEWS',
            'columns': {
                'review_id': {
                    'oracle': 'NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY',
                    'postgres': 'SERIAL PRIMARY KEY',
                    'description': 'Unique identifier for each review'
                },
                'bank_id': {
                    'oracle': 'NUMBER NOT NULL',
                    'postgres': 'INTEGER NOT NULL',
                    'description': 'Foreign key to banks table'
                },
                'review_text': {
                    'oracle': 'CLOB NOT NULL',
                    'postgres': 'TEXT NOT NULL',
                    'description': 'Review text content'
                },
                'rating': {
                    'oracle': 'NUMBER(1) NOT NULL CHECK (rating >= 1 AND rating <= 5)',
                    'postgres': 'INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5)',
                    'description': 'User rating (1-5 stars)'
                },
                'review_date': {
                    'oracle': 'DATE NOT NULL',
                    'postgres': 'DATE NOT NULL',
                    'description': 'Date when review was posted'
                },
                'source': {
                    'oracle': 'VARCHAR2(50) DEFAULT \'Google Play\'',
                    'postgres': 'VARCHAR(50) DEFAULT \'Google Play\'',
                    'description': 'Source of the review'
                },
                'scraped_at': {
                    'oracle': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                    'postgres': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                    'description': 'When the review was scraped'
                },
                'created_at': {
                    'oracle': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                    'postgres': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                    'description': 'Record creation timestamp'
                }
            },
            'constraints': [
                'CONSTRAINT pk_reviews PRIMARY KEY (review_id)',
                'CONSTRAINT fk_reviews_bank FOREIGN KEY (bank_id) REFERENCES BANKS(bank_id)',
                'CONSTRAINT ck_reviews_rating CHECK (rating >= 1 AND rating <= 5)'
            ]
        }
    
    def _get_sentiment_analysis_table_schema(self) -> Dict[str, Any]:
        """Get sentiment analysis table schema"""
        return {
            'table_name': 'SENTIMENT_ANALYSIS',
            'columns': {
                'sentiment_id': {
                    'oracle': 'NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY',
                    'postgres': 'SERIAL PRIMARY KEY',
                    'description': 'Unique identifier for sentiment analysis'
                },
                'review_id': {
                    'oracle': 'NUMBER NOT NULL',
                    'postgres': 'INTEGER NOT NULL',
                    'description': 'Foreign key to reviews table'
                },
                'sentiment_label': {
                    'oracle': 'VARCHAR2(20) NOT NULL',
                    'postgres': 'VARCHAR(20) NOT NULL',
                    'description': 'Sentiment classification (positive/negative/neutral)'
                },
                'sentiment_confidence': {
                    'oracle': 'NUMBER(5,4) NOT NULL',
                    'postgres': 'DECIMAL(5,4) NOT NULL',
                    'description': 'Confidence score (0.0000 to 1.0000)'
                },
                'sentiment_method': {
                    'oracle': 'VARCHAR2(50) NOT NULL',
                    'postgres': 'VARCHAR(50) NOT NULL',
                    'description': 'Method used for sentiment analysis'
                },
                'vader_compound': {
                    'oracle': 'NUMBER(5,4)',
                    'postgres': 'DECIMAL(5,4)',
                    'description': 'VADER compound sentiment score'
                },
                'vader_positive': {
                    'oracle': 'NUMBER(5,4)',
                    'postgres': 'DECIMAL(5,4)',
                    'description': 'VADER positive sentiment score'
                },
                'vader_negative': {
                    'oracle': 'NUMBER(5,4)',
                    'postgres': 'DECIMAL(5,4)',
                    'description': 'VADER negative sentiment score'
                },
                'vader_neutral': {
                    'oracle': 'NUMBER(5,4)',
                    'postgres': 'DECIMAL(5,4)',
                    'description': 'VADER neutral sentiment score'
                },
                'textblob_polarity': {
                    'oracle': 'NUMBER(5,4)',
                    'postgres': 'DECIMAL(5,4)',
                    'description': 'TextBlob polarity score'
                },
                'textblob_subjectivity': {
                    'oracle': 'NUMBER(5,4)',
                    'postgres': 'DECIMAL(5,4)',
                    'description': 'TextBlob subjectivity score'
                },
                'consensus_agreement': {
                    'oracle': 'NUMBER(5,4)',
                    'postgres': 'DECIMAL(5,4)',
                    'description': 'Agreement level between different methods'
                },
                'all_sentiments': {
                    'oracle': 'CLOB',
                    'postgres': 'TEXT',
                    'description': 'JSON string of all sentiment method results'
                },
                'created_at': {
                    'oracle': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                    'postgres': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                    'description': 'Record creation timestamp'
                }
            },
            'constraints': [
                'CONSTRAINT pk_sentiment_analysis PRIMARY KEY (sentiment_id)',
                'CONSTRAINT fk_sentiment_review FOREIGN KEY (review_id) REFERENCES REVIEWS(review_id)',
                'CONSTRAINT ck_sentiment_confidence CHECK (sentiment_confidence >= 0 AND sentiment_confidence <= 1)',
                'CONSTRAINT ck_sentiment_label CHECK (sentiment_label IN (\'positive\', \'negative\', \'neutral\'))'
            ]
        }
    
    def _get_theme_analysis_table_schema(self) -> Dict[str, Any]:
        """Get theme analysis table schema"""
        return {
            'table_name': 'THEME_ANALYSIS',
            'columns': {
                'theme_id': {
                    'oracle': 'NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY',
                    'postgres': 'SERIAL PRIMARY KEY',
                    'description': 'Unique identifier for theme analysis'
                },
                'review_id': {
                    'oracle': 'NUMBER NOT NULL',
                    'postgres': 'INTEGER NOT NULL',
                    'description': 'Foreign key to reviews table'
                },
                'primary_theme': {
                    'oracle': 'VARCHAR2(50) NOT NULL',
                    'postgres': 'VARCHAR(50) NOT NULL',
                    'description': 'Primary identified theme'
                },
                'theme_scores': {
                    'oracle': 'CLOB',
                    'postgres': 'TEXT',
                    'description': 'JSON string of theme scores'
                },
                'keywords': {
                    'oracle': 'CLOB',
                    'postgres': 'TEXT',
                    'description': 'JSON string of extracted keywords'
                },
                'identified_themes': {
                    'oracle': 'CLOB',
                    'postgres': 'TEXT',
                    'description': 'JSON string of all identified themes'
                },
                'created_at': {
                    'oracle': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                    'postgres': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                    'description': 'Record creation timestamp'
                }
            },
            'constraints': [
                'CONSTRAINT pk_theme_analysis PRIMARY KEY (theme_id)',
                'CONSTRAINT fk_theme_review FOREIGN KEY (review_id) REFERENCES REVIEWS(review_id)'
            ]
        }
    
    def get_create_table_sql(self, table_name: str, db_type: str = 'oracle') -> str:
        """Generate CREATE TABLE SQL for specified table and database type"""
        if table_name not in self.tables:
            raise ValueError(f"Unknown table: {table_name}")
        
        table_schema = self.tables[table_name]
        columns = []
        
        # Build column definitions
        for col_name, col_def in table_schema['columns'].items():
            if db_type.lower() == 'oracle':
                col_type = col_def['oracle']
            elif db_type.lower() == 'postgres':
                col_type = col_def['postgres']
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            columns.append(f"    {col_name} {col_type}")
        
        # Build constraints
        constraints = []
        for constraint in table_schema['constraints']:
            constraints.append(f"    {constraint}")
        
        # Generate SQL
        sql_parts = [
            f"CREATE TABLE {table_schema['table_name']} (",
            ",\n".join(columns)
        ]
        
        if constraints:
            sql_parts.append(",\n".join(constraints))
        
        sql_parts.append(");")
        
        return "\n".join(sql_parts)
    
    def get_create_schema_sql(self, db_type: str = 'oracle') -> str:
        """Generate CREATE SCHEMA SQL for Oracle"""
        if db_type.lower() != 'oracle':
            return ""
        
        return f"""
-- Create schema for Oracle
CREATE USER {self.schema_name} IDENTIFIED BY password123;
GRANT CONNECT, RESOURCE TO {self.schema_name};
GRANT CREATE SESSION TO {self.schema_name};
GRANT UNLIMITED TABLESPACE TO {self.schema_name};
"""
    
    def get_drop_table_sql(self, table_name: str, db_type: str = 'oracle') -> str:
        """Generate DROP TABLE SQL"""
        if table_name not in self.tables:
            raise ValueError(f"Unknown table: {table_name}")
        
        table_schema = self.tables[table_name]
        return f"DROP TABLE {table_schema['table_name']} CASCADE;"
    
    def get_all_create_sql(self, db_type: str = 'oracle') -> List[str]:
        """Get all CREATE TABLE SQL statements in dependency order"""
        # Order tables by dependencies
        table_order = ['banks', 'reviews', 'sentiment_analysis', 'theme_analysis']
        
        sql_statements = []
        
        # Add schema creation for Oracle
        if db_type.lower() == 'oracle':
            schema_sql = self.get_create_schema_sql(db_type)
            if schema_sql.strip():
                sql_statements.append(schema_sql)
        
        # Add table creation in dependency order
        for table_name in table_order:
            if table_name in self.tables:
                sql_statements.append(self.get_create_table_sql(table_name, db_type))
        
        return sql_statements
    
    def get_table_info(self) -> Dict[str, Any]:
        """Get information about all tables"""
        info = {}
        for table_name, table_schema in self.tables.items():
            info[table_name] = {
                'table_name': table_schema['table_name'],
                'column_count': len(table_schema['columns']),
                'columns': list(table_schema['columns'].keys()),
                'constraints': len(table_schema['constraints'])
            }
        return info

def main():
    """Test the schema module"""
    schema = DatabaseSchema()
    
    print("🗄️  Database Schema Information")
    print("=" * 50)
    
    # Show table information
    table_info = schema.get_table_info()
    for table_name, info in table_info.items():
        print(f"\n📋 {table_name.upper()}:")
        print(f"   Table: {info['table_name']}")
        print(f"   Columns: {info['column_count']}")
        print(f"   Constraints: {info['constraints']}")
    
    # Show sample SQL for Oracle
    print(f"\n🔧 Sample Oracle CREATE TABLE SQL:")
    print("=" * 50)
    print(schema.get_create_table_sql('banks', 'oracle'))
    
    # Show sample SQL for PostgreSQL
    print(f"\n🔧 Sample PostgreSQL CREATE TABLE SQL:")
    print("=" * 50)
    print(schema.get_create_table_sql('banks', 'postgres'))

if __name__ == "__main__":
    main()
