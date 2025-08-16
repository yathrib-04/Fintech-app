"""
Task 3 Runner: Store Cleaned Data in Oracle
Fintech App Analysis Project

This script orchestrates the complete Task 3 workflow:
1. Load analyzed data from Task 2
2. Set up database connection (Oracle or PostgreSQL fallback)
3. Create database schema and tables
4. Insert all data (banks, reviews, sentiment, themes)
5. Generate database dump and summary
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from database.manager import DatabaseManager
from database.schema import DatabaseSchema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('task3_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_task2_data():
    """
    Load analyzed data from Task 2
    
    Returns:
        tuple: (banks_data, reviews_data, sentiment_data, theme_data)
    """
    logger.info("Loading Task 2 analyzed data...")
    
    # Look for Task 2 complete analysis file
    processed_dir = Path('../data/processed')
    
    # Find the most recent Task 2 complete analysis file
    task2_files = list(processed_dir.glob('task2_complete_analysis_*.csv'))
    
    if not task2_files:
        raise FileNotFoundError("No Task 2 complete analysis data found. Please run Task 2 first.")
    
    # Load the most recent file
    latest_file = max(task2_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading data from: {latest_file}")
    
    df = pd.read_csv(latest_file)
    logger.info(f"Loaded {len(df)} analyzed reviews from Task 2")
    
    # Prepare banks data
    banks_data = []
    for bank in df['bank'].unique():
        bank_rows = df[df['bank'] == bank]
        banks_data.append({
            'bank': bank,
            'app_id': bank_rows['app_id'].iloc[0],
            'name': f"{bank} Bank",
            'app_name': f"{bank} Mobile Banking"
        })
    
    # Prepare reviews data (basic review information)
    reviews_data = df[['review', 'rating', 'date', 'bank', 'app_id', 'scraped_at']].copy()
    
    # Prepare sentiment analysis data
    sentiment_columns = [
        'review', 'sentiment_label', 'sentiment_confidence', 'sentiment_method',
        'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral',
        'textblob_polarity', 'textblob_subjectivity', 'consensus_agreement', 'all_sentiments'
    ]
    sentiment_data = df[sentiment_columns].copy()
    
    # Prepare theme analysis data
    theme_columns = ['review', 'primary_theme', 'theme_scores', 'keywords', 'identified_themes']
    theme_data = df[theme_columns].copy()
    
    logger.info(f"Prepared data: {len(banks_data)} banks, {len(reviews_data)} reviews")
    
    return banks_data, reviews_data, sentiment_data, theme_data

def setup_database():
    """
    Set up database connection and create schema
    
    Returns:
        DatabaseManager: Configured database manager
    """
    logger.info("="*50)
    logger.info("STEP 1: DATABASE SETUP")
    logger.info("="*50)
    
    # Initialize database manager
    manager = DatabaseManager()
    logger.info(f"Database manager initialized for: {manager.db_type}")
    
    # Test connection
    if not manager.connect():
        raise RuntimeError(f"Failed to connect to {manager.db_type} database")
    
    # Create tables
    logger.info("Creating database tables...")
    if not manager.create_tables():
        raise RuntimeError("Failed to create database tables")
    
    logger.info("✅ Database setup completed successfully")
    return manager

def insert_data(manager: DatabaseManager, banks_data, reviews_data, sentiment_data, theme_data):
    """
    Insert all data into the database
    
    Args:
        manager: Database manager instance
        banks_data: Bank information
        reviews_data: Review data
        sentiment_data: Sentiment analysis data
        theme_data: Theme analysis data
    """
    logger.info("="*50)
    logger.info("STEP 2: DATA INSERTION")
    logger.info("="*50)
    
    # Insert banks first (required for foreign key relationships)
    logger.info("Inserting bank data...")
    if not manager.insert_banks(banks_data):
        raise RuntimeError("Failed to insert bank data")
    
    # Insert reviews
    logger.info("Inserting review data...")
    if not manager.insert_reviews(reviews_data):
        raise RuntimeError("Failed to insert review data")
    
    # Insert sentiment analysis
    logger.info("Inserting sentiment analysis data...")
    if not manager.insert_sentiment_analysis(sentiment_data):
        raise RuntimeError("Failed to insert sentiment analysis data")
    
    # Insert theme analysis
    logger.info("Inserting theme analysis data...")
    if not manager.insert_theme_analysis(theme_data):
        raise RuntimeError("Failed to insert theme analysis data")
    
    logger.info("✅ All data inserted successfully")

def verify_data(manager: DatabaseManager):
    """
    Verify that data was inserted correctly
    
    Args:
        manager: Database manager instance
    """
    logger.info("="*50)
    logger.info("STEP 3: DATA VERIFICATION")
    logger.info("="*50)
    
    # Get table counts
    counts = manager.get_table_counts()
    
    logger.info("Table record counts:")
    for table_name, count in counts.items():
        logger.info(f"  {table_name}: {count} records")
    
    # Verify data integrity
    total_reviews = counts.get('REVIEWS', 0)
    total_sentiment = counts.get('SENTIMENT_ANALYSIS', 0)
    total_themes = counts.get('THEME_ANALYSIS', 0)
    
    if total_reviews > 0:
        logger.info(f"✅ Reviews table: {total_reviews} records")
        
        if total_sentiment > 0:
            logger.info(f"✅ Sentiment analysis: {total_sentiment} records")
        else:
            logger.warning("⚠️  No sentiment analysis records found")
        
        if total_themes > 0:
            logger.info(f"✅ Theme analysis: {total_themes} records")
        else:
            logger.warning("⚠️  No theme analysis records found")
    
    return counts

def export_database_info(manager: DatabaseManager, counts: dict):
    """
    Export database information and schema
    
    Args:
        manager: Database manager instance
        counts: Table record counts
    """
    logger.info("="*50)
    logger.info("STEP 4: EXPORT DATABASE INFO")
    logger.info("="*50)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('../data/processed')
    
    # Export schema SQL
    schema_file = manager.export_schema_sql()
    logger.info(f"Schema exported to: {schema_file}")
    
    # Create database summary
    summary = {
        'task': 'Task 3: Store Cleaned Data in Oracle',
        'timestamp': timestamp,
        'database_type': manager.db_type,
        'table_counts': counts,
        'schema_file': schema_file,
        'total_records': sum(counts.values()),
        'status': 'completed'
    }
    
    # Save summary
    summary_file = output_dir / f"task3_database_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Database summary saved to: {summary_file}")
    
    return schema_file, summary_file

def run_task3():
    """
    Execute the complete Task 3 workflow
    """
    logger.info("="*60)
    logger.info("STARTING TASK 3: STORE CLEANED DATA IN ORACLE")
    logger.info("="*60)
    
    start_time = datetime.now()
    manager = None
    
    try:
        # Step 1: Load Task 2 data
        logger.info("Loading data from Task 2...")
        banks_data, reviews_data, sentiment_data, theme_data = load_task2_data()
        
        # Step 2: Set up database
        manager = setup_database()
        
        # Step 3: Insert data
        insert_data(manager, banks_data, reviews_data, sentiment_data, theme_data)
        
        # Step 4: Verify data
        counts = verify_data(manager)
        
        # Step 5: Export database info
        schema_file, summary_file = export_database_info(manager, counts)
        
        # Task completion
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info("TASK 3 COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Duration: {duration}")
        logger.info(f"Database type: {manager.db_type}")
        logger.info(f"Total records inserted: {sum(counts.values())}")
        logger.info(f"Tables created: {list(counts.keys())}")
        logger.info(f"Schema exported to: {schema_file}")
        logger.info(f"Summary saved to: {summary_file}")
        logger.info(f"Ready for Task 4: Insights and Recommendations")
        
        return True
        
    except Exception as e:
        logger.error(f"Task 3 failed with error: {str(e)}")
        return False
        
    finally:
        # Always disconnect from database
        if manager:
            manager.disconnect()

def main():
    """Main execution function"""
    print("🚀 Starting Task 3: Store Cleaned Data in Oracle")
    print("=" * 60)
    
    success = run_task3()
    
    if success:
        print("\n🎉 Task 3 completed successfully!")
        print("🗄️  Database schema created and tables populated")
        print("📊 All review data stored with sentiment and theme analysis")
        print("🔗 Foreign key relationships established")
        print("📁 Schema SQL and database summary exported")
        print("➡️  Ready to proceed with Task 4: Insights and Recommendations")
    else:
        print("\n❌ Task 3 failed. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
