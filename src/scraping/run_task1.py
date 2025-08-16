"""
Task 1 Runner: Data Collection and Preprocessing
Fintech App Analysis Project

This script orchestrates the complete Task 1 workflow:
1. Scrape reviews from Google Play Store
2. Preprocess and clean the data
3. Generate quality reports
4. Save processed data for next tasks
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scraping.playstore_scraper import BankingAppScraper
from scraping.data_preprocessor import ReviewDataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('task1_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_task1():
    """
    Execute the complete Task 1 workflow
    """
    logger.info("="*60)
    logger.info("STARTING TASK 1: DATA COLLECTION AND PREPROCESSING")
    logger.info("="*60)
    
    start_time = datetime.now()
    
    try:
        # Step 1: Data Collection (Scraping)
        logger.info("\n" + "="*50)
        logger.info("STEP 1: SCRAPING GOOGLE PLAY STORE REVIEWS")
        logger.info("="*50)
        
        scraper = BankingAppScraper()
        scraping_results = scraper.run_scraping()
        
        if scraping_results[0] is None:
            logger.error("Scraping failed. Cannot proceed with preprocessing.")
            return False
        
        combined_data, scraping_summary = scraping_results
        logger.info(f"✅ Scraping completed successfully!")
        logger.info(f"   Total reviews collected: {len(combined_data)}")
        
        # Step 2: Data Preprocessing
        logger.info("\n" + "="*50)
        logger.info("STEP 2: DATA PREPROCESSING AND CLEANING")
        logger.info("="*50)
        
        preprocessor = ReviewDataPreprocessor()
        preprocessing_results = preprocessor.run_preprocessing()
        
        if preprocessing_results[0] is None:
            logger.error("Preprocessing failed.")
            return False
        
        processed_data, quality_report = preprocessing_results
        logger.info(f"✅ Preprocessing completed successfully!")
        logger.info(f"   Total processed reviews: {len(processed_data)}")
        
        # Step 3: Quality Assessment
        logger.info("\n" + "="*50)
        logger.info("STEP 3: QUALITY ASSESSMENT")
        logger.info("="*50)
        
        total_reviews = len(processed_data)
        target_reviews = 1200
        quality_score = (total_reviews / target_reviews) * 100
        
        logger.info(f"Quality Metrics:")
        logger.info(f"   Target reviews: {target_reviews}")
        logger.info(f"   Actual reviews: {total_reviews}")
        logger.info(f"   Quality score: {quality_score:.1f}%")
        
        # Check if we meet minimum requirements
        if total_reviews >= 1000:  # Allow some flexibility
            logger.info("✅ Quality requirements met!")
        else:
            logger.warning("⚠️  Quality requirements not fully met, but proceeding...")
        
        # Step 4: Generate Summary Report
        logger.info("\n" + "="*50)
        logger.info("STEP 4: GENERATING SUMMARY REPORT")
        logger.info("="*50)
        
        summary_report = generate_summary_report(scraping_summary, quality_report, processed_data)
        save_summary_report(summary_report)
        
        # Step 5: Task Completion
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info("TASK 1 COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Duration: {duration}")
        logger.info(f"Total reviews processed: {len(processed_data)}")
        logger.info(f"Data saved to: data/processed/")
        logger.info(f"Ready for Task 2: Sentiment and Thematic Analysis")
        
        return True
        
    except Exception as e:
        logger.error(f"Task 1 failed with error: {str(e)}")
        return False

def generate_summary_report(scraping_summary, quality_report, processed_data):
    """
    Generate a comprehensive summary report for Task 1
    
    Args:
        scraping_summary (dict): Summary of scraping results
        quality_report (dict): Quality metrics for each bank
        processed_data (pd.DataFrame): Final processed data
        
    Returns:
        dict: Summary report
    """
    report = {
        'task': 'Task 1: Data Collection and Preprocessing',
        'timestamp': datetime.now().isoformat(),
        'overview': {
            'total_reviews_collected': len(processed_data),
            'target_reviews': 1200,
            'quality_score': f"{(len(processed_data) / 1200) * 100:.1f}%"
        },
        'scraping_results': scraping_summary,
        'quality_metrics': quality_report,
        'data_structure': {
            'columns': list(processed_data.columns),
            'banks_covered': list(processed_data['bank'].unique()),
            'date_range': f"{processed_data['date'].min()} to {processed_data['date'].max()}",
            'rating_distribution': processed_data['rating'].value_counts().sort_index().to_dict()
        },
        'next_steps': [
            'Task 2: Sentiment and Thematic Analysis',
            'Task 3: Database Storage in Oracle',
            'Task 4: Insights and Recommendations'
        ]
    }
    
    return report

def save_summary_report(report):
    """
    Save the summary report to file
    
    Args:
        report (dict): Summary report to save
    """
    try:
        # Create reports directory
        reports_dir = Path('../reports')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f"task1_summary_{timestamp}.json"
        
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Summary report saved to: {report_file}")
        
        # Also save a human-readable version
        txt_file = reports_dir / f"task1_summary_{timestamp}.txt"
        with open(txt_file, 'w') as f:
            f.write("TASK 1 SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Task: {report['task']}\n")
            f.write(f"Completed: {report['timestamp']}\n\n")
            f.write(f"Total Reviews: {report['overview']['total_reviews_collected']}\n")
            f.write(f"Quality Score: {report['overview']['quality_score']}\n\n")
            
            f.write("BANK-SPECIFIC RESULTS:\n")
            f.write("-" * 30 + "\n")
            for bank, metrics in report['quality_metrics'].items():
                f.write(f"{bank}: {metrics['total_reviews']} reviews\n")
            
            f.write("\nNEXT STEPS:\n")
            f.write("-" * 20 + "\n")
            for step in report['next_steps']:
                f.write(f"• {step}\n")
        
        logger.info(f"Human-readable report saved to: {txt_file}")
        
    except Exception as e:
        logger.error(f"Failed to save summary report: {str(e)}")

def main():
    """
    Main execution function
    """
    print("🚀 Starting Task 1: Data Collection and Preprocessing")
    print("=" * 60)
    
    success = run_task1()
    
    if success:
        print("\n🎉 Task 1 completed successfully!")
        print("📊 Check the 'data/processed' directory for cleaned data")
        print("📋 Check the 'reports' directory for summary reports")
        print("➡️  Ready to proceed with Task 2: Sentiment Analysis")
    else:
        print("\n❌ Task 1 failed. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
