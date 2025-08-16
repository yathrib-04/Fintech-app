"""
Demo Script for Task 1: Data Collection and Preprocessing
Fintech App Analysis Project

This script demonstrates how to use the Task 1 functionality
without actually running the full scraping process.
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

def demo_task1_functionality():
    """
    Demonstrate Task 1 functionality with sample data
    """
    print("🚀 Task 1 Demo: Data Collection and Preprocessing")
    print("=" * 60)
    
    try:
        # Import our modules
        from scraping.playstore_scraper import BankingAppScraper
        from scraping.data_preprocessor import ReviewDataPreprocessor
        from scraping.config import BANK_APPS, QUALITY_THRESHOLDS
        
        print("✅ Successfully imported all modules")
        
        # Show configuration
        print("\n📋 Configuration:")
        print(f"   Target total reviews: {QUALITY_THRESHOLDS['target_total_reviews']}")
        print(f"   Minimum reviews per bank: {QUALITY_THRESHOLDS['min_reviews_per_bank']}")
        
        print("\n🏦 Target Banking Apps:")
        for bank_code, app_info in BANK_APPS.items():
            print(f"   {bank_code}: {app_info['name']}")
            print(f"      App ID: {app_info['app_id']}")
            print(f"      Target: {app_info['target_reviews']} reviews")
        
        # Create sample data for demonstration
        print("\n📊 Creating sample data for demonstration...")
        sample_data = create_sample_data()
        
        # Demonstrate preprocessing
        print("\n🔧 Demonstrating data preprocessing...")
        preprocessor = ReviewDataPreprocessor()
        
        # Save sample data to raw directory
        raw_dir = Path('data/raw')
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        sample_file = raw_dir / 'sample_reviews.csv'
        sample_data.to_csv(sample_file, index=False)
        print(f"   Saved sample data to: {sample_file}")
        
        # Demonstrate preprocessing on sample data
        print("\n   Processing sample data...")
        processed_data, quality_report = preprocessor.run_preprocessing()
        
        if processed_data is not None:
            print(f"   ✅ Successfully processed {len(processed_data)} reviews")
            print(f"   📈 Quality report generated for {len(quality_report)} banks")
        else:
            print("   ❌ Preprocessing failed")
        
        print("\n🎯 Task 1 Demo Complete!")
        print("   The system is ready for actual data collection.")
        print("   Run 'python src/scraping/run_task1.py' to execute the full workflow.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def create_sample_data():
    """
    Create sample review data for demonstration
    
    Returns:
        pd.DataFrame: Sample review data
    """
    import random
    from datetime import datetime, timedelta
    
    # Sample review texts
    sample_reviews = [
        "Great app, very easy to use and secure",
        "The app crashes frequently when making transfers",
        "Excellent user interface and fast loading times",
        "Slow response times and difficult navigation",
        "Best banking app I've ever used, highly recommend",
        "Frequent login issues and poor customer support",
        "Clean design and intuitive features",
        "App freezes during transactions, needs improvement",
        "Reliable and secure mobile banking experience",
        "Poor performance and outdated interface"
    ]
    
    # Sample banks
    banks = ['CBE', 'BOA', 'Dashen']
    
    # Generate sample data
    data = []
    for i in range(50):  # 50 sample reviews
        bank = random.choice(banks)
        review = random.choice(sample_reviews)
        rating = random.randint(1, 5)
        
        # Generate random date within last 6 months
        days_ago = random.randint(0, 180)
        date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        data.append({
            'review': review,
            'rating': rating,
            'date': date,
            'bank': bank,
            'app_id': f'com.{bank.lower()}.mobile',
            'scraped_at': datetime.now().isoformat()
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    demo_task1_functionality()
