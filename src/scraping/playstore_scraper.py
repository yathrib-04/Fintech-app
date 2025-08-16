"""
Google Play Store Review Scraper for Banking Apps
Task 1: Data Collection and Preprocessing

This script scrapes reviews from three banking apps:
- Commercial Bank of Ethiopia (CBE)
- Bank of America (BOA) 
- Dashen Bank

Target: 400+ reviews per bank (1,200 total)
"""

import pandas as pd
import time
from datetime import datetime
from google_play_scraper import app, reviews_all
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraping.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BankingAppScraper:
    """Scraper for banking app reviews from Google Play Store"""
    
    def __init__(self):
        self.bank_apps = {
            'CBE': {
                'name': 'Commercial Bank of Ethiopia Mobile',
                'app_id': 'com.cbe.mobile',
                'target_reviews': 400
            },
            'BOA': {
                'name': 'Bank of America Mobile Banking',
                'app_id': 'com.infonow.bofa',
                'target_reviews': 400
            },
            'Dashen': {
                'name': 'Dashen Bank Mobile Banking',
                'app_id': 'com.dashenbank.mobile',
                'target_reviews': 400
            }
        }
        
        # Create data directories
        self.data_dir = Path('../data')
        self.raw_dir = self.data_dir / 'raw'
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
    def scrape_app_reviews(self, app_id, app_name, target_count=400):
        """
        Scrape reviews for a specific app
        
        Args:
            app_id (str): Google Play Store app ID
            app_name (str): Human-readable app name
            target_count (int): Target number of reviews to collect
            
        Returns:
            list: List of review dictionaries
        """
        logger.info(f"Starting to scrape reviews for {app_name} (ID: {app_id})")
        
        try:
            # Get app information first
            app_info = app(app_id)
            logger.info(f"App: {app_info['title']}")
            logger.info(f"Current Rating: {app_info['score']} stars")
            logger.info(f"Total Reviews: {app_info['reviews']}")
            
            # Scrape reviews
            reviews = reviews_all(
                app_id,
                lang='en',  # English reviews
                country='us',  # US region
                sort='newest',  # Sort by newest first
                count=target_count
            )
            
            logger.info(f"Successfully scraped {len(reviews)} reviews for {app_name}")
            
            # Add metadata to each review
            for review in reviews:
                review['app_name'] = app_name
                review['app_id'] = app_id
                review['scraped_at'] = datetime.now().isoformat()
                
            return reviews
            
        except Exception as e:
            logger.error(f"Error scraping {app_name}: {str(e)}")
            return []
    
    def preprocess_reviews(self, reviews):
        """
        Preprocess scraped reviews
        
        Args:
            reviews (list): Raw review data
            
        Returns:
            pd.DataFrame: Cleaned review data
        """
        if not reviews:
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(reviews)
        
        # Handle missing data
        df = df.dropna(subset=['content', 'score'])
        
        # Normalize dates
        df['at'] = pd.to_datetime(df['at'])
        df['date'] = df['at'].dt.strftime('%Y-%m-%d')
        
        # Clean and standardize columns
        df_clean = df[[
            'content',           # Review text
            'score',             # Rating (1-5)
            'date',              # Normalized date
            'app_name',          # Bank name
            'app_id',            # App identifier
            'scraped_at'         # When review was scraped
        ]].copy()
        
        # Rename columns for clarity
        df_clean.columns = ['review', 'rating', 'date', 'bank', 'app_id', 'scraped_at']
        
        # Remove duplicates based on review content and date
        df_clean = df_clean.drop_duplicates(subset=['review', 'date', 'bank'])
        
        # Filter out very short reviews (likely spam)
        df_clean = df_clean[df_clean['review'].str.len() > 10]
        
        logger.info(f"Preprocessing complete. Clean reviews: {len(df_clean)}")
        return df_clean
    
    def save_reviews(self, df, bank_name, file_type='csv'):
        """
        Save reviews to file
        
        Args:
            df (pd.DataFrame): Review data
            bank_name (str): Name of the bank
            file_type (str): File format ('csv' or 'json')
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if file_type == 'csv':
            filename = f"{bank_name}_reviews_{timestamp}.csv"
            filepath = self.raw_dir / filename
            df.to_csv(filepath, index=False, encoding='utf-8')
        elif file_type == 'json':
            filename = f"{bank_name}_reviews_{timestamp}.json"
            filepath = self.raw_dir / filename
            df.to_json(filepath, orient='records', indent=2)
        
        logger.info(f"Saved {len(df)} reviews to {filepath}")
        return filepath
    
    def run_scraping(self):
        """
        Main method to run the scraping process for all banks
        """
        logger.info("Starting banking app review scraping process")
        
        all_reviews = []
        scraping_summary = {}
        
        for bank_code, app_info in self.bank_apps.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {bank_code}: {app_info['name']}")
            logger.info(f"{'='*50}")
            
            # Scrape reviews
            reviews = self.scrape_app_reviews(
                app_info['app_id'],
                app_info['name'],
                app_info['target_reviews']
            )
            
            if reviews:
                # Preprocess
                df_clean = self.preprocess_reviews(reviews)
                
                if not df_clean.empty:
                    # Save individual bank data
                    self.save_reviews(df_clean, bank_code, 'csv')
                    self.save_reviews(df_clean, bank_code, 'json')
                    
                    # Add to combined dataset
                    all_reviews.append(df_clean)
                    
                    # Update summary
                    scraping_summary[bank_code] = {
                        'total_scraped': len(reviews),
                        'clean_reviews': len(df_clean),
                        'avg_rating': df_clean['rating'].mean(),
                        'date_range': f"{df_clean['date'].min()} to {df_clean['date'].max()}"
                    }
                    
                    logger.info(f"✅ {bank_code}: {len(df_clean)} clean reviews collected")
                else:
                    logger.warning(f"⚠️ {bank_code}: No clean reviews after preprocessing")
                    scraping_summary[bank_code] = {'error': 'No clean reviews'}
            else:
                logger.error(f"❌ {bank_code}: Failed to scrape reviews")
                scraping_summary[bank_code] = {'error': 'Scraping failed'}
            
            # Be respectful - wait between requests
            time.sleep(2)
        
        # Combine all reviews
        if all_reviews:
            combined_df = pd.concat(all_reviews, ignore_index=True)
            
            # Save combined dataset
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            combined_csv = self.raw_dir / f"all_bank_reviews_{timestamp}.csv"
            combined_df.to_csv(combined_csv, index=False, encoding='utf-8')
            
            logger.info(f"\n{'='*50}")
            logger.info("SCRAPING COMPLETE")
            logger.info(f"{'='*50}")
            logger.info(f"Total reviews collected: {len(combined_df)}")
            logger.info(f"Combined data saved to: {combined_csv}")
            
            # Print summary
            for bank, stats in scraping_summary.items():
                if 'error' not in stats:
                    logger.info(f"{bank}: {stats['clean_reviews']} reviews, "
                              f"Avg rating: {stats['avg_rating']:.2f}")
                else:
                    logger.info(f"{bank}: {stats['error']}")
            
            return combined_df, scraping_summary
        else:
            logger.error("No reviews were successfully scraped")
            return None, scraping_summary

def main():
    """Main execution function"""
    scraper = BankingAppScraper()
    
    try:
        combined_data, summary = scraper.run_scraping()
        
        if combined_data is not None:
            print(f"\n🎉 Successfully collected {len(combined_data)} total reviews!")
            print("Check the 'data/raw' directory for the scraped data files.")
        else:
            print("\n❌ Scraping failed. Check the logs for details.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\n❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
