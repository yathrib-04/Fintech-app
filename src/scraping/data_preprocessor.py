"""
Data Preprocessing Script for Banking App Reviews
Task 1: Data Collection and Preprocessing

This script handles data cleaning, validation, and quality checks
for the scraped Google Play Store reviews.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import re
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReviewDataPreprocessor:
    """Preprocesses and validates scraped review data"""
    
    def __init__(self, data_dir='../data'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def load_latest_data(self):
        """
        Load the most recent scraped data files
        
        Returns:
            dict: Dictionary with bank codes as keys and DataFrames as values
        """
        bank_data = {}
        
        # Find the most recent combined file
        combined_files = list(self.raw_dir.glob('all_bank_reviews_*.csv'))
        if combined_files:
            latest_combined = max(combined_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading combined data from: {latest_combined}")
            
            df = pd.read_csv(latest_combined)
            
            # Split by bank
            for bank in df['bank'].unique():
                bank_data[bank] = df[df['bank'] == bank].copy()
                logger.info(f"Loaded {len(bank_data[bank])} reviews for {bank}")
        else:
            # Fallback: load individual bank files
            logger.info("No combined file found, loading individual bank files")
            for bank_file in self.raw_dir.glob('*_reviews_*.csv'):
                if not bank_file.name.startswith('all_bank_reviews'):
                    bank_code = bank_file.name.split('_')[0]
                    bank_data[bank_code] = pd.read_csv(bank_file)
                    logger.info(f"Loaded {len(bank_data[bank_code])} reviews for {bank_code}")
        
        return bank_data
    
    def clean_text(self, text):
        """
        Clean review text by removing special characters and normalizing
        
        Args:
            text (str): Raw review text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text
    
    def validate_rating(self, rating):
        """
        Validate and clean rating values
        
        Args:
            rating: Rating value
            
        Returns:
            float: Valid rating (1.0-5.0) or NaN
        """
        try:
            rating = float(rating)
            if 1.0 <= rating <= 5.0:
                return rating
            else:
                return np.nan
        except (ValueError, TypeError):
            return np.nan
    
    def validate_date(self, date_str):
        """
        Validate and normalize date strings
        
        Args:
            date_str: Date string
            
        Returns:
            str: Normalized date in YYYY-MM-DD format or NaN
        """
        try:
            # Try to parse the date
            parsed_date = pd.to_datetime(date_str)
            return parsed_date.strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            return np.nan
    
    def remove_duplicates(self, df):
        """
        Remove duplicate reviews while preserving the most recent ones
        
        Args:
            df (pd.DataFrame): Review data
            
        Returns:
            pd.DataFrame: Data with duplicates removed
        """
        initial_count = len(df)
        
        # Sort by date (most recent first) and remove duplicates
        df_sorted = df.sort_values('date', ascending=False)
        df_deduped = df_sorted.drop_duplicates(
            subset=['review', 'bank'], 
            keep='first'
        )
        
        removed_count = initial_count - len(df_deduped)
        logger.info(f"Removed {removed_count} duplicate reviews")
        
        return df_deduped
    
    def quality_check(self, df, bank_name):
        """
        Perform quality checks on the data
        
        Args:
            df (pd.DataFrame): Review data
            bank_name (str): Name of the bank
            
        Returns:
            dict: Quality metrics
        """
        total_reviews = len(df)
        
        # Check for missing values
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / total_reviews * 100).round(2)
        
        # Check rating distribution
        rating_dist = df['rating'].value_counts().sort_index()
        
        # Check date range
        date_range = f"{df['date'].min()} to {df['date'].max()}"
        
        # Check text length distribution
        text_lengths = df['review'].str.len()
        avg_length = text_lengths.mean()
        min_length = text_lengths.min()
        max_length = text_lengths.max()
        
        quality_metrics = {
            'bank': bank_name,
            'total_reviews': total_reviews,
            'missing_data_percentage': missing_percentage.to_dict(),
            'rating_distribution': rating_dist.to_dict(),
            'date_range': date_range,
            'text_length_stats': {
                'average': round(avg_length, 2),
                'minimum': min_length,
                'maximum': max_length
            }
        }
        
        # Log quality metrics
        logger.info(f"\nQuality Check for {bank_name}:")
        logger.info(f"Total reviews: {total_reviews}")
        logger.info(f"Missing data: {missing_percentage.to_dict()}")
        logger.info(f"Rating distribution: {rating_dist.to_dict()}")
        logger.info(f"Date range: {date_range}")
        logger.info(f"Text length - Avg: {avg_length:.2f}, Min: {min_length}, Max: {max_length}")
        
        return quality_metrics
    
    def preprocess_bank_data(self, df, bank_name):
        """
        Preprocess data for a specific bank
        
        Args:
            df (pd.DataFrame): Raw review data
            bank_name (str): Name of the bank
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        logger.info(f"Preprocessing data for {bank_name}")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Clean text
        df_clean['review'] = df_clean['review'].apply(self.clean_text)
        
        # Validate ratings
        df_clean['rating'] = df_clean['rating'].apply(self.validate_rating)
        
        # Validate dates
        df_clean['date'] = df_clean['date'].apply(self.validate_date)
        
        # Remove rows with missing critical data
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=['review', 'rating', 'date'])
        removed_count = initial_count - len(df_clean)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} rows with missing critical data")
        
        # Remove duplicates
        df_clean = self.remove_duplicates(df_clean)
        
        # Filter out very short reviews (likely spam)
        df_clean = df_clean[df_clean['review'].str.len() >= 15]
        
        # Reset index
        df_clean = df_clean.reset_index(drop=True)
        
        logger.info(f"Preprocessing complete for {bank_name}: {len(df_clean)} clean reviews")
        return df_clean
    
    def run_preprocessing(self):
        """
        Main method to run preprocessing for all banks
        
        Returns:
            tuple: (processed_data, quality_report)
        """
        logger.info("Starting data preprocessing process")
        
        # Load data
        bank_data = self.load_latest_data()
        
        if not bank_data:
            logger.error("No data found to preprocess")
            return None, None
        
        processed_data = {}
        quality_report = {}
        
        # Preprocess each bank's data
        for bank_name, df in bank_data.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {bank_name}")
            logger.info(f"{'='*50}")
            
            # Preprocess
            df_processed = self.preprocess_bank_data(df, bank_name)
            
            if not df_processed.empty:
                # Quality check
                quality_metrics = self.quality_check(df_processed, bank_name)
                quality_report[bank_name] = quality_metrics
                
                # Save processed data
                processed_data[bank_name] = df_processed
                
                # Save individual processed file
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                processed_file = self.processed_dir / f"{bank_name}_processed_{timestamp}.csv"
                df_processed.to_csv(processed_file, index=False, encoding='utf-8')
                logger.info(f"Saved processed data to: {processed_file}")
            else:
                logger.warning(f"No valid data remaining for {bank_name}")
        
        # Combine all processed data
        if processed_data:
            combined_processed = pd.concat(processed_data.values(), ignore_index=True)
            
            # Save combined processed data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            combined_file = self.processed_dir / f"all_banks_processed_{timestamp}.csv"
            combined_processed.to_csv(combined_file, index=False, encoding='utf-8')
            
            # Save quality report
            quality_file = self.processed_dir / f"quality_report_{timestamp}.json"
            with open(quality_file, 'w') as f:
                json.dump(quality_report, f, indent=2)
            
            logger.info(f"\n{'='*50}")
            logger.info("PREPROCESSING COMPLETE")
            logger.info(f"{'='*50}")
            logger.info(f"Total processed reviews: {len(combined_processed)}")
            logger.info(f"Combined data saved to: {combined_file}")
            logger.info(f"Quality report saved to: {quality_file}")
            
            return combined_processed, quality_report
        else:
            logger.error("No data was successfully preprocessed")
            return None, None

def main():
    """Main execution function"""
    preprocessor = ReviewDataPreprocessor()
    
    try:
        processed_data, quality_report = preprocessor.run_preprocessing()
        
        if processed_data is not None:
            print(f"\n🎉 Successfully preprocessed {len(processed_data)} total reviews!")
            print("Check the 'data/processed' directory for the cleaned data files.")
            
            # Print summary
            for bank, metrics in quality_report.items():
                print(f"\n{bank}:")
                print(f"  - Clean reviews: {metrics['total_reviews']}")
                print(f"  - Avg rating: {metrics['rating_distribution']}")
                print(f"  - Date range: {metrics['date_range']}")
        else:
            print("\n❌ Preprocessing failed. Check the logs for details.")
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\n❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
