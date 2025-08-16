"""
Sentiment Analysis Module for Banking App Reviews
Task 2: Sentiment and Thematic Analysis

This module provides comprehensive sentiment analysis using:
- DistilBERT (fine-tuned for sentiment)
- VADER (rule-based sentiment)
- TextBlob (additional validation)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Using VADER and TextBlob only.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Comprehensive sentiment analysis for banking app reviews"""
    
    def __init__(self, model_name='distilbert-base-uncased-finetuned-sst-2-english'):
        self.model_name = model_name
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize transformers if available
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading {model_name} for sentiment analysis...")
                self.transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=-1  # Use CPU
                )
                self.transformer_available = True
                logger.info("✅ Transformer model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load transformer model: {e}")
                self.transformer_available = False
        else:
            self.transformer_available = False
        
        # Sentiment mapping
        self.sentiment_mapping = {
            'POSITIVE': 'positive',
            'NEGATIVE': 'negative',
            'LABEL_0': 'negative',
            'LABEL_1': 'positive'
        }
        
        # Create output directory
        self.output_dir = Path('../data/processed')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_sentiment_transformer(self, text):
        """
        Analyze sentiment using transformer model
        
        Args:
            text (str): Review text
            
        Returns:
            dict: Sentiment analysis results
        """
        if not self.transformer_available:
            return None
        
        try:
            # Truncate text if too long for model
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            # Get prediction
            result = self.transformer_pipeline(text)[0]
            
            # Extract sentiment and confidence
            sentiment = result['label']
            confidence = result['score']
            
            # Map to standard format
            mapped_sentiment = self.sentiment_mapping.get(sentiment, sentiment.lower())
            
            return {
                'sentiment': mapped_sentiment,
                'confidence': confidence,
                'method': 'transformer'
            }
            
        except Exception as e:
            logger.warning(f"Transformer analysis failed for text: {str(e)}")
            return None
    
    def analyze_sentiment_vader(self, text):
        """
        Analyze sentiment using VADER
        
        Args:
            text (str): Review text
            
        Returns:
            dict: VADER sentiment scores
        """
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            
            # Determine sentiment based on compound score
            if scores['compound'] >= 0.05:
                sentiment = 'positive'
            elif scores['compound'] <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'method': 'vader'
            }
            
        except Exception as e:
            logger.warning(f"VADER analysis failed: {str(e)}")
            return None
    
    def analyze_sentiment_textblob(self, text):
        """
        Analyze sentiment using TextBlob
        
        Args:
            text (str): Review text
            
        Returns:
            dict: TextBlob sentiment results
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Map polarity to sentiment
            if polarity > 0:
                sentiment = 'positive'
            elif polarity < 0:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'method': 'textblob'
            }
            
        except Exception as e:
            logger.warning(f"TextBlob analysis failed: {str(e)}")
            return None
    
    def get_consensus_sentiment(self, results):
        """
        Get consensus sentiment from multiple methods
        
        Args:
            results (dict): Results from different sentiment methods
            
        Returns:
            dict: Consensus sentiment analysis
        """
        sentiments = []
        confidences = []
        
        # Collect all available results
        for method, result in results.items():
            if result and 'sentiment' in result:
                sentiments.append(result['sentiment'])
                if 'confidence' in result:
                    confidences.append(result['confidence'])
                elif 'compound' in result:
                    confidences.append(abs(result['compound']))
                elif 'polarity' in result:
                    confidences.append(abs(result['polarity']))
        
        if not sentiments:
            return {'sentiment': 'neutral', 'confidence': 0.0, 'method': 'consensus'}
        
        # Get most common sentiment
        from collections import Counter
        sentiment_counts = Counter(sentiments)
        consensus_sentiment = sentiment_counts.most_common(1)[0][0]
        
        # Calculate average confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Calculate agreement level
        agreement = sentiment_counts[consensus_sentiment] / len(sentiments)
        
        return {
            'sentiment': consensus_sentiment,
            'confidence': avg_confidence,
            'agreement': agreement,
            'method': 'consensus',
            'all_sentiments': dict(sentiment_counts)
        }
    
    def analyze_review_sentiment(self, text):
        """
        Perform comprehensive sentiment analysis on a single review
        
        Args:
            text (str): Review text
            
        Returns:
            dict: Comprehensive sentiment analysis results
        """
        results = {}
        
        # Run all available methods
        if self.transformer_available:
            results['transformer'] = self.analyze_sentiment_transformer(text)
        
        results['vader'] = self.analyze_sentiment_vader(text)
        results['textblob'] = self.analyze_sentiment_textblob(text)
        
        # Get consensus
        consensus = self.get_consensus_sentiment(results)
        results['consensus'] = consensus
        
        return results
    
    def analyze_batch_sentiment(self, df):
        """
        Analyze sentiment for a batch of reviews
        
        Args:
            df (pd.DataFrame): DataFrame with 'review' column
            
        Returns:
            pd.DataFrame: DataFrame with sentiment analysis results
        """
        logger.info(f"Starting sentiment analysis for {len(df)} reviews...")
        
        # Initialize results columns
        df_analyzed = df.copy()
        
        # Add sentiment columns
        sentiment_columns = [
            'sentiment_label', 'sentiment_confidence', 'sentiment_method',
            'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral',
            'textblob_polarity', 'textblob_subjectivity',
            'consensus_agreement', 'all_sentiments'
        ]
        
        for col in sentiment_columns:
            df_analyzed[col] = None
        
        # Process each review
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processed {idx}/{len(df)} reviews...")
            
            text = str(row['review'])
            
            # Analyze sentiment
            sentiment_results = self.analyze_review_sentiment(text)
            
            # Extract consensus results
            consensus = sentiment_results.get('consensus', {})
            df_analyzed.at[idx, 'sentiment_label'] = consensus.get('sentiment', 'neutral')
            df_analyzed.at[idx, 'sentiment_confidence'] = consensus.get('confidence', 0.0)
            df_analyzed.at[idx, 'sentiment_method'] = consensus.get('method', 'consensus')
            df_analyzed.at[idx, 'consensus_agreement'] = consensus.get('agreement', 0.0)
            df_analyzed.at[idx, 'all_sentiments'] = json.dumps(consensus.get('all_sentiments', {}))
            
            # Extract VADER results
            vader = sentiment_results.get('vader', {})
            df_analyzed.at[idx, 'vader_compound'] = vader.get('compound', 0.0)
            df_analyzed.at[idx, 'vader_positive'] = vader.get('positive', 0.0)
            df_analyzed.at[idx, 'vader_negative'] = vader.get('negative', 0.0)
            df_analyzed.at[idx, 'vader_neutral'] = vader.get('neutral', 0.0)
            
            # Extract TextBlob results
            textblob = sentiment_results.get('textblob', {})
            df_analyzed.at[idx, 'textblob_polarity'] = textblob.get('polarity', 0.0)
            df_analyzed.at[idx, 'textblob_subjectivity'] = textblob.get('subjectivity', 0.0)
        
        logger.info("✅ Sentiment analysis completed!")
        return df_analyzed
    
    def generate_sentiment_summary(self, df):
        """
        Generate summary statistics for sentiment analysis
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment analysis results
            
        Returns:
            dict: Sentiment summary statistics
        """
        summary = {}
        
        # Overall sentiment distribution
        sentiment_dist = df['sentiment_label'].value_counts()
        summary['overall_sentiment_distribution'] = sentiment_dist.to_dict()
        
        # Sentiment by bank
        bank_sentiment = df.groupby('bank')['sentiment_label'].value_counts().unstack(fill_value=0)
        summary['sentiment_by_bank'] = bank_sentiment.to_dict()
        
        # Average confidence by sentiment
        confidence_by_sentiment = df.groupby('sentiment_label')['sentiment_confidence'].mean()
        summary['average_confidence_by_sentiment'] = confidence_by_sentiment.to_dict()
        
        # Agreement level distribution
        agreement_stats = df['consensus_agreement'].describe()
        summary['agreement_statistics'] = agreement_stats.to_dict()
        
        # VADER compound score distribution
        vader_stats = df['vader_compound'].describe()
        summary['vader_compound_statistics'] = vader_stats.to_dict()
        
        # TextBlob polarity distribution
        textblob_stats = df['textblob_polarity'].describe()
        summary['textblob_polarity_statistics'] = textblob_stats.to_dict()
        
        return summary
    
    def save_sentiment_results(self, df, filename_prefix='sentiment_analysis'):
        """
        Save sentiment analysis results
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment analysis
            filename_prefix (str): Prefix for output files
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save CSV
        csv_file = self.output_dir / f"{filename_prefix}_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"Sentiment results saved to: {csv_file}")
        
        # Generate and save summary
        summary = self.generate_sentiment_summary(df)
        summary_file = self.output_dir / f"{filename_prefix}_summary_{timestamp}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Sentiment summary saved to: {summary_file}")
        
        return csv_file, summary_file

def main():
    """Main execution function for testing"""
    analyzer = SentimentAnalyzer()
    
    # Test with sample text
    test_texts = [
        "Great app, very easy to use and secure!",
        "The app crashes frequently when making transfers.",
        "Excellent user interface and fast loading times.",
        "Slow response times and difficult navigation."
    ]
    
    print("🧪 Testing Sentiment Analysis...")
    print("=" * 50)
    
    for text in test_texts:
        print(f"\nText: {text}")
        results = analyzer.analyze_review_sentiment(text)
        
        print(f"Consensus: {results['consensus']['sentiment']} "
              f"(confidence: {results['consensus']['confidence']:.3f})")
        
        if 'vader' in results:
            print(f"VADER: {results['vader']['sentiment']} "
                  f"(compound: {results['vader']['compound']:.3f})")
        
        if 'textblob' in results:
            print(f"TextBlob: {results['textblob']['sentiment']} "
                  f"(polarity: {results['textblob']['polarity']:.3f})")

if __name__ == "__main__":
    main()
