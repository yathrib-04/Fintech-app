"""
Task 2 Runner: Sentiment and Thematic Analysis
Fintech App Analysis Project

This script orchestrates the complete Task 2 workflow:
1. Load processed data from Task 1
2. Perform sentiment analysis using multiple methods
3. Extract themes and keywords
4. Generate comprehensive analysis reports
5. Save results for next tasks
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

from analysis.sentiment_analyzer import SentimentAnalyzer
from analysis.theme_analyzer import ThemeAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('task2_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_task1_data():
    """
    Load processed data from Task 1
    
    Returns:
        pd.DataFrame: Processed review data
    """
    logger.info("Loading Task 1 processed data...")
    
    # Look for processed data files
    processed_dir = Path('../data/processed')
    
    # Find the most recent processed file
    processed_files = list(processed_dir.glob('all_banks_processed_*.csv'))
    
    if not processed_files:
        # Fallback: look for individual bank files
        processed_files = list(processed_dir.glob('*_processed_*.csv'))
    
    if not processed_files:
        raise FileNotFoundError("No processed data found from Task 1. Please run Task 1 first.")
    
    # Load the most recent file
    latest_file = max(processed_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading data from: {latest_file}")
    
    df = pd.read_csv(latest_file)
    logger.info(f"Loaded {len(df)} reviews from Task 1")
    
    return df

def run_sentiment_analysis(df):
    """
    Run sentiment analysis on the review data
    
    Args:
        df (pd.DataFrame): Review data
        
    Returns:
        pd.DataFrame: Data with sentiment analysis results
    """
    logger.info("="*50)
    logger.info("STEP 1: SENTIMENT ANALYSIS")
    logger.info("="*50)
    
    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer()
    
    # Run sentiment analysis
    df_with_sentiment = sentiment_analyzer.analyze_batch_sentiment(df)
    
    # Save sentiment results
    sentiment_csv, sentiment_summary = sentiment_analyzer.save_sentiment_results(
        df_with_sentiment, 'sentiment_analysis'
    )
    
    logger.info(f"✅ Sentiment analysis completed!")
    logger.info(f"   Results saved to: {sentiment_csv}")
    logger.info(f"   Summary saved to: {sentiment_summary}")
    
    return df_with_sentiment, sentiment_summary

def run_thematic_analysis(df):
    """
    Run thematic analysis on the review data
    
    Args:
        df (pd.DataFrame): Review data
        
    Returns:
        pd.DataFrame: Data with thematic analysis results
    """
    logger.info("="*50)
    logger.info("STEP 2: THEMATIC ANALYSIS")
    logger.info("="*50)
    
    # Initialize theme analyzer
    theme_analyzer = ThemeAnalyzer()
    
    # Run thematic analysis
    df_with_themes = theme_analyzer.analyze_batch_themes(df)
    
    # Save theme results
    theme_csv, theme_summary = theme_analyzer.save_theme_results(
        df_with_themes, 'theme_analysis'
    )
    
    logger.info(f"✅ Thematic analysis completed!")
    logger.info(f"   Results saved to: {theme_csv}")
    logger.info(f"   Summary saved to: {theme_summary}")
    
    return df_with_themes, theme_summary

def perform_cross_analysis(df):
    """
    Perform cross-analysis between sentiment and themes
    
    Args:
        df (pd.DataFrame): Data with both sentiment and theme analysis
        
    Returns:
        dict: Cross-analysis results
    """
    logger.info("="*50)
    logger.info("STEP 3: CROSS-ANALYSIS")
    logger.info("="*50)
    
    cross_analysis = {}
    
    # Sentiment vs Theme analysis
    sentiment_theme_cross = df.groupby(['sentiment_label', 'primary_theme']).size().unstack(fill_value=0)
    cross_analysis['sentiment_vs_theme'] = sentiment_theme_cross.to_dict()
    
    # Bank vs Sentiment analysis
    bank_sentiment_cross = df.groupby(['bank', 'sentiment_label']).size().unstack(fill_value=0)
    cross_analysis['bank_vs_sentiment'] = bank_sentiment_cross.to_dict()
    
    # Bank vs Theme analysis
    bank_theme_cross = df.groupby(['bank', 'primary_theme']).size().unstack(fill_value=0)
    cross_analysis['bank_vs_theme'] = bank_theme_cross.to_dict()
    
    # Rating vs Sentiment analysis
    rating_sentiment_cross = df.groupby(['rating', 'sentiment_label']).size().unstack(fill_value=0)
    cross_analysis['rating_vs_sentiment'] = rating_sentiment_cross.to_dict()
    
    # Rating vs Theme analysis
    rating_theme_cross = df.groupby(['rating', 'primary_theme']).size().unstack(fill_value=0)
    cross_analysis['rating_vs_theme'] = rating_theme_cross.to_dict()
    
    logger.info("✅ Cross-analysis completed!")
    return cross_analysis

def generate_insights(df, cross_analysis):
    """
    Generate actionable insights from the analysis
    
    Args:
        df (pd.DataFrame): Complete analyzed data
        cross_analysis (dict): Cross-analysis results
        
    Returns:
        dict: Generated insights
    """
    logger.info("="*50)
    logger.info("STEP 4: INSIGHT GENERATION")
    logger.info("="*50)
    
    insights = {
        'timestamp': datetime.now().isoformat(),
        'overview': {
            'total_reviews': len(df),
            'banks_analyzed': list(df['bank'].unique()),
            'sentiment_distribution': df['sentiment_label'].value_counts().to_dict(),
            'theme_distribution': df['primary_theme'].value_counts().to_dict()
        },
        'key_findings': [],
        'bank_specific_insights': {},
        'recommendations': []
    }
    
    # Overall sentiment insights
    sentiment_dist = df['sentiment_label'].value_counts()
    total_reviews = len(df)
    
    if 'positive' in sentiment_dist:
        positive_pct = (sentiment_dist['positive'] / total_reviews) * 100
        insights['key_findings'].append({
            'finding': f"Overall positive sentiment: {positive_pct:.1f}% of reviews are positive",
            'type': 'sentiment',
            'metric': positive_pct
        })
    
    if 'negative' in sentiment_dist:
        negative_pct = (sentiment_dist['negative'] / total_reviews) * 100
        insights['key_findings'].append({
            'finding': f"Overall negative sentiment: {negative_pct:.1f}% of reviews are negative",
            'type': 'sentiment',
            'metric': negative_pct
        })
    
    # Theme insights
    theme_dist = df['primary_theme'].value_counts()
    top_theme = theme_dist.index[0] if len(theme_dist) > 0 else 'general'
    top_theme_count = theme_dist.iloc[0] if len(theme_dist) > 0 else 0
    
    insights['key_findings'].append({
        'finding': f"Most common theme: '{top_theme}' appears in {top_theme_count} reviews",
        'type': 'theme',
        'metric': top_theme_count
    })
    
    # Bank-specific insights
    for bank in df['bank'].unique():
        bank_data = df[df['bank'] == bank]
        bank_insights = {
            'total_reviews': len(bank_data),
            'avg_rating': bank_data['rating'].mean(),
            'sentiment_distribution': bank_data['sentiment_label'].value_counts().to_dict(),
            'top_themes': bank_data['primary_theme'].value_counts().head(3).to_dict(),
            'key_issues': [],
            'strengths': []
        }
        
        # Identify key issues (negative sentiment + specific themes)
        negative_reviews = bank_data[bank_data['sentiment_label'] == 'negative']
        if len(negative_reviews) > 0:
            negative_themes = negative_reviews['primary_theme'].value_counts()
            if len(negative_themes) > 0:
                top_negative_theme = negative_themes.index[0]
                bank_insights['key_issues'].append(f"High negative sentiment around '{top_negative_theme}' theme")
        
        # Identify strengths (positive sentiment + specific themes)
        positive_reviews = bank_data[bank_data['sentiment_label'] == 'positive']
        if len(positive_reviews) > 0:
            positive_themes = positive_reviews['primary_theme'].value_counts()
            if len(positive_themes) > 0:
                top_positive_theme = positive_themes.index[0]
                bank_insights['strengths'].append(f"Strong positive sentiment around '{top_positive_theme}' theme")
        
        insights['bank_specific_insights'][bank] = bank_data
    
    # Generate recommendations
    insights['recommendations'] = generate_recommendations(df, cross_analysis)
    
    logger.info("✅ Insights generated!")
    return insights

def generate_recommendations(df, cross_analysis):
    """
    Generate actionable recommendations based on analysis
    
    Args:
        df (pd.DataFrame): Analyzed data
        cross_analysis (dict): Cross-analysis results
        
    Returns:
        list: List of recommendations
    """
    recommendations = []
    
    # Analyze sentiment distribution
    sentiment_dist = df['sentiment_label'].value_counts()
    total_reviews = len(df)
    
    if 'negative' in sentiment_dist:
        negative_pct = (sentiment_dist['negative'] / total_reviews) * 100
        if negative_pct > 30:
            recommendations.append({
                'priority': 'high',
                'category': 'sentiment',
                'recommendation': f"High negative sentiment ({negative_pct:.1f}%) - Focus on addressing user pain points",
                'action': 'Conduct user research to identify specific issues'
            })
    
    # Analyze theme distribution
    theme_dist = df['primary_theme'].value_counts()
    if len(theme_dist) > 0:
        top_theme = theme_dist.index[0]
        top_theme_count = theme_dist.iloc[0]
        top_theme_pct = (top_theme_count / total_reviews) * 100
        
        if top_theme_pct > 25:
            recommendations.append({
                'priority': 'medium',
                'category': 'theme',
                'recommendation': f"'{top_theme}' is the dominant theme ({top_theme_pct:.1f}%) - Consider if this aligns with business goals",
                'action': 'Review if this theme focus supports strategic objectives'
            })
    
    # Bank-specific recommendations
    for bank in df['bank'].unique():
        bank_data = df[df['bank'] == bank]
        bank_negative = bank_data[bank_data['sentiment_label'] == 'negative']
        
        if len(bank_negative) > 0:
            negative_themes = bank_negative['primary_theme'].value_counts()
            if len(negative_themes) > 0:
                top_negative_theme = negative_themes.index[0]
                recommendations.append({
                    'priority': 'high',
                    'category': 'bank_specific',
                    'recommendation': f"{bank}: Focus on improving '{top_negative_theme}' to reduce negative sentiment",
                    'action': f'Prioritize {top_negative_theme} improvements in {bank} app development roadmap'
                })
    
    return recommendations

def save_final_results(df, insights, cross_analysis):
    """
    Save final Task 2 results
    
    Args:
        df (pd.DataFrame): Complete analyzed data
        insights (dict): Generated insights
        cross_analysis (dict): Cross-analysis results
    """
    logger.info("="*50)
    logger.info("STEP 5: SAVING FINAL RESULTS")
    logger.info("="*50)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('../data/processed')
    
    # Save complete analyzed data
    final_csv = output_dir / f"task2_complete_analysis_{timestamp}.csv"
    df.to_csv(final_csv, index=False, encoding='utf-8')
    logger.info(f"Complete analysis saved to: {final_csv}")
    
    # Save insights
    insights_file = output_dir / f"task2_insights_{timestamp}.json"
    with open(insights_file, 'w') as f:
        json.dump(insights, f, indent=2)
    logger.info(f"Insights saved to: {insights_file}")
    
    # Save cross-analysis
    cross_analysis_file = output_dir / f"task2_cross_analysis_{timestamp}.json"
    with open(cross_analysis_file, 'w') as f:
        json.dump(cross_analysis, f, indent=2)
    logger.info(f"Cross-analysis saved to: {cross_analysis_file}")
    
    # Save comprehensive summary
    summary = {
        'task': 'Task 2: Sentiment and Thematic Analysis',
        'timestamp': timestamp,
        'overview': insights['overview'],
        'key_findings': insights['key_findings'],
        'recommendations': insights['recommendations'],
        'files_generated': [
            str(final_csv),
            str(insights_file),
            str(cross_analysis_file)
        ]
    }
    
    summary_file = output_dir / f"task2_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_file}")
    
    return final_csv, insights_file, cross_analysis_file, summary_file

def run_task2():
    """
    Execute the complete Task 2 workflow
    """
    logger.info("="*60)
    logger.info("STARTING TASK 2: SENTIMENT AND THEMATIC ANALYSIS")
    logger.info("="*60)
    
    start_time = datetime.now()
    
    try:
        # Step 1: Load Task 1 data
        logger.info("Loading data from Task 1...")
        df = load_task1_data()
        
        # Step 2: Sentiment Analysis
        df_with_sentiment, sentiment_summary = run_sentiment_analysis(df)
        
        # Step 3: Thematic Analysis
        df_with_themes, theme_summary = run_thematic_analysis(df_with_sentiment)
        
        # Step 4: Cross-analysis
        cross_analysis = perform_cross_analysis(df_with_themes)
        
        # Step 5: Generate insights
        insights = generate_insights(df_with_themes, cross_analysis)
        
        # Step 6: Save final results
        final_csv, insights_file, cross_analysis_file, summary_file = save_final_results(
            df_with_themes, insights, cross_analysis
        )
        
        # Task completion
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info("TASK 2 COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Duration: {duration}")
        logger.info(f"Total reviews analyzed: {len(df_with_themes)}")
        logger.info(f"Sentiment analysis: ✅ Complete")
        logger.info(f"Thematic analysis: ✅ Complete")
        logger.info(f"Cross-analysis: ✅ Complete")
        logger.info(f"Insights generated: ✅ Complete")
        logger.info(f"Results saved to: {final_csv}")
        logger.info(f"Ready for Task 3: Database Storage in Oracle")
        
        return True
        
    except Exception as e:
        logger.error(f"Task 2 failed with error: {str(e)}")
        return False

def main():
    """Main execution function"""
    print("🚀 Starting Task 2: Sentiment and Thematic Analysis")
    print("=" * 60)
    
    success = run_task2()
    
    if success:
        print("\n🎉 Task 2 completed successfully!")
        print("📊 Sentiment analysis completed with multiple methods")
        print("🎯 Thematic analysis identified key themes and keywords")
        print("🔍 Cross-analysis revealed insights across dimensions")
        print("💡 Actionable insights and recommendations generated")
        print("📁 Results saved in 'data/processed' directory")
        print("➡️  Ready to proceed with Task 3: Database Storage")
    else:
        print("\n❌ Task 2 failed. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
