"""
Demo Script for Task 2: Sentiment and Thematic Analysis
Fintech App Analysis Project

This script demonstrates how to use the Task 2 functionality
without actually running the full analysis process.
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

def demo_task2_functionality():
    """
    Demonstrate Task 2 functionality with sample data
    """
    print("🚀 Task 2 Demo: Sentiment and Thematic Analysis")
    print("=" * 60)
    
    try:
        # Import our modules
        from analysis.sentiment_analyzer import SentimentAnalyzer
        from analysis.theme_analyzer import ThemeAnalyzer
        
        print("✅ Successfully imported all modules")
        
        # Create sample data for demonstration
        print("\n📊 Creating sample data for demonstration...")
        sample_data = create_sample_data()
        
        # Demonstrate sentiment analysis
        print("\n🔍 Demonstrating sentiment analysis...")
        sentiment_analyzer = SentimentAnalyzer()
        
        # Test individual review sentiment
        test_text = "Great app, very easy to use and secure!"
        sentiment_result = sentiment_analyzer.analyze_review_sentiment(test_text)
        
        print(f"   Sample text: {test_text}")
        print(f"   Consensus sentiment: {sentiment_result['consensus']['sentiment']}")
        print(f"   Confidence: {sentiment_result['consensus']['confidence']:.3f}")
        
        # Demonstrate thematic analysis
        print("\n🎯 Demonstrating thematic analysis...")
        theme_analyzer = ThemeAnalyzer()
        
        # Test individual review themes
        test_text = "The app is very slow when making transfers and crashes frequently."
        theme_result = theme_analyzer.analyze_review_themes(test_text)
        
        print(f"   Sample text: {test_text}")
        print(f"   Primary theme: {theme_result['primary_theme']}")
        print(f"   Keywords: {[kw for kw, _ in theme_result['keywords'][:5]]}")
        
        # Show banking themes configuration
        print("\n🏦 Banking Theme Categories:")
        for theme_name, theme_info in theme_analyzer.banking_themes.items():
            print(f"   {theme_name}: {theme_info['description']}")
            print(f"      Keywords: {', '.join(theme_info['keywords'][:5])}...")
        
        # Demonstrate batch processing on sample data
        print("\n📈 Demonstrating batch processing...")
        
        # Process sentiment
        df_with_sentiment = sentiment_analyzer.analyze_batch_sentiment(sample_data)
        print(f"   ✅ Sentiment analysis completed on {len(df_with_sentiment)} reviews")
        
        # Process themes
        df_with_themes = theme_analyzer.analyze_batch_themes(df_with_sentiment)
        print(f"   ✅ Thematic analysis completed on {len(df_with_themes)} reviews")
        
        # Show results summary
        print("\n📊 Analysis Results Summary:")
        sentiment_dist = df_with_themes['sentiment_label'].value_counts()
        theme_dist = df_with_themes['primary_theme'].value_counts()
        
        print(f"   Sentiment distribution: {sentiment_dist.to_dict()}")
        print(f"   Theme distribution: {theme_dist.to_dict()}")
        
        print("\n🎯 Task 2 Demo Complete!")
        print("   The system is ready for actual sentiment and thematic analysis.")
        print("   Run 'python src/analysis/run_task2.py' to execute the full workflow.")
        
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
    
    # Sample review texts with different sentiments and themes
    sample_reviews = [
        "Great app, very easy to use and secure!",
        "The app crashes frequently when making transfers.",
        "Excellent user interface and fast loading times.",
        "Slow response times and difficult navigation.",
        "Best banking app I've ever used, highly recommend",
        "Frequent login issues and poor customer support.",
        "Clean design and intuitive features",
        "App freezes during transactions, needs improvement",
        "Reliable and secure mobile banking experience",
        "Poor performance and outdated interface",
        "Transfer feature works perfectly, very fast",
        "Security features are excellent, feel safe using this app",
        "Customer service is responsive and helpful",
        "App is stable and rarely crashes",
        "Navigation is confusing and menu structure is poor"
    ]
    
    # Sample banks
    banks = ['CBE', 'BOA', 'Dashen']
    
    # Generate sample data
    data = []
    for i in range(30):  # 30 sample reviews
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
    demo_task2_functionality()
