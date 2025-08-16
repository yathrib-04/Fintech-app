"""
Chart Generator Module
Task 4: Insights and Recommendations

This module generates various charts and visualizations for the banking app analysis:
- Sentiment analysis charts
- Rating distributions
- Theme analysis visualizations
- Bank comparison charts
- Trend analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime

# Configure matplotlib for better output
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Generates charts and visualizations for banking app analysis"""
    
    def __init__(self, output_dir: str = "reports/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up color schemes
        self.colors = {
            'positive': '#2E8B57',  # Sea Green
            'negative': '#DC143C',  # Crimson
            'neutral': '#696969',   # Dim Gray
            'banks': ['#FF6B6B', '#4ECDC4', '#45B7D1']  # CBE, BOA, Dashen
        }
        
        # Set figure size and DPI for high-quality output
        self.figsize = (12, 8)
        self.dpi = 300
        
        logger.info(f"Chart generator initialized. Output directory: {self.output_dir}")
    
    def create_sentiment_distribution_chart(self, df: pd.DataFrame, save: bool = True) -> str:
        """
        Create sentiment distribution chart across all banks
        
        Args:
            df: DataFrame with sentiment analysis data
            save: Whether to save the chart
            
        Returns:
            str: Path to saved chart file
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Count sentiment by bank
        sentiment_counts = df.groupby(['bank', 'sentiment_label']).size().unstack(fill_value=0)
        
        # Create stacked bar chart
        ax = sentiment_counts.plot(
            kind='bar', 
            stacked=True, 
            color=[self.colors['positive'], self.colors['negative'], self.colors['neutral']],
            alpha=0.8
        )
        
        plt.title('Sentiment Distribution by Bank', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Bank', fontsize=12)
        plt.ylabel('Number of Reviews', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Sentiment', title_fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            filename = f"sentiment_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Sentiment distribution chart saved to: {filepath}")
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return ""
    
    def create_rating_distribution_chart(self, df: pd.DataFrame, save: bool = True) -> str:
        """
        Create rating distribution chart
        
        Args:
            df: DataFrame with rating data
            save: Whether to save the chart
            
        Returns:
            str: Path to saved chart file
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Create subplots for each bank
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=self.dpi)
        fig.suptitle('Rating Distribution by Bank', fontsize=18, fontweight='bold', y=1.02)
        
        banks = df['bank'].unique()
        
        for i, bank in enumerate(banks):
            bank_data = df[df['bank'] == bank]
            rating_counts = bank_data['rating'].value_counts().sort_index()
            
            # Create bar chart for each bank
            bars = axes[i].bar(
                rating_counts.index, 
                rating_counts.values, 
                color=self.colors['banks'][i],
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5
            )
            
            axes[i].set_title(f'{bank}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Rating (Stars)', fontsize=12)
            axes[i].set_ylabel('Number of Reviews', fontsize=12)
            axes[i].set_xticks(range(1, 6))
            axes[i].grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[i].text(
                    bar.get_x() + bar.get_width()/2., 
                    height + height*0.01,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10
                )
        
        plt.tight_layout()
        
        if save:
            filename = f"rating_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Rating distribution chart saved to: {filepath}")
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return ""
    
    def create_theme_analysis_chart(self, df: pd.DataFrame, save: bool = True) -> str:
        """
        Create theme analysis visualization
        
        Args:
            df: DataFrame with theme analysis data
            save: Whether to save the chart
            
        Returns:
            str: Path to saved chart file
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Count themes by bank
        theme_counts = df.groupby(['bank', 'primary_theme']).size().unstack(fill_value=0)
        
        # Create heatmap
        plt.figure(figsize=(12, 8), dpi=self.dpi)
        sns.heatmap(
            theme_counts, 
            annot=True, 
            fmt='d', 
            cmap='YlOrRd',
            cbar_kws={'label': 'Number of Reviews'},
            linewidths=0.5
        )
        
        plt.title('Theme Distribution by Bank', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Primary Theme', fontsize=12)
        plt.ylabel('Bank', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save:
            filename = f"theme_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Theme analysis chart saved to: {filepath}")
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return ""
    
    def create_sentiment_trend_chart(self, df: pd.DataFrame, save: bool = True) -> str:
        """
        Create sentiment trend over time chart
        
        Args:
            df: DataFrame with sentiment and date data
            save: Whether to save the chart
            
        Returns:
            str: Path to saved chart file
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Convert date column to datetime if needed
        df['date'] = pd.to_datetime(df['date'])
        
        # Group by date and sentiment, count occurrences
        daily_sentiment = df.groupby([df['date'].dt.date, 'sentiment_label']).size().unstack(fill_value=0)
        
        # Create line chart
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in daily_sentiment.columns:
                plt.plot(
                    daily_sentiment.index, 
                    daily_sentiment[sentiment], 
                    marker='o', 
                    linewidth=2, 
                    markersize=6,
                    label=sentiment.capitalize(),
                    color=self.colors[sentiment]
                )
        
        plt.title('Sentiment Trends Over Time', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Number of Reviews', fontsize=12)
        plt.legend(title='Sentiment', title_fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save:
            filename = f"sentiment_trends_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Sentiment trend chart saved to: {filepath}")
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return ""
    
    def create_bank_comparison_chart(self, df: pd.DataFrame, save: bool = True) -> str:
        """
        Create comprehensive bank comparison chart
        
        Args:
            df: DataFrame with all analysis data
            save: Whether to save the chart
            
        Returns:
            str: Path to saved chart file
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
        fig.suptitle('Comprehensive Bank Comparison Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        banks = df['bank'].unique()
        
        # 1. Average Rating by Bank
        avg_ratings = df.groupby('bank')['rating'].mean()
        axes[0, 0].bar(banks, avg_ratings.values, color=self.colors['banks'], alpha=0.8)
        axes[0, 0].set_title('Average Rating by Bank', fontweight='bold')
        axes[0, 0].set_ylabel('Average Rating')
        axes[0, 0].set_ylim(0, 5)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(avg_ratings.values):
            axes[0, 0].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Sentiment Distribution by Bank
        sentiment_pct = df.groupby(['bank', 'sentiment_label']).size().groupby('bank').apply(
            lambda x: x / x.sum() * 100
        ).unstack(fill_value=0)
        
        sentiment_pct.plot(
            kind='bar', 
            ax=axes[0, 1], 
            color=[self.colors['positive'], self.colors['negative'], self.colors['neutral']],
            alpha=0.8
        )
        axes[0, 1].set_title('Sentiment Distribution by Bank (%)', fontweight='bold')
        axes[0, 1].set_ylabel('Percentage')
        axes[0, 1].legend(title='Sentiment')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Review Count by Bank
        review_counts = df.groupby('bank').size()
        axes[1, 0].pie(review_counts.values, labels=review_counts.index, autopct='%1.1f%%', 
                       colors=self.colors['banks'], startangle=90)
        axes[1, 0].set_title('Review Count Distribution', fontweight='bold')
        
        # 4. Theme Distribution by Bank
        theme_counts = df.groupby(['bank', 'primary_theme']).size().unstack(fill_value=0)
        theme_counts.plot(kind='bar', ax=axes[1, 1], alpha=0.8)
        axes[1, 1].set_title('Theme Distribution by Bank', fontweight='bold')
        axes[1, 1].set_ylabel('Number of Reviews')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend(title='Theme', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save:
            filename = f"bank_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Bank comparison chart saved to: {filepath}")
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return ""
    
    def create_keyword_cloud(self, df: pd.DataFrame, save: bool = True) -> str:
        """
        Create keyword cloud visualization
        
        Args:
            df: DataFrame with keyword data
            save: Whether to save the chart
            
        Returns:
            str: Path to saved chart file
        """
        try:
            from wordcloud import WordCloud
        except ImportError:
            logger.warning("WordCloud not available. Skipping keyword cloud generation.")
            return ""
        
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Extract and process keywords
        all_keywords = []
        for keywords_str in df['keywords'].dropna():
            try:
                if isinstance(keywords_str, str):
                    keywords = json.loads(keywords_str)
                    if isinstance(keywords, dict):
                        # If keywords is a dict with scores, use the keys
                        all_keywords.extend(list(keywords.keys()))
                    elif isinstance(keywords, list):
                        all_keywords.extend(keywords)
            except (json.JSONDecodeError, TypeError):
                continue
        
        if not all_keywords:
            logger.warning("No valid keywords found for word cloud")
            return ""
        
        # Create word cloud
        text = ' '.join(all_keywords)
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5
        ).generate(text)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Keywords in Reviews', fontsize=16, fontweight='bold', pad=20)
        
        if save:
            filename = f"keyword_cloud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            logger.info(f"Keyword cloud saved to: {filepath}")
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return ""
    
    def generate_all_charts(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate all charts for the analysis
        
        Args:
            df: DataFrame with all analysis data
            
        Returns:
            Dict[str, str]: Dictionary mapping chart names to file paths
        """
        logger.info("Generating all charts...")
        
        charts = {}
        
        try:
            # Generate each chart type
            charts['sentiment_distribution'] = self.create_sentiment_distribution_chart(df)
            charts['rating_distribution'] = self.create_rating_distribution_chart(df)
            charts['theme_analysis'] = self.create_theme_analysis_chart(df)
            charts['sentiment_trends'] = self.create_sentiment_trend_chart(df)
            charts['bank_comparison'] = self.create_bank_comparison_chart(df)
            charts['keyword_cloud'] = self.create_keyword_cloud(df)
            
            logger.info(f"✅ Generated {len(charts)} charts successfully")
            
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")
        
        return charts

def main():
    """Test the chart generator"""
    print("📊 Chart Generator Test")
    print("=" * 40)
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'bank': ['CBE', 'CBE', 'BOA', 'BOA', 'Dashen', 'Dashen'] * 10,
        'rating': np.random.randint(1, 6, 60),
        'sentiment_label': np.random.choice(['positive', 'negative', 'neutral'], 60),
        'primary_theme': np.random.choice(['UI', 'Performance', 'Security', 'Features'], 60),
        'date': pd.date_range('2024-01-01', periods=60, freq='D'),
        'keywords': ['["fast", "easy", "good"]'] * 60
    })
    
    # Initialize chart generator
    generator = ChartGenerator()
    
    # Test chart generation
    try:
        charts = generator.generate_all_charts(sample_data)
        print(f"Generated {len(charts)} charts:")
        for chart_name, filepath in charts.items():
            print(f"  {chart_name}: {filepath}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
