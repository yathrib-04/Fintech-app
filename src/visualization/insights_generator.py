"""
Insights Generator Module
Task 4: Insights and Recommendations

This module analyzes the banking app review data to generate:
- Key insights about satisfaction drivers and pain points
- Bank performance comparisons
- Actionable recommendations for improvement
- Business impact analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class InsightsGenerator:
    """Generates business insights from banking app review analysis"""
    
    def __init__(self):
        self.insights = {}
        self.recommendations = {}
        self.performance_metrics = {}
        
        logger.info("Insights generator initialized")
    
    def analyze_satisfaction_drivers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify key satisfaction drivers for each bank
        
        Args:
            df: DataFrame with review and sentiment data
            
        Returns:
            Dict containing satisfaction driver analysis
        """
        logger.info("Analyzing satisfaction drivers...")
        
        drivers = {}
        
        for bank in df['bank'].unique():
            bank_data = df[df['bank'] == bank]
            
            # Analyze positive reviews (4-5 stars and positive sentiment)
            positive_reviews = bank_data[
                (bank_data['rating'] >= 4) & 
                (bank_data['sentiment_label'] == 'positive')
            ]
            
            # Extract common themes from positive reviews
            positive_themes = positive_reviews['primary_theme'].value_counts().head(3)
            
            # Extract common keywords from positive reviews
            positive_keywords = self._extract_common_keywords(positive_reviews)
            
            drivers[bank] = {
                'positive_review_count': len(positive_reviews),
                'positive_review_percentage': len(positive_reviews) / len(bank_data) * 100,
                'top_positive_themes': positive_themes.to_dict(),
                'top_positive_keywords': positive_keywords,
                'average_rating': bank_data['rating'].mean(),
                'sentiment_score': bank_data['sentiment_confidence'].mean()
            }
        
        self.insights['satisfaction_drivers'] = drivers
        return drivers
    
    def analyze_pain_points(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify key pain points for each bank
        
        Args:
            df: DataFrame with review and sentiment data
            
        Returns:
            Dict containing pain point analysis
        """
        logger.info("Analyzing pain points...")
        
        pain_points = {}
        
        for bank in df['bank'].unique():
            bank_data = df[df['bank'] == bank]
            
            # Analyze negative reviews (1-2 stars and negative sentiment)
            negative_reviews = bank_data[
                (bank_data['rating'] <= 2) & 
                (bank_data['sentiment_label'] == 'negative')
            ]
            
            # Extract common themes from negative reviews
            negative_themes = negative_reviews['primary_theme'].value_counts().head(3)
            
            # Extract common keywords from negative reviews
            negative_keywords = self._extract_common_keywords(negative_reviews)
            
            # Analyze review text for common complaints
            common_complaints = self._extract_common_complaints(negative_reviews)
            
            pain_points[bank] = {
                'negative_review_count': len(negative_reviews),
                'negative_review_percentage': len(negative_reviews) / len(bank_data) * 100,
                'top_negative_themes': negative_themes.to_dict(),
                'top_negative_keywords': negative_keywords,
                'common_complaints': common_complaints,
                'critical_issues': self._identify_critical_issues(negative_reviews)
            }
        
        self.insights['pain_points'] = pain_points
        return pain_points
    
    def compare_bank_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare performance metrics across all banks
        
        Args:
            df: DataFrame with all analysis data
            
        Returns:
            Dict containing performance comparison
        """
        logger.info("Comparing bank performance...")
        
        comparison = {}
        
        for bank in df['bank'].unique():
            bank_data = df[df['bank'] == bank]
            
            comparison[bank] = {
                'total_reviews': len(bank_data),
                'average_rating': bank_data['rating'].mean(),
                'rating_distribution': bank_data['rating'].value_counts().sort_index().to_dict(),
                'sentiment_distribution': bank_data['sentiment_label'].value_counts().to_dict(),
                'theme_distribution': bank_data['primary_theme'].value_counts().head(5).to_dict(),
                'positive_sentiment_rate': len(bank_data[bank_data['sentiment_label'] == 'positive']) / len(bank_data) * 100,
                'negative_sentiment_rate': len(bank_data[bank_data['sentiment_label'] == 'negative']) / len(bank_data) * 100,
                'top_performing_aspects': self._identify_top_aspects(bank_data),
                'areas_for_improvement': self._identify_improvement_areas(bank_data)
            }
        
        # Calculate relative performance rankings
        comparison['rankings'] = self._calculate_rankings(comparison)
        
        self.insights['bank_comparison'] = comparison
        return comparison
    
    def generate_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate actionable recommendations for each bank
        
        Args:
            df: DataFrame with all analysis data
            
        Returns:
            Dict containing recommendations
        """
        logger.info("Generating recommendations...")
        
        recommendations = {}
        
        # Analyze overall patterns
        overall_insights = self._analyze_overall_patterns(df)
        
        for bank in df['bank'].unique():
            bank_data = df[df['bank'] == bank]
            
            # Generate bank-specific recommendations
            bank_recommendations = self._generate_bank_specific_recommendations(
                bank, bank_data, overall_insights
            )
            
            recommendations[bank] = {
                'immediate_actions': bank_recommendations['immediate'],
                'short_term_improvements': bank_recommendations['short_term'],
                'long_term_strategies': bank_recommendations['long_term'],
                'priority_levels': bank_recommendations['priorities'],
                'expected_impact': bank_recommendations['impact']
            }
        
        # Generate industry-wide recommendations
        recommendations['industry_wide'] = self._generate_industry_recommendations(df)
        
        self.recommendations = recommendations
        return recommendations
    
    def _extract_common_keywords(self, reviews_df: pd.DataFrame) -> List[str]:
        """Extract common keywords from reviews"""
        all_keywords = []
        
        for keywords_str in reviews_df['keywords'].dropna():
            try:
                if isinstance(keywords_str, str):
                    keywords = json.loads(keywords_str)
                    if isinstance(keywords, dict):
                        all_keywords.extend(list(keywords.keys()))
                    elif isinstance(keywords, list):
                        all_keywords.extend(keywords)
            except (json.JSONDecodeError, TypeError):
                continue
        
        # Count and return top keywords
        if all_keywords:
            keyword_counts = pd.Series(all_keywords).value_counts()
            return keyword_counts.head(10).index.tolist()
        
        return []
    
    def _extract_common_complaints(self, reviews_df: pd.DataFrame) -> List[str]:
        """Extract common complaints from negative reviews"""
        # This is a simplified version - in practice, you might use more sophisticated NLP
        common_phrases = [
            'slow', 'crash', 'error', 'bug', 'problem', 'issue', 'difficult',
            'complicated', 'frustrating', 'annoying', 'broken', 'fail'
        ]
        
        complaints = []
        for phrase in common_phrases:
            count = reviews_df['review'].str.contains(phrase, case=False, na=False).sum()
            if count > 0:
                complaints.append({'phrase': phrase, 'count': count})
        
        # Sort by count and return top complaints
        complaints.sort(key=lambda x: x['count'], reverse=True)
        return complaints[:5]
    
    def _identify_critical_issues(self, negative_reviews: pd.DataFrame) -> List[str]:
        """Identify critical issues that need immediate attention"""
        critical_keywords = ['crash', 'error', 'bug', 'broken', 'fail', 'security']
        critical_issues = []
        
        for keyword in critical_keywords:
            count = negative_reviews['review'].str.contains(keyword, case=False, na=False).sum()
            if count > 0:
                critical_issues.append({
                    'issue': keyword,
                    'count': count,
                    'severity': 'high' if count > 5 else 'medium'
                })
        
        return critical_issues
    
    def _identify_top_aspects(self, bank_data: pd.DataFrame) -> List[str]:
        """Identify top performing aspects for a bank"""
        # Analyze positive reviews to find strengths
        positive_reviews = bank_data[bank_data['sentiment_label'] == 'positive']
        
        if len(positive_reviews) > 0:
            top_themes = positive_reviews['primary_theme'].value_counts().head(3)
            return top_themes.index.tolist()
        
        return []
    
    def _identify_improvement_areas(self, bank_data: pd.DataFrame) -> List[str]:
        """Identify areas that need improvement for a bank"""
        # Analyze negative reviews to find weaknesses
        negative_reviews = bank_data[bank_data['sentiment_label'] == 'negative']
        
        if len(negative_reviews) > 0:
            weak_themes = negative_reviews['primary_theme'].value_counts().head(3)
            return weak_themes.index.tolist()
        
        return []
    
    def _calculate_rankings(self, comparison: Dict) -> Dict[str, List[str]]:
        """Calculate performance rankings across banks"""
        metrics = ['average_rating', 'positive_sentiment_rate', 'total_reviews']
        rankings = {}
        
        for metric in metrics:
            if metric in comparison.get(list(comparison.keys())[0], {}):
                # Sort banks by metric (higher is better for rating and sentiment)
                sorted_banks = sorted(
                    comparison.keys(),
                    key=lambda x: comparison[x].get(metric, 0),
                    reverse=True
                )
                rankings[metric] = sorted_banks
        
        return rankings
    
    def _analyze_overall_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall patterns across all banks"""
        patterns = {
            'total_reviews': len(df),
            'overall_average_rating': df['rating'].mean(),
            'overall_sentiment_distribution': df['sentiment_label'].value_counts().to_dict(),
            'most_common_themes': df['primary_theme'].value_counts().head(5).to_dict(),
            'rating_trends': df.groupby(df['date'].dt.to_period('M'))['rating'].mean().to_dict(),
            'sentiment_trends': df.groupby([df['date'].dt.to_period('M'), 'sentiment_label']).size().unstack(fill_value=0).to_dict()
        }
        
        return patterns
    
    def _generate_bank_specific_recommendations(self, bank: str, bank_data: pd.DataFrame, overall_insights: Dict) -> Dict[str, Any]:
        """Generate bank-specific recommendations"""
        recommendations = {
            'immediate': [],
            'short_term': [],
            'long_term': [],
            'priorities': {},
            'impact': {}
        }
        
        # Analyze bank's current performance
        avg_rating = bank_data['rating'].mean()
        negative_rate = len(bank_data[bank_data['sentiment_label'] == 'negative']) / len(bank_data) * 100
        
        # Immediate actions based on critical issues
        if negative_rate > 30:
            recommendations['immediate'].append("Address high negative sentiment rate immediately")
            recommendations['priorities']['sentiment'] = 'high'
        
        if avg_rating < 3.0:
            recommendations['immediate'].append("Investigate low rating issues")
            recommendations['priorities']['rating'] = 'high'
        
        # Short-term improvements
        recommendations['short_term'].extend([
            "Implement user feedback collection system",
            "Enhance customer support response time",
            "Optimize app performance based on common complaints"
        ])
        
        # Long-term strategies
        recommendations['long_term'].extend([
            "Develop comprehensive user experience improvement plan",
            "Invest in advanced analytics and monitoring",
            "Establish continuous improvement framework"
        ])
        
        # Expected impact
        recommendations['impact'] = {
            'immediate': "Reduce negative sentiment by 20-30%",
            'short_term': "Improve average rating by 0.5-1.0 points",
            'long_term': "Achieve industry-leading user satisfaction scores"
        }
        
        return recommendations
    
    def _generate_industry_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate industry-wide recommendations"""
        return {
            'common_improvements': [
                "Standardize user interface design across banking apps",
                "Implement industry-wide security best practices",
                "Establish performance benchmarking standards"
            ],
            'innovation_opportunities': [
                "Explore AI-powered customer support solutions",
                "Develop predictive analytics for user behavior",
                "Implement blockchain-based security features"
            ],
            'collaboration_areas': [
                "Share best practices for mobile banking UX",
                "Collaborate on security standards and compliance",
                "Joint research on emerging fintech technologies"
            ]
        }
    
    def generate_insights_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive insights summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'insights': self.insights,
            'recommendations': self.recommendations,
            'summary_statistics': {
                'total_insights': len(self.insights),
                'total_recommendations': len(self.recommendations),
                'analysis_complete': True
            }
        }
        
        return summary
    
    def save_insights(self, output_dir: str = "reports/insights") -> str:
        """Save insights to JSON file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"insights_analysis_{timestamp}.json"
        filepath = output_path / filename
        
        insights_data = self.generate_insights_summary()
        
        with open(filepath, 'w') as f:
            json.dump(insights_data, f, indent=2, default=str)
        
        logger.info(f"Insights saved to: {filepath}")
        return str(filepath)

def main():
    """Test the insights generator"""
    print("🔍 Insights Generator Test")
    print("=" * 40)
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'bank': ['CBE', 'CBE', 'BOA', 'BOA', 'Dashen', 'Dashen'] * 10,
        'rating': np.random.randint(1, 6, 60),
        'sentiment_label': np.random.choice(['positive', 'negative', 'neutral'], 60),
        'primary_theme': np.random.choice(['UI', 'Performance', 'Security', 'Features'], 60),
        'date': pd.date_range('2024-01-01', periods=60, freq='D'),
        'keywords': ['["fast", "easy", "good"]'] * 60,
        'sentiment_confidence': np.random.random(60)
    })
    
    # Initialize insights generator
    generator = InsightsGenerator()
    
    # Test insights generation
    try:
        print("Analyzing satisfaction drivers...")
        drivers = generator.analyze_satisfaction_drivers(sample_data)
        
        print("Analyzing pain points...")
        pain_points = generator.analyze_pain_points(sample_data)
        
        print("Comparing bank performance...")
        comparison = generator.compare_bank_performance(sample_data)
        
        print("Generating recommendations...")
        recommendations = generator.generate_recommendations(sample_data)
        
        print("✅ All insights generated successfully!")
        
        # Save insights
        output_file = generator.save_insights()
        print(f"Insights saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
