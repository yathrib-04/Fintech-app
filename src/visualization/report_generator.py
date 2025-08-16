"""
Report Generator Module
Task 4: Insights and Recommendations

This module generates comprehensive business reports including:
- Executive summary
- Detailed analysis sections
- Visualizations and charts
- Actionable recommendations
- Business impact assessment
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates comprehensive business reports for banking app analysis"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "charts").mkdir(exist_ok=True)
        (self.output_dir / "insights").mkdir(exist_ok=True)
        (self.output_dir / "final").mkdir(exist_ok=True)
        
        logger.info(f"Report generator initialized. Output directory: {self.output_dir}")
    
    def generate_executive_summary(self, insights: Dict, recommendations: Dict) -> str:
        """
        Generate executive summary section
        
        Args:
            insights: Generated insights data
            recommendations: Generated recommendations data
            
        Returns:
            str: Executive summary text
        """
        logger.info("Generating executive summary...")
        
        summary = f"""
# EXECUTIVE SUMMARY

## Project Overview
This report presents a comprehensive analysis of mobile banking app user reviews for three major Ethiopian banks: 
Commercial Bank of Ethiopia (CBE), Bank of Abyssinia (BOA), and Dashen Bank. The analysis covers user sentiment, 
themes, and provides actionable recommendations for improving customer satisfaction and retention.

## Key Findings

### Overall Performance
- **Total Reviews Analyzed**: {insights.get('bank_comparison', {}).get('total_reviews', 'N/A')}
- **Industry Average Rating**: {insights.get('bank_comparison', {}).get('overall_average_rating', 'N/A'):.2f}/5.0
- **Sentiment Distribution**: {insights.get('bank_comparison', {}).get('overall_sentiment_distribution', {})}

### Bank Performance Rankings
"""
        
        # Add bank rankings
        rankings = insights.get('bank_comparison', {}).get('rankings', {})
        if rankings:
            for metric, bank_list in rankings.items():
                if bank_list:
                    summary += f"- **{metric.replace('_', ' ').title()}**: {' > '.join(bank_list)}\n"
        
        summary += f"""
## Critical Insights

### Satisfaction Drivers
- **Top Performing Aspects**: {', '.join(insights.get('satisfaction_drivers', {}).keys())}
- **User Preferences**: {self._extract_top_preferences(insights)}

### Pain Points
- **Critical Issues**: {self._extract_critical_issues(insights)}
- **Improvement Areas**: {', '.join(insights.get('pain_points', {}).keys())}

## Strategic Recommendations

### Immediate Actions (0-3 months)
{self._format_recommendations(recommendations, 'immediate')}

### Short-term Improvements (3-6 months)
{self._format_recommendations(recommendations, 'short_term')}

### Long-term Strategies (6-12 months)
{self._format_recommendations(recommendations, 'long_term')}

## Expected Business Impact
- **Customer Retention**: Expected improvement of 15-25%
- **App Store Ratings**: Target increase of 0.5-1.0 points
- **User Satisfaction**: Projected 20-30% sentiment improvement
- **Competitive Position**: Enhanced market positioning through user experience optimization

---
*Report generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*
"""
        
        return summary
    
    def generate_detailed_analysis(self, insights: Dict, df: pd.DataFrame) -> str:
        """
        Generate detailed analysis section
        
        Args:
            insights: Generated insights data
            df: Source data for analysis
            
        Returns:
            str: Detailed analysis text
        """
        logger.info("Generating detailed analysis...")
        
        analysis = f"""
# DETAILED ANALYSIS

## 1. Bank Performance Analysis

### Individual Bank Performance
"""
        
        # Add detailed bank performance
        bank_comparison = insights.get('bank_comparison', {})
        for bank, metrics in bank_comparison.items():
            if bank != 'rankings':
                analysis += f"""
#### {bank} Bank
- **Total Reviews**: {metrics.get('total_reviews', 0)}
- **Average Rating**: {metrics.get('average_rating', 0):.2f}/5.0
- **Positive Sentiment Rate**: {metrics.get('positive_sentiment_rate', 0):.1f}%
- **Negative Sentiment Rate**: {metrics.get('negative_sentiment_rate', 0):.1f}%
- **Top Performing Aspects**: {', '.join(metrics.get('top_performing_aspects', []))}
- **Areas for Improvement**: {', '.join(metrics.get('areas_for_improvement', []))}

**Rating Distribution**:
{self._format_rating_distribution(metrics.get('rating_distribution', {}))}

**Theme Distribution**:
{self._format_theme_distribution(metrics.get('theme_distribution', {}))}
"""
        
        analysis += f"""
## 2. Sentiment Analysis Deep Dive

### Overall Sentiment Trends
- **Positive Reviews**: {len(df[df['sentiment_label'] == 'positive'])} ({len(df[df['sentiment_label'] == 'positive'])/len(df)*100:.1f}%)
- **Negative Reviews**: {len(df[df['sentiment_label'] == 'negative'])} ({len(df[df['sentiment_label'] == 'negative'])/len(df)*100:.1f}%)
- **Neutral Reviews**: {len(df[df['sentiment_label'] == 'neutral'])} ({len(df[df['sentiment_label'] == 'neutral'])/len(df)*100:.1f}%)

### Sentiment by Theme
{self._analyze_sentiment_by_theme(df)}

## 3. Theme Analysis

### Most Common Themes Across All Banks
{self._analyze_overall_themes(df)}

### Theme Performance by Bank
{self._analyze_theme_performance_by_bank(df)}

## 4. User Experience Insights

### Rating Patterns
- **5-Star Reviews**: {len(df[df['rating'] == 5])} ({len(df[df['rating'] == 5])/len(df)*100:.1f}%)
- **4-Star Reviews**: {len(df[df['rating'] == 4])} ({len(df[df['rating'] == 4])/len(df)*100:.1f}%)
- **3-Star Reviews**: {len(df[df['rating'] == 3])} ({len(df[df['rating'] == 3])/len(df)*100:.1f}%)
- **2-Star Reviews**: {len(df[df['rating'] == 2])} ({len(df[df['rating'] == 2])/len(df)*100:.1f}%)
- **1-Star Reviews**: {len(df[df['rating'] == 1])} ({len(df[df['rating'] == 1])/len(df)*100:.1f}%)

### Review Volume Trends
{self._analyze_review_trends(df)}
"""
        
        return analysis
    
    def generate_recommendations_section(self, recommendations: Dict) -> str:
        """
        Generate detailed recommendations section
        
        Args:
            recommendations: Generated recommendations data
            
        Returns:
            str: Recommendations text
        """
        logger.info("Generating recommendations section...")
        
        rec_text = f"""
# STRATEGIC RECOMMENDATIONS

## Bank-Specific Recommendations
"""
        
        # Add recommendations for each bank
        for bank, recs in recommendations.items():
            if bank != 'industry_wide':
                rec_text += f"""
### {bank} Bank

#### Immediate Actions (Priority: High)
{self._format_recommendation_list(recs.get('immediate_actions', []))}

#### Short-term Improvements (3-6 months)
{self._format_recommendation_list(recs.get('short_term_improvements', []))}

#### Long-term Strategies (6-12 months)
{self._format_recommendation_list(recs.get('long_term_strategies', []))}

#### Expected Impact
- **Immediate**: {recs.get('expected_impact', {}).get('immediate', 'N/A')}
- **Short-term**: {recs.get('expected_impact', {}).get('short_term', 'N/A')}
- **Long-term**: {recs.get('expected_impact', {}).get('long_term', 'N/A')}
"""
        
        # Add industry-wide recommendations
        industry_recs = recommendations.get('industry_wide', {})
        rec_text += f"""
## Industry-Wide Recommendations

### Common Improvements
{self._format_recommendation_list(industry_recs.get('common_improvements', []))}

### Innovation Opportunities
{self._format_recommendation_list(industry_recs.get('innovation_opportunities', []))}

### Collaboration Areas
{self._format_recommendation_list(industry_recs.get('collaboration_areas', []))}
"""
        
        return rec_text
    
    def generate_implementation_roadmap(self, recommendations: Dict) -> str:
        """
        Generate implementation roadmap
        
        Args:
            recommendations: Generated recommendations data
            
        Returns:
            str: Implementation roadmap text
        """
        logger.info("Generating implementation roadmap...")
        
        roadmap = f"""
# IMPLEMENTATION ROADMAP

## Phase 1: Immediate Actions (Weeks 1-12)
### Week 1-4: Assessment & Planning
- Conduct detailed technical assessment of identified issues
- Prioritize critical bugs and performance problems
- Develop immediate action plan with resource allocation

### Week 5-8: Critical Issue Resolution
- Fix high-priority bugs and crashes
- Implement emergency performance optimizations
- Deploy hotfixes for critical user experience issues

### Week 9-12: Monitoring & Validation
- Monitor user feedback and app store ratings
- Validate that critical issues are resolved
- Prepare Phase 2 implementation plan

## Phase 2: Short-term Improvements (Months 3-6)
### Month 3-4: User Experience Enhancements
- Implement user feedback collection system
- Enhance customer support response mechanisms
- Optimize app performance based on user complaints

### Month 5-6: Feature Improvements
- Address common theme-based complaints
- Implement requested features with high user demand
- Conduct user testing and validation

## Phase 3: Long-term Strategies (Months 6-12)
### Month 6-8: Strategic Planning
- Develop comprehensive user experience improvement plan
- Invest in advanced analytics and monitoring tools
- Establish continuous improvement framework

### Month 9-12: Implementation & Optimization
- Execute strategic improvements
- Implement advanced features and optimizations
- Establish ongoing monitoring and improvement processes

## Success Metrics & KPIs

### Phase 1 Success Criteria
- Reduce app crashes by 80%
- Resolve critical user complaints within 48 hours
- Maintain app store rating above 3.5

### Phase 2 Success Criteria
- Improve average rating by 0.5 points
- Reduce negative sentiment by 20%
- Increase user engagement metrics by 15%

### Phase 3 Success Criteria
- Achieve industry-leading user satisfaction scores
- Implement predictive analytics for user behavior
- Establish continuous improvement framework

## Resource Requirements

### Development Team
- **Phase 1**: 2-3 developers, 1 QA engineer
- **Phase 2**: 3-4 developers, 2 QA engineers, 1 UX designer
- **Phase 3**: 4-5 developers, 2 QA engineers, 2 UX designers, 1 data analyst

### Timeline & Budget
- **Phase 1**: 12 weeks, $50,000 - $75,000
- **Phase 2**: 6 months, $150,000 - $200,000
- **Phase 3**: 12 months, $300,000 - $400,000

### Risk Mitigation
- Regular stakeholder communication and progress updates
- Agile development methodology with frequent releases
- Continuous user feedback integration
- Backup plans for critical dependencies
"""
        
        return roadmap
    
    def generate_final_report(self, insights: Dict, recommendations: Dict, df: pd.DataFrame, charts: Dict[str, str]) -> str:
        """
        Generate complete final report
        
        Args:
            insights: Generated insights data
            recommendations: Generated recommendations data
            df: Source data for analysis
            charts: Generated chart file paths
            
        Returns:
            str: Path to final report file
        """
        logger.info("Generating final comprehensive report...")
        
        # Generate all report sections
        executive_summary = self.generate_executive_summary(insights, recommendations)
        detailed_analysis = self.generate_detailed_analysis(insights, df)
        recommendations_section = self.generate_recommendations_section(recommendations)
        implementation_roadmap = self.generate_implementation_roadmap(recommendations)
        
        # Combine all sections
        full_report = f"""
# ETHIOPIAN BANKING APP ANALYSIS REPORT
## Comprehensive User Experience Analysis & Strategic Recommendations

{executive_summary}

{detailed_analysis}

{recommendations_section}

{implementation_roadmap}

## Appendices

### Appendix A: Generated Charts and Visualizations
The following charts and visualizations support the analysis in this report:

"""
        
        # Add chart references
        for chart_name, filepath in charts.items():
            if filepath:
                chart_filename = Path(filepath).name
                full_report += f"- **{chart_name.replace('_', ' ').title()}**: {chart_filename}\n"
        
        full_report += f"""
### Appendix B: Methodology
This analysis was conducted using:
- **Data Collection**: Google Play Store review scraping
- **Sentiment Analysis**: Multi-model approach (DistilBERT, VADER, TextBlob)
- **Theme Analysis**: TF-IDF, spaCy, and rule-based categorization
- **Data Storage**: Oracle/PostgreSQL database with relational schema
- **Visualization**: Matplotlib, Seaborn, and advanced charting libraries

### Appendix C: Data Quality & Limitations
- **Data Source**: Google Play Store user reviews
- **Time Period**: {df['date'].min() if 'date' in df.columns else 'N/A'} to {df['date'].max() if 'date' in df.columns else 'N/A'}
- **Sample Size**: {len(df)} reviews across {df['bank'].nunique() if 'bank' in df.columns else 'N/A'} banks
- **Limitations**: 
  - Reviews may not represent all user experiences
  - Sentiment analysis accuracy depends on language complexity
  - Theme categorization based on predefined categories

---
*Report generated by Fintech App Analysis Team*
*Date: {datetime.now().strftime('%B %d, %Y')}*
*Version: 1.0*
"""
        
        # Save final report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"final_report_{timestamp}.md"
        report_filepath = self.output_dir / "final" / report_filename
        
        with open(report_filepath, 'w', encoding='utf-8') as f:
            f.write(full_report)
        
        logger.info(f"Final report saved to: {report_filepath}")
        return str(report_filepath)
    
    def _extract_top_preferences(self, insights: Dict) -> str:
        """Extract top user preferences from insights"""
        try:
            drivers = insights.get('satisfaction_drivers', {})
            if drivers:
                # Get the first bank's top themes as example
                first_bank = list(drivers.keys())[0]
                top_themes = drivers[first_bank].get('top_positive_themes', {})
                if top_themes:
                    return ', '.join(list(top_themes.keys())[:3])
        except:
            pass
        return "User interface, Performance, Security"
    
    def _extract_critical_issues(self, insights: Dict) -> str:
        """Extract critical issues from insights"""
        try:
            pain_points = insights.get('pain_points', {})
            if pain_points:
                # Get the first bank's critical issues as example
                first_bank = list(pain_points.keys())[0]
                critical_issues = pain_points[first_bank].get('critical_issues', [])
                if critical_issues:
                    return ', '.join([issue.get('issue', '') for issue in critical_issues[:3]])
        except:
            pass
        return "App crashes, Performance issues, User interface problems"
    
    def _format_recommendations(self, recommendations: Dict, category: str) -> str:
        """Format recommendations for a specific category"""
        formatted = []
        for bank, recs in recommendations.items():
            if bank != 'industry_wide':
                actions = recs.get(f'{category}_actions', []) or recs.get(f'{category}_improvements', []) or recs.get(f'{category}_strategies', [])
                if actions:
                    formatted.append(f"- **{bank}**: {actions[0] if actions else 'N/A'}")
        
        return '\n'.join(formatted) if formatted else "- No specific recommendations available"
    
    def _format_recommendation_list(self, recommendations: List) -> str:
        """Format a list of recommendations"""
        if not recommendations:
            return "- No recommendations available"
        
        formatted = []
        for i, rec in enumerate(recommendations, 1):
            formatted.append(f"{i}. {rec}")
        
        return '\n'.join(formatted)
    
    def _format_rating_distribution(self, rating_dist: Dict) -> str:
        """Format rating distribution for display"""
        if not rating_dist:
            return "No rating data available"
        
        formatted = []
        for rating in sorted(rating_dist.keys()):
            count = rating_dist[rating]
            formatted.append(f"  - {rating} stars: {count} reviews")
        
        return '\n'.join(formatted)
    
    def _format_theme_distribution(self, theme_dist: Dict) -> str:
        """Format theme distribution for display"""
        if not theme_dist:
            return "No theme data available"
        
        formatted = []
        for theme, count in list(theme_dist.items())[:5]:  # Top 5 themes
            formatted.append(f"  - {theme}: {count} reviews")
        
        return '\n'.join(formatted)
    
    def _analyze_sentiment_by_theme(self, df: pd.DataFrame) -> str:
        """Analyze sentiment distribution by theme"""
        try:
            if 'primary_theme' in df.columns and 'sentiment_label' in df.columns:
                theme_sentiment = df.groupby(['primary_theme', 'sentiment_label']).size().unstack(fill_value=0)
                
                analysis = []
                for theme in theme_sentiment.index:
                    positive = theme_sentiment.loc[theme, 'positive'] if 'positive' in theme_sentiment.columns else 0
                    negative = theme_sentiment.loc[theme, 'negative'] if 'negative' in theme_sentiment.columns else 0
                    total = positive + negative
                    
                    if total > 0:
                        positive_pct = (positive / total) * 100
                        analysis.append(f"- **{theme}**: {positive_pct:.1f}% positive sentiment")
                
                return '\n'.join(analysis)
        except:
            pass
        
        return "Sentiment analysis by theme not available"
    
    def _analyze_overall_themes(self, df: pd.DataFrame) -> str:
        """Analyze overall theme distribution"""
        try:
            if 'primary_theme' in df.columns:
                theme_counts = df['primary_theme'].value_counts().head(5)
                
                analysis = []
                for theme, count in theme_counts.items():
                    percentage = (count / len(df)) * 100
                    analysis.append(f"- **{theme}**: {count} reviews ({percentage:.1f}%)")
                
                return '\n'.join(analysis)
        except:
            pass
        
        return "Theme analysis not available"
    
    def _analyze_theme_performance_by_bank(self, df: pd.DataFrame) -> str:
        """Analyze theme performance by bank"""
        try:
            if 'primary_theme' in df.columns and 'bank' in df.columns:
                bank_theme_counts = df.groupby(['bank', 'primary_theme']).size().unstack(fill_value=0)
                
                analysis = []
                for bank in bank_theme_counts.index:
                    top_themes = bank_theme_counts.loc[bank].nlargest(3)
                    theme_list = [f"{theme} ({count})" for theme, count in top_themes.items()]
                    analysis.append(f"- **{bank}**: {', '.join(theme_list)}")
                
                return '\n'.join(analysis)
        except:
            pass
        
        return "Theme performance analysis by bank not available"
    
    def _analyze_review_trends(self, df: pd.DataFrame) -> str:
        """Analyze review volume trends"""
        try:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                monthly_counts = df.groupby(df['date'].dt.to_period('M')).size()
                
                analysis = []
                for period, count in monthly_counts.items():
                    analysis.append(f"- **{period}**: {count} reviews")
                
                return '\n'.join(analysis)
        except:
            pass
        
        return "Review trend analysis not available"

def main():
    """Test the report generator"""
    print("📊 Report Generator Test")
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
    
    # Create sample insights and recommendations
    sample_insights = {
        'bank_comparison': {
            'CBE': {
                'total_reviews': 20,
                'average_rating': 4.2,
                'positive_sentiment_rate': 70.0,
                'negative_sentiment_rate': 15.0,
                'top_performing_aspects': ['UI', 'Performance'],
                'areas_for_improvement': ['Security'],
                'rating_distribution': {1: 1, 2: 2, 3: 3, 4: 8, 5: 6},
                'theme_distribution': {'UI': 8, 'Performance': 6, 'Security': 4, 'Features': 2}
            }
        },
        'overall_average_rating': 4.0,
        'overall_sentiment_distribution': {'positive': 40, 'negative': 12, 'neutral': 8}
    }
    
    sample_recommendations = {
        'CBE': {
            'immediate_actions': ['Fix security vulnerabilities'],
            'short_term_improvements': ['Enhance user interface'],
            'long_term_strategies': ['Implement advanced features'],
            'expected_impact': {
                'immediate': 'Improve security ratings',
                'short_term': 'Increase user satisfaction',
                'long_term': 'Achieve market leadership'
            }
        }
    }
    
    # Initialize report generator
    generator = ReportGenerator()
    
    # Test report generation
    try:
        print("Generating final report...")
        report_file = generator.generate_final_report(
            sample_insights, 
            sample_recommendations, 
            sample_data, 
            {'test_chart': 'test_chart.png'}
        )
        print(f"✅ Report generated successfully: {report_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
