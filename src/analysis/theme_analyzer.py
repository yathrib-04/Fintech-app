"""
Thematic Analysis Module for Banking App Reviews
Task 2: Sentiment and Thematic Analysis

This module provides comprehensive thematic analysis including:
- Keyword extraction using TF-IDF and spaCy
- Topic modeling with LDA
- Theme clustering and categorization
- Banking-specific theme identification
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
try:
    import spacy
    SPACY_AVAILABLE = True
    # Load English language model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # If model not available, download it
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Using basic text processing.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Using basic keyword extraction.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ThemeAnalyzer:
    """Comprehensive thematic analysis for banking app reviews"""
    
    def __init__(self):
        self.spacy_available = SPACY_AVAILABLE
        self.sklearn_available = SKLEARN_AVAILABLE
        
        # Banking-specific theme categories
        self.banking_themes = {
            'performance': {
                'keywords': ['slow', 'fast', 'speed', 'loading', 'lag', 'crash', 'freeze', 'response', 'performance'],
                'description': 'App performance and speed issues'
            },
            'usability': {
                'keywords': ['easy', 'hard', 'difficult', 'simple', 'complex', 'intuitive', 'user-friendly', 'complicated'],
                'description': 'Ease of use and user experience'
            },
            'features': {
                'keywords': ['transfer', 'payment', 'login', 'security', 'notification', 'balance', 'transaction', 'account'],
                'description': 'App features and functionality'
            },
            'support': {
                'keywords': ['help', 'support', 'customer service', 'contact', 'assistance', 'service', 'helpful'],
                'description': 'Customer support and assistance'
            },
            'security': {
                'keywords': ['secure', 'safe', 'password', 'biometric', 'authentication', 'privacy', 'protection'],
                'description': 'Security and privacy features'
            },
            'ui_ux': {
                'keywords': ['interface', 'design', 'layout', 'button', 'menu', 'navigation', 'visual', 'appearance'],
                'description': 'User interface and design'
            },
            'reliability': {
                'keywords': ['stable', 'reliable', 'bug', 'error', 'issue', 'problem', 'work', 'function'],
                'description': 'App stability and reliability'
            },
            'accessibility': {
                'keywords': ['accessible', 'disability', 'font', 'size', 'color', 'contrast', 'readable'],
                'description': 'Accessibility features'
            }
        }
        
        # Create output directory
        self.output_dir = Path('../data/processed')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TF-IDF vectorizer if available
        if self.sklearn_available:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.8
            )
    
    def preprocess_text(self, text):
        """
        Preprocess text for analysis
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Preprocessed text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords_basic(self, text, top_n=20):
        """
        Basic keyword extraction without advanced NLP
        
        Args:
            text (str): Preprocessed text
            top_n (int): Number of top keywords to return
            
        Returns:
            list: List of (keyword, frequency) tuples
        """
        # Simple word frequency analysis
        words = text.split()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        # Filter words
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count frequencies
        word_counts = Counter(filtered_words)
        
        return word_counts.most_common(top_n)
    
    def extract_keywords_spacy(self, text, top_n=20):
        """
        Extract keywords using spaCy NLP
        
        Args:
            text (str): Preprocessed text
            top_n (int): Number of top keywords to return
            
        Returns:
            list: List of (keyword, frequency) tuples
        """
        if not self.spacy_available:
            return self.extract_keywords_basic(text, top_n)
        
        try:
            doc = nlp(text)
            
            # Extract nouns, adjectives, and verbs
            keywords = []
            for token in doc:
                if (token.pos_ in ['NOUN', 'ADJ', 'VERB'] and 
                    not token.is_stop and 
                    not token.is_punct and 
                    len(token.text) > 2):
                    keywords.append(token.lemma_.lower())
            
            # Count frequencies
            keyword_counts = Counter(keywords)
            
            return keyword_counts.most_common(top_n)
            
        except Exception as e:
            logger.warning(f"spaCy keyword extraction failed: {str(e)}")
            return self.extract_keywords_basic(text, top_n)
    
    def extract_keywords_tfidf(self, texts, top_n=20):
        """
        Extract keywords using TF-IDF
        
        Args:
            texts (list): List of preprocessed texts
            top_n (int): Number of top keywords to return
            
        Returns:
            list: List of (keyword, tfidf_score) tuples
        """
        if not self.sklearn_available:
            return []
        
        try:
            # Fit TF-IDF vectorizer
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Get feature names
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Calculate average TF-IDF scores across all documents
            avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Create keyword-score pairs
            keyword_scores = list(zip(feature_names, avg_scores))
            
            # Sort by score and return top keywords
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return keyword_scores[:top_n]
            
        except Exception as e:
            logger.warning(f"TF-IDF keyword extraction failed: {str(e)}")
            return []
    
    def identify_themes(self, text, keywords):
        """
        Identify themes based on keywords and banking theme categories
        
        Args:
            text (str): Review text
            keywords (list): Extracted keywords
            
        Returns:
            dict: Theme identification results
        """
        theme_scores = defaultdict(float)
        identified_themes = []
        
        # Convert keywords to set for faster lookup
        keyword_set = {kw.lower() for kw, _ in keywords}
        
        # Score each theme based on keyword matches
        for theme_name, theme_info in self.banking_themes.items():
            score = 0
            matches = []
            
            for keyword in theme_info['keywords']:
                if keyword.lower() in keyword_set:
                    score += 1
                    matches.append(keyword)
            
            # Normalize score by theme keyword count
            if theme_info['keywords']:
                normalized_score = score / len(theme_info['keywords'])
                theme_scores[theme_name] = normalized_score
                
                if normalized_score > 0.1:  # Threshold for theme identification
                    identified_themes.append({
                        'theme': theme_name,
                        'score': normalized_score,
                        'matches': matches,
                        'description': theme_info['description']
                    })
        
        # Sort themes by score
        identified_themes.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'themes': identified_themes,
            'theme_scores': dict(theme_scores),
            'primary_theme': identified_themes[0]['theme'] if identified_themes else 'general'
        }
    
    def perform_topic_modeling(self, texts, n_topics=5):
        """
        Perform LDA topic modeling
        
        Args:
            texts (list): List of preprocessed texts
            n_topics (int): Number of topics to extract
            
        Returns:
            dict: Topic modeling results
        """
        if not self.sklearn_available:
            return {}
        
        try:
            # Use TF-IDF features
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Perform LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=100
            )
            
            lda_output = lda.fit_transform(tfidf_matrix)
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topic_words = [(word, topic[i]) for i, word in zip(top_words_idx, top_words)]
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': topic_words,
                    'top_words': top_words[:5]  # Top 5 words
                })
            
            return {
                'n_topics': n_topics,
                'topics': topics,
                'document_topics': lda_output.tolist()
            }
            
        except Exception as e:
            logger.warning(f"Topic modeling failed: {str(e)}")
            return {}
    
    def analyze_review_themes(self, text):
        """
        Perform comprehensive thematic analysis on a single review
        
        Args:
            text (str): Review text
            
        Returns:
            dict: Thematic analysis results
        """
        # Preprocess text
        preprocessed_text = self.preprocess_text(text)
        
        # Extract keywords using different methods
        keywords_basic = self.extract_keywords_basic(preprocessed_text)
        keywords_spacy = self.extract_keywords_spacy(preprocessed_text)
        
        # Combine keywords (prioritize spaCy if available)
        if self.spacy_available:
            keywords = keywords_spacy
        else:
            keywords = keywords_basic
        
        # Identify themes
        theme_analysis = self.identify_themes(preprocessed_text, keywords)
        
        return {
            'keywords': keywords,
            'themes': theme_analysis['themes'],
            'primary_theme': theme_analysis['primary_theme'],
            'theme_scores': theme_analysis['theme_scores']
        }
    
    def analyze_batch_themes(self, df):
        """
        Perform thematic analysis on a batch of reviews
        
        Args:
            df (pd.DataFrame): DataFrame with 'review' column
            
        Returns:
            pd.DataFrame: DataFrame with thematic analysis results
        """
        logger.info(f"Starting thematic analysis for {len(df)} reviews...")
        
        # Initialize results columns
        df_themes = df.copy()
        
        # Add theme columns
        theme_columns = [
            'primary_theme', 'theme_scores', 'keywords', 'identified_themes'
        ]
        
        for col in theme_columns:
            df_themes[col] = None
        
        # Process each review
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processed {idx}/{len(df)} reviews...")
            
            text = str(row['review'])
            
            # Analyze themes
            theme_results = self.analyze_review_themes(text)
            
            # Store results
            df_themes.at[idx, 'primary_theme'] = theme_results['primary_theme']
            df_themes.at[idx, 'theme_scores'] = json.dumps(theme_results['theme_scores'])
            df_themes.at[idx, 'keywords'] = json.dumps(theme_results['keywords'])
            df_themes.at[idx, 'identified_themes'] = json.dumps(theme_results['themes'])
        
        logger.info("✅ Thematic analysis completed!")
        return df_themes
    
    def generate_theme_summary(self, df):
        """
        Generate summary statistics for thematic analysis
        
        Args:
            df (pd.DataFrame): DataFrame with thematic analysis results
            
        Returns:
            dict: Theme summary statistics
        """
        summary = {}
        
        # Primary theme distribution
        theme_dist = df['primary_theme'].value_counts()
        summary['primary_theme_distribution'] = theme_dist.to_dict()
        
        # Theme distribution by bank
        bank_themes = df.groupby('bank')['primary_theme'].value_counts().unstack(fill_value=0)
        summary['themes_by_bank'] = bank_themes.to_dict()
        
        # Most common keywords
        all_keywords = []
        for keywords_json in df['keywords'].dropna():
            try:
                keywords = json.loads(keywords_json)
                all_keywords.extend([kw for kw, _ in keywords])
            except:
                continue
        
        keyword_counts = Counter(all_keywords)
        summary['top_keywords'] = dict(keyword_counts.most_common(50))
        
        # Theme frequency analysis
        all_themes = []
        for themes_json in df['identified_themes'].dropna():
            try:
                themes = json.loads(themes_json)
                all_themes.extend([theme['theme'] for theme in themes])
            except:
                continue
        
        theme_counts = Counter(all_themes)
        summary['theme_frequency'] = dict(theme_counts.most_common())
        
        return summary
    
    def save_theme_results(self, df, filename_prefix='theme_analysis'):
        """
        Save thematic analysis results
        
        Args:
            df (pd.DataFrame): DataFrame with thematic analysis
            filename_prefix (str): Prefix for output files
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save CSV
        csv_file = self.output_dir / f"{filename_prefix}_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"Theme analysis results saved to: {csv_file}")
        
        # Generate and save summary
        summary = self.generate_theme_summary(df)
        summary_file = self.output_dir / f"{filename_prefix}_summary_{timestamp}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Theme analysis summary saved to: {summary_file}")
        
        return csv_file, summary_file

def main():
    """Main execution function for testing"""
    analyzer = ThemeAnalyzer()
    
    # Test with sample texts
    test_texts = [
        "The app is very slow when making transfers and crashes frequently.",
        "Great user interface design and easy to navigate. Very secure!",
        "Poor customer support and difficult to contact help desk.",
        "Excellent features for mobile banking, fast loading times."
    ]
    
    print("🧪 Testing Thematic Analysis...")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nReview {i}: {text}")
        results = analyzer.analyze_review_themes(text)
        
        print(f"Primary Theme: {results['primary_theme']}")
        print(f"Keywords: {[kw for kw, _ in results['keywords'][:5]]}")
        print(f"Identified Themes: {[t['theme'] for t in results['themes'][:3]]}")

if __name__ == "__main__":
    main()
