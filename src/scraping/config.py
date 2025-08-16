"""
Configuration file for Google Play Store scraping
Task 1: Data Collection and Preprocessing

Contains app IDs, settings, and constants for the scraping process.
"""

# Google Play Store App IDs for Banking Apps
BANK_APPS = {
    'CBE': {
        'name': 'Commercial Bank of Ethiopia Mobile',
        'app_id': 'com.cbe.mobile',
        'target_reviews': 400,
        'description': 'Official mobile banking app for CBE'
    },
    'BOA': {
        'name': 'Bank of America Mobile Banking',
        'app_id': 'com.infonow.bofa',
        'target_reviews': 400,
        'description': 'Official mobile banking app for Bank of America'
    },
    'Dashen': {
        'name': 'Dashen Bank Mobile Banking',
        'app_id': 'com.dashenbank.mobile',
        'target_reviews': 400,
        'description': 'Official mobile banking app for Dashen Bank'
    }
}

# Alternative app IDs (in case primary ones don't work)
ALTERNATIVE_APP_IDS = {
    'CBE': [
        'com.cbe.mobile',
        'com.cbe.ethiopia.mobile',
        'com.commercialbank.ethiopia.mobile'
    ],
    'BOA': [
        'com.infonow.bofa',
        'com.bankofamerica.mobile',
        'com.boa.mobilebanking'
    ],
    'Dashen': [
        'com.dashenbank.mobile',
        'com.dashen.ethiopia.mobile',
        'com.dashenbank.ethiopia.mobile'
    ]
}

# Scraping Settings
SCRAPING_SETTINGS = {
    'language': 'en',           # Language for reviews
    'country': 'us',            # Country/region
    'sort_order': 'newest',     # Sort order: newest, rating, helpfulness
    'delay_between_requests': 2,  # Seconds to wait between requests
    'max_retries': 3,           # Maximum retry attempts for failed requests
    'timeout': 30,              # Request timeout in seconds
}

# Data Quality Thresholds
QUALITY_THRESHOLDS = {
    'min_review_length': 15,    # Minimum characters for a valid review
    'max_missing_data': 5.0,    # Maximum percentage of missing data allowed
    'min_reviews_per_bank': 300, # Minimum reviews required per bank
    'target_total_reviews': 1200, # Target total reviews across all banks
}

# File Paths
DATA_PATHS = {
    'raw_data': '../data/raw',
    'processed_data': '../data/processed',
    'logs': '../logs',
    'reports': '../reports'
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_handler': True,
    'console_handler': True
}

# Review Categories for Analysis
REVIEW_CATEGORIES = {
    'performance': ['slow', 'fast', 'speed', 'loading', 'lag', 'crash', 'freeze'],
    'usability': ['easy', 'hard', 'difficult', 'simple', 'complex', 'intuitive'],
    'features': ['transfer', 'payment', 'login', 'security', 'notification'],
    'support': ['help', 'support', 'customer service', 'contact', 'assistance'],
    'security': ['secure', 'safe', 'password', 'biometric', 'authentication'],
    'ui_ux': ['interface', 'design', 'layout', 'button', 'menu', 'navigation']
}

# Expected CSV Columns
EXPECTED_COLUMNS = [
    'review',      # Review text
    'rating',      # Rating (1-5)
    'date',        # Review date (YYYY-MM-DD)
    'bank',        # Bank name/code
    'app_id',      # App identifier
    'scraped_at'   # When review was scraped
]

# Data Validation Rules
VALIDATION_RULES = {
    'rating_range': (1.0, 5.0),
    'date_format': '%Y-%m-%d',
    'required_columns': ['review', 'rating', 'date', 'bank'],
    'text_cleaning': {
        'min_length': 15,
        'max_length': 10000,
        'remove_special_chars': True,
        'normalize_whitespace': True
    }
}
