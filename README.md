# Fintech App Analysis Project

## Business Objective
Omega Consultancy is supporting banks to improve their mobile apps to enhance customer retention and satisfaction. This project analyzes user reviews from Google Play Store for three banking apps to identify satisfaction drivers and pain points.

## Project Overview
- **Data Collection**: Scrape reviews from Google Play Store for 3 banking apps
- **Analysis**: Sentiment analysis and thematic extraction using NLP
- **Database**: Store cleaned data in Oracle database
- **Insights**: Generate actionable recommendations with visualizations

## Target Banks
- Commercial Bank of Ethiopia (CBE) - 4.4★ rating
- Bank of America (BOA) - 2.8★ rating  
- Dashen Bank - 4.0★ rating

## Project Structure
```
Fintech-app/
├── data/                   # Raw and processed data
├── src/                    # Source code
│   ├── scraping/          # Web scraping scripts
│   ├── analysis/          # NLP and analysis scripts
│   ├── database/          # Database operations
│   └── visualization/     # Plotting and reporting
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Tasks
1. **Task 1**: Data Collection and Preprocessing
2. **Task 2**: Sentiment and Thematic Analysis  
3. **Task 3**: Database Storage in Oracle
4. **Task 4**: Insights and Recommendations

## Setup Instructions
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up Oracle XE database
4. Run scripts in order: scraping → analysis → database → insights

## Key Metrics
- Target: 1,200+ reviews (400 per bank)
- Data quality: <5% missing data
- 3+ themes per bank identified
- 2+ drivers/pain points per bank

## Technologies Used
- Python 3.8+
- google-play-scraper
- Transformers (DistilBERT)
- spaCy/TF-IDF for keyword extraction
- Oracle Database
- Matplotlib/Seaborn for visualization
