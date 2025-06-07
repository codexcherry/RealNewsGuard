# RealNewsGuard

RealNewsGuard is an AI-powered fake news detection system that combines text analysis, image verification, and fact-checking to help users identify potentially misleading or false information.

## Features

- **Text Analysis**: Detects clickbait headlines, sensationalist language, and suspicious claims
- **Image Verification**: Checks for image manipulation using Error Level Analysis (ELA)
- **External API Integration**: Fetches related news articles from NewsAPI
- **Fact-Checking**: Searches fact-checking websites (Snopes, PolitiFact, FactCheck.org) for related claims
- **Comprehensive Results**: Provides prediction (REAL/FAKE/SUSPICIOUS), confidence score, and explanation

## Project Structure

```
RealNewsGuard/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── news_api.py
│   │   └── news_verification.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── prediction.py
│   ├── static/
│   ├── uploads/
│   │   └── .gitkeep
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_utils.py
│   │   └── text_utils.py
│   ├── main.py
│   └── requirements.txt
├── data/
│   └── .gitkeep
├── frontend/
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── app.js
│       └── styles.css
├── models/
│   └── .gitkeep
└── README.md
```

## How It Works

1. **Input**: Users provide a news headline, content, and optionally an image
2. **Analysis**:
   - Text is analyzed for clickbait patterns, sensationalist language, and suspicious claims
   - Images are checked for manipulation using Error Level Analysis and metadata examination
   - Related news articles are fetched from NewsAPI
   - Fact-checking sites are queried for relevant fact-checks
3. **Output**: The system provides a prediction (REAL/FAKE/SUSPICIOUS) with confidence score, explanation, related news articles, and fact-check results

## Technologies Used

- **Backend**: FastAPI, Python
- **Frontend**: HTML, CSS, JavaScript
- **Text Analysis**: scikit-learn, LIME, SHAP
- **Image Analysis**: Pillow
- **External APIs**: NewsAPI

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NewsAPI for providing access to news articles
- Fact-checking organizations (Snopes, PolitiFact, FactCheck.org) for their valuable work

## Recent Updates

### Improved Fake News Detection
- Enhanced the detection algorithm to better identify fake news content
- Added heuristic analysis of combined patterns (clickbait + suspicious claims, etc.)
- Refined confidence scoring to provide more accurate results
- Improved explanations to provide more detailed information about detected issues

### Testing
- Added test scripts to verify system functionality
  - `test_prediction.py` - Tests the core prediction model directly
  - `test_api.py` - Tests the API endpoints with various examples
  
### Detection Features
The system now looks for multiple indicators of fake news, including:
- Clickbait language in headlines
- Sensationalist language in headlines and content
- Suspicious claims in both headlines and content
- Combinations of these features that strongly indicate fake news
- Image manipulation (when images are provided) 