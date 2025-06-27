# RealNewsGuard

RealNewsGuard is an **AI-powered fake news detection system** that combines advanced text analysis, image verification, and automated fact-checking to help users identify potentially misleading or false information in news articles.

---

## 🚀 Features

- **Text Analysis:** Detects clickbait headlines, sensationalist language, and suspicious claims using NLP and ML models.
- **Image Verification:** Checks for image manipulation using Error Level Analysis (ELA) and metadata inspection.
- **External API Integration:** Fetches related news articles from NewsAPI to provide context.
- **Automated Fact-Checking:** Searches leading fact-checking websites (Snopes, PolitiFact, FactCheck.org) for related claims.
- **Comprehensive Output:** Returns prediction (REAL/FAKE/SUSPICIOUS), confidence score, explanations, and reference links.
- **User-Friendly Interface:** Clean front-end for easy interaction.
- **Easy to Extend:** Modular backend for adding new sources, models, or verification methods.

---

## 🗂️ Project Structure

```
RealNewsGuard/
├── backend/
│   ├── api/
│   ├── models/
│   ├── static/
│   ├── uploads/
│   ├── utils/
│   ├── main.py
│   └── requirements.txt
├── data/
├── frontend/
│   ├── public/
│   └── src/
├── models/
└── README.md
```

---

## ⚙️ Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/codexcherry/RealNewsGuard.git
cd RealNewsGuard
```

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Frontend Setup

```bash
cd ../frontend
python -m http.server  # Or use any preferred static server
```

---

## 🧠 How It Works

1. **Input:** User submits a news headline, content, and an optional image.
2. **Analysis:**  
   - Text is checked for clickbait, sensationalist language, and suspicious claims.
   - Images are analyzed for manipulation (ELA and metadata).
   - Related news is fetched from NewsAPI.
   - Fact-checking platforms are queried for relevant articles.
3. **Output:**  
   - Prediction: REAL, FAKE, or SUSPICIOUS.
   - Confidence score and explainable AI output (LIME/SHAP).
   - Related news links and fact-check summaries.

---

## 🛠️ Technologies Used

- **Backend:** FastAPI, Python
- **Frontend:** HTML, CSS, JavaScript
- **ML/NLP:** scikit-learn, LIME, SHAP
- **Image Analysis:** Pillow
- **External APIs:** NewsAPI
- **Testing:** Pytest

---

## 📝 Example Use Cases

- **Journalists:** Rapidly verify the authenticity of viral stories.
- **Educators:** Teach students about misinformation and critical reading.
- **Fact-Checkers:** Automate initial screening of news submissions.

---

## 📈 Recent Updates

- Enhanced detection algorithms and added heuristic analysis.
- Refined confidence scoring and explanations.
- Added comprehensive test scripts (`test_prediction.py`, `test_api.py`).
- Broadened detection for clickbait, sensationalism, and suspicious claims.

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 🙏 Acknowledgments

- NewsAPI for providing access to reliable news.
- Fact-checking organizations (Snopes, PolitiFact, FactCheck.org).
- Open source contributors and reviewers.

---
