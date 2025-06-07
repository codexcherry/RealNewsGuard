from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import aiofiles
from typing import Optional
import uuid

# Import our modules
from api.news_verification import verify_news
from api.news_api import get_related_news
from utils.image_utils import process_image
from utils.text_utils import clean_text
from models.prediction import predict_fake_news

# Create FastAPI app
app = FastAPI(
    title="RealNewsGuard API",
    description="API for fake news detection using text and image analysis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return {"message": "Welcome to RealNewsGuard API"}

@app.post("/analyze")
async def analyze_news(
    headline: str = Form(...),
    content: str = Form(...),
    image: Optional[UploadFile] = File(None),
):
    try:
        # Save image if provided
        image_path = None
        if image:
            file_extension = os.path.splitext(image.filename)[1]
            image_name = f"{uuid.uuid4()}{file_extension}"
            image_path = f"uploads/{image_name}"
            
            async with aiofiles.open(image_path, "wb") as out_file:
                content = await image.read()
                await out_file.write(content)
        
        # Process the text
        cleaned_headline = clean_text(headline)
        cleaned_content = clean_text(content) if content else ""
        
        # Process the image
        image_path = process_image(image_path)
        
        # Predict if news is fake
        prediction_result = predict_fake_news(cleaned_headline, cleaned_content, image_path)
        
        # Get related news articles
        related_news = get_related_news(cleaned_headline)
        
        # Verify with fact-checking sites
        verification_results = verify_news(cleaned_headline, cleaned_content)
        
        # Combine results
        result = {
            "prediction": prediction_result["prediction"],
            "confidence": prediction_result["confidence"],
            "explanation": prediction_result["explanation"],
            "related_news": related_news,
            "fact_checks": verification_results
        }
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 