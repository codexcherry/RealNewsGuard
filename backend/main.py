from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import aiofiles
from typing import Optional
import uuid
import shutil

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

# Configure CORS - update to allow all origins explicitly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Ensure directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("static/processed_images", exist_ok=True)

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
        public_image_url = None
        if image:
            # Generate a unique filename
            file_extension = os.path.splitext(image.filename)[1]
            image_name = f"{uuid.uuid4()}{file_extension}"
            image_path = f"uploads/{image_name}"
            processed_image_path = f"static/processed_images/{image_name}"
            
            # Save the uploaded image
            async with aiofiles.open(image_path, "wb") as out_file:
                content_bytes = await image.read()
                await out_file.write(content_bytes)
            
            # Create a copy in the static directory for public access
            shutil.copy(image_path, processed_image_path)
            
            # Create a URL for the processed image
            public_image_url = f"/static/processed_images/{image_name}"
            
            print(f"Image saved to {image_path} and {processed_image_path}")
        
        # Process the text
        cleaned_headline = clean_text(headline)
        cleaned_content = clean_text(content) if content else ""
        
        # Process the image if provided
        if image_path:
            processed_image_path = process_image(image_path)
            print(f"Image processed: {processed_image_path}")
        
        # Predict if news is fake
        prediction_result = predict_fake_news(cleaned_headline, cleaned_content, image_path)
        
        # Get related news articles
        try:
            related_news = get_related_news(cleaned_headline)
        except Exception as e:
            print(f"Error fetching related news: {str(e)}")
            related_news = {"status": "error", "message": f"Failed to fetch related news: {str(e)}"}
        
        # Verify with fact-checking sites
        verification_results = verify_news(cleaned_headline, cleaned_content)
        
        # Check if headline and image match (if image is provided)
        image_text_match = None
        if image_path and "image_analysis" in prediction_result:
            image_analysis = prediction_result.get("image_analysis", {})
            text_analysis = prediction_result.get("text_analysis", {})
            
            # Determine if there's a mismatch between image and text
            if image_analysis and text_analysis:
                image_text_match = {
                    "match_score": 0.8,  # Placeholder - in a real implementation, this would be calculated
                    "match_status": "Consistent",  # Placeholder
                }
        
        # Combine results
        result = {
            "prediction": prediction_result["prediction"],
            "confidence": prediction_result["confidence"],
            "explanation": prediction_result["explanation"],
            "related_news": related_news,
            "fact_checks": verification_results,
            "image_url": public_image_url,
            "image_text_match": image_text_match
        }
        
        return result
    
    except Exception as e:
        print(f"Error in analyze_news: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 