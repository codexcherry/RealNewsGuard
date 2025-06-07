import os
import numpy as np
import json
from utils.text_utils import clean_text, extract_keywords
from utils.image_utils import process_image, extract_exif_data, error_level_analysis
from api.news_verification import verify_news
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import lime
import shap
from PIL import Image
from utils.text_utils import (
    detect_clickbait, 
    detect_sensationalist_language,
    detect_suspicious_claims,
)

# Initialize model and tokenizer
MODEL_NAME = "distilbert-base-uncased"

# Create cache directory if it doesn't exist
os.makedirs("backend/models/cache", exist_ok=True)

def load_model():
    """
    Load the pre-trained model and tokenizer.
    
    Returns:
        tuple: (model, tokenizer)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def get_prediction(headline, content="", image_path=None):
    """
    Get fake news prediction by combining text and image analysis
    
    Args:
        headline (str): News headline
        content (str, optional): News content
        image_path (str, optional): Path to the news image
        
    Returns:
        dict: Prediction results including label, confidence, and explanation
    """
    # Get text-based prediction
    text_pred = predict_from_text(headline, content)
    
    # Get image-based prediction if image is provided
    image_pred = predict_from_image(image_path, headline) if image_path else None
    
    # Combine predictions
    final_pred = combine_predictions(text_pred, image_pred)
    
    # Generate explanation
    explanation = generate_explanation(headline, content, image_path, text_pred, image_pred)
    
    return {
        "label": final_pred["label"],
        "confidence": final_pred["confidence"],
        "explanation": explanation,
        "text_analysis": text_pred,
        "image_analysis": image_pred
    }

def predict_from_text(headline, content=""):
    """
    Predict fake news based on text analysis
    
    Args:
        headline (str): News headline
        content (str, optional): News content
        
    Returns:
        dict: Text-based prediction results
    """
    # In a real implementation, we would:
    # 1. Preprocess the text
    # 2. Convert to embeddings or features
    # 3. Run through a trained model (e.g., BERT, RoBERTa)
    
    # For demonstration purposes, we'll use a simple rule-based approach
    # This is NOT a real fake news detector
    
    # Clean the text
    cleaned_headline = clean_text(headline)
    cleaned_content = clean_text(content) if content else ""
    
    # Extract keywords
    keywords = extract_keywords(cleaned_headline + " " + cleaned_content)
    
    # Check for suspicious patterns (very simplified)
    suspicious_words = ["shocking", "you won't believe", "secret", "conspiracy", 
                       "they don't want you to know", "miracle", "cure", "hoax"]
    
    suspicion_score = 0
    for word in suspicious_words:
        if word in cleaned_headline.lower() or (cleaned_content and word in cleaned_content.lower()):
            suspicion_score += 0.1
    
    # Calculate confidence (this is just a demonstration)
    fake_confidence = min(0.5 + suspicion_score, 0.95)
    real_confidence = 1 - fake_confidence
    
    # Determine label
    if fake_confidence > real_confidence:
        label = "FAKE"
    else:
        label = "REAL"
    
    return {
        "label": label,
        "confidence": max(fake_confidence, real_confidence),
        "keywords": keywords,
        "suspicion_score": suspicion_score
    }

def predict_from_image(image_path, headline=""):
    """
    Predict fake news based on image analysis
    
    Args:
        image_path (str): Path to the news image
        headline (str, optional): News headline for text-image matching
        
    Returns:
        dict: Image-based prediction results
    """
    if not image_path or not os.path.exists(image_path):
        return None
    
    # In a real implementation, we would:
    # 1. Extract image features using a pre-trained model
    # 2. Check for manipulation using forensic techniques
    # 3. Match image with text using models like CLIP
    
    # For demonstration purposes only
    
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Extract EXIF data
        exif_data = extract_exif_data(img)
        
        # Check for manipulation using ELA
        ela_score = error_level_analysis(img, image_path)
        manipulation_detected = ela_score > 15.0
        
        # Check for metadata issues
        metadata_issues = len(exif_data) == 0 or "Software" in exif_data
        
        # Calculate confidence (this is just a demonstration)
        fake_confidence = 0.3  # Base value
        
        if manipulation_detected:
            fake_confidence += 0.4
        
        if metadata_issues:
            fake_confidence += 0.3
        
        real_confidence = 1 - fake_confidence
        
        # Determine label
        if fake_confidence > real_confidence:
            label = "FAKE"
        else:
            label = "REAL"
        
        return {
            "label": label,
            "confidence": max(fake_confidence, real_confidence),
            "manipulation_detected": manipulation_detected,
            "metadata_issues": metadata_issues
        }
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        return {
            "label": "UNKNOWN",
            "confidence": 0.5,
            "error": str(e)
        }

def combine_predictions(text_pred, image_pred):
    """
    Combine text and image predictions
    
    Args:
        text_pred (dict): Text-based prediction
        image_pred (dict, optional): Image-based prediction
        
    Returns:
        dict: Combined prediction results
    """
    if not image_pred:
        return text_pred
    
    # In a real implementation, we would use a more sophisticated fusion method
    # like a weighted average, MLP, or ensemble technique
    
    # For demonstration, we'll use a simple weighted average
    text_weight = 0.7
    image_weight = 0.3
    
    # Calculate weighted confidence for fake
    if text_pred["label"] == "FAKE":
        text_fake_conf = text_pred["confidence"]
    else:
        text_fake_conf = 1 - text_pred["confidence"]
        
    if image_pred["label"] == "FAKE":
        image_fake_conf = image_pred["confidence"]
    else:
        image_fake_conf = 1 - image_pred["confidence"]
    
    # Combined fake confidence
    combined_fake_conf = (text_weight * text_fake_conf) + (image_weight * image_fake_conf)
    
    # Determine final label and confidence
    if combined_fake_conf > 0.5:
        label = "FAKE"
        confidence = combined_fake_conf
    else:
        label = "REAL"
        confidence = 1 - combined_fake_conf
    
    return {
        "label": label,
        "confidence": confidence
    }

def generate_explanation(headline, content, image_path, text_pred, image_pred):
    """
    Generate human-readable explanation for the prediction
    
    Args:
        headline (str): News headline
        content (str): News content
        image_path (str): Path to the news image
        text_pred (dict): Text-based prediction
        image_pred (dict): Image-based prediction
        
    Returns:
        str: Explanation of the prediction
    """
    explanations = []
    
    # Text-based explanations
    if text_pred:
        if text_pred["label"] == "FAKE":
            if text_pred["suspicion_score"] > 0:
                explanations.append("The text contains suspicious language patterns often found in fake news.")
            explanations.append(f"Keywords that raised concerns: {', '.join(text_pred['keywords'][:3])}")
        else:
            explanations.append("The text appears to use language consistent with legitimate news sources.")
    
    # Image-based explanations
    if image_pred:
        if image_pred["manipulation_detected"]:
            explanations.append("The image shows signs of digital manipulation.")
        
        if image_pred["metadata_issues"]:
            explanations.append("The image metadata suggests possible editing.")
    
    # Combined explanation
    if not explanations:
        return "No specific issues were detected with this content."
    
    return " ".join(explanations)

def predict_fake_news(headline, content, image_path=None):
    """
    Predict if news is fake based on headline, content, and image.
    
    Args:
        headline (str): News headline
        content (str): News content
        image_path (str): Path to the image (optional)
        
    Returns:
        dict: Prediction results
    """
    # Text-based features
    text_features = extract_text_features(headline, content)
    
    # Image-based features (if image is provided)
    image_features = extract_image_features(image_path) if image_path else {}
    
    # Combine features for prediction
    features = {**text_features, **image_features}
    
    # Add additional check for health-related misinformation
    health_keywords = ['covid', 'covid-19', 'coronavirus', 'vaccine', 'cure', 'treatment', 'disease', 'virus', 'cancer', 'aids']
    has_health_keywords = False
    for keyword in health_keywords:
        if keyword.lower() in headline.lower() or keyword.lower() in content.lower():
            has_health_keywords = True
            break
    
    # Calculate suspicion score based on features
    suspicion_score = calculate_suspicion_score(features)
    
    # Apply additional suspicion for health-related content
    if has_health_keywords:
        suspicious_patterns = (features.get("suspicious_patterns_headline", []) + 
                              features.get("suspicious_patterns_content", []))
        if suspicious_patterns:
            suspicion_score += 0.3
            print(f"DEBUG: Added 0.3 to suspicion score for health misinformation, now {suspicion_score}")
    
    print(f"DEBUG: Final suspicion score: {suspicion_score}")
    
    # Check for certain combinations that strongly indicate fake news
    is_likely_fake = (
        (features.get("is_clickbait", False) and features.get("has_suspicious_claims_content", False)) or
        (features.get("is_clickbait", False) and features.get("is_sensationalist_content", False) and "secret" in str(features.get("sensationalist_patterns_content", [])).lower()) or
        (features.get("has_suspicious_claims_headline", False) and features.get("has_suspicious_claims_content", False)) or
        # Special handling for medical misinformation
        (has_health_keywords and features.get("has_suspicious_claims_headline", False)) or
        (has_health_keywords and features.get("has_suspicious_claims_content", False))
    )
    
    # Determine prediction based on score and feature combinations
    if is_likely_fake or suspicion_score > 0.5:  # Lower threshold for FAKE classification
        prediction = "FAKE"
        confidence = min(suspicion_score + 0.3, 0.99)  # Increase confidence for fake news
    elif suspicion_score > 0.3:  # Lower threshold for SUSPICIOUS classification
        prediction = "SUSPICIOUS"
        confidence = suspicion_score + 0.1  # Slight boost to confidence
    else:
        prediction = "REAL"
        confidence = 1 - suspicion_score
    
    print(f"DEBUG: Prediction: {prediction}, Confidence: {confidence}")
    
    # Generate explanation
    explanation = generate_explanation(features, suspicion_score, has_health_keywords)
    
    return {
        "prediction": prediction,
        "confidence": round(float(confidence), 2),
        "explanation": explanation,
        "features": features
    }

def extract_text_features(headline, content):
    """
    Extract features from headline and content.
    
    Args:
        headline (str): News headline
        content (str): News content
        
    Returns:
        dict: Text features
    """
    features = {}
    
    # Check for clickbait headline
    is_clickbait, clickbait_patterns = detect_clickbait(headline)
    features["is_clickbait"] = is_clickbait
    features["clickbait_patterns"] = clickbait_patterns
    
    # Check for sensationalist language
    is_sensationalist_headline, sensationalist_patterns_headline = detect_sensationalist_language(headline)
    is_sensationalist_content, sensationalist_patterns_content = detect_sensationalist_language(content)
    
    features["is_sensationalist_headline"] = is_sensationalist_headline
    features["sensationalist_patterns_headline"] = sensationalist_patterns_headline
    features["is_sensationalist_content"] = is_sensationalist_content
    features["sensationalist_patterns_content"] = sensationalist_patterns_content
    
    # Check for suspicious claims
    has_suspicious_claims_headline, suspicious_patterns_headline = detect_suspicious_claims(headline)
    has_suspicious_claims_content, suspicious_patterns_content = detect_suspicious_claims(content)
    
    features["has_suspicious_claims_headline"] = has_suspicious_claims_headline
    features["suspicious_patterns_headline"] = suspicious_patterns_headline
    features["has_suspicious_claims_content"] = has_suspicious_claims_content
    features["suspicious_patterns_content"] = suspicious_patterns_content
    
    # Print debug information
    print(f"DEBUG: Headline: '{headline}'")
    print(f"DEBUG: Content: '{content[:100]}...'")
    print(f"DEBUG: is_clickbait: {is_clickbait}, patterns: {clickbait_patterns}")
    print(f"DEBUG: is_sensationalist_headline: {is_sensationalist_headline}, patterns: {sensationalist_patterns_headline}")
    print(f"DEBUG: is_sensationalist_content: {is_sensationalist_content}, patterns: {sensationalist_patterns_content}")
    print(f"DEBUG: has_suspicious_claims_headline: {has_suspicious_claims_headline}, patterns: {suspicious_patterns_headline}")
    print(f"DEBUG: has_suspicious_claims_content: {has_suspicious_claims_content}, patterns: {suspicious_patterns_content}")
    
    # Extract keywords
    headline_keywords = extract_keywords(headline, top_n=5)
    content_keywords = extract_keywords(content, top_n=10)
    
    features["headline_keywords"] = headline_keywords
    features["content_keywords"] = content_keywords
    
    # Text length features
    features["headline_length"] = len(headline.split())
    features["content_length"] = len(content.split())
    
    # Readability metrics (simplified)
    features["avg_word_length"] = calculate_avg_word_length(content)
    
    return features

def extract_image_features(image_path):
    """
    Extract features from an image.
    
    Args:
        image_path (str): Path to the image
        
    Returns:
        dict: Image features
    """
    if not image_path or not os.path.exists(image_path):
        return {}
    
    features = {}
    
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Extract EXIF data
        exif_data = extract_exif_data(img)
        
        # Check for metadata issues
        features["has_metadata_issues"] = len(exif_data) == 0 or "Software" in exif_data
        
        # Perform error level analysis
        ela_score = error_level_analysis(img, image_path)
        features["ela_score"] = ela_score
        features["potential_manipulation"] = ela_score > 15.0
        
        return features
    
    except Exception as e:
        print(f"Error extracting image features: {str(e)}")
        return {"error": str(e)}

def calculate_suspicion_score(features):
    """
    Calculate a suspicion score based on extracted features.
    
    Args:
        features (dict): Extracted features
        
    Returns:
        float: Suspicion score between 0 and 1
    """
    score = 0.0
    
    # Text-based scores
    if features.get("is_clickbait", False):
        score += 0.2
        print(f"DEBUG: Adding 0.2 for clickbait, score now {score}")
    
    if features.get("is_sensationalist_headline", False):
        score += 0.15
        print(f"DEBUG: Adding 0.15 for sensationalist headline, score now {score}")
    
    if features.get("is_sensationalist_content", False):
        score += 0.1
        print(f"DEBUG: Adding 0.1 for sensationalist content, score now {score}")
    
    if features.get("has_suspicious_claims_headline", False):
        score += 0.25
        print(f"DEBUG: Adding 0.25 for suspicious claims in headline, score now {score}")
    
    if features.get("has_suspicious_claims_content", False):
        score += 0.2
        print(f"DEBUG: Adding 0.2 for suspicious claims in content, score now {score}")
    
    # Image-based scores
    if features.get("has_metadata_issues", False):
        score += 0.1
        print(f"DEBUG: Adding 0.1 for metadata issues, score now {score}")
    
    if features.get("potential_manipulation", False):
        score += 0.3
        print(f"DEBUG: Adding 0.3 for potential manipulation, score now {score}")
    
    # Cap the score at 1.0
    return min(score, 1.0)

def generate_explanation(features, suspicion_score, has_health_keywords=False):
    """
    Generate an explanation for the prediction based on features.
    
    Args:
        features (dict): Extracted features
        suspicion_score (float): Calculated suspicion score
        has_health_keywords (bool): Whether health-related keywords were detected
        
    Returns:
        str: Explanation text
    """
    explanations = []
    
    # Text-based explanations
    if features.get("is_clickbait", False):
        patterns = features.get("clickbait_patterns", [])
        if patterns:
            explanations.append(f"Clickbait patterns detected in headline ({', '.join(patterns[:3])}).")
        else:
            explanations.append("Clickbait language detected in headline.")
    
    if features.get("is_sensationalist_headline", False):
        patterns = features.get("sensationalist_patterns_headline", [])
        if patterns:
            explanations.append(f"Sensationalist language in headline ({', '.join(patterns[:3])}).")
        else:
            explanations.append("Sensationalist language detected in headline.")
    
    if features.get("is_sensationalist_content", False):
        patterns = features.get("sensationalist_patterns_content", [])
        if patterns:
            explanations.append(f"Sensationalist language in content ({', '.join(patterns[:3])}).")
        else:
            explanations.append("Sensationalist language detected in content.")
    
    if features.get("has_suspicious_claims_headline", False):
        patterns = features.get("suspicious_patterns_headline", [])
        if patterns:
            explanations.append(f"Suspicious claims in headline ({', '.join(patterns[:3])}).")
        else:
            explanations.append("Suspicious claims detected in headline.")
    
    if features.get("has_suspicious_claims_content", False):
        patterns = features.get("suspicious_patterns_content", [])
        if patterns:
            explanations.append(f"Suspicious claims detected ({', '.join(patterns[:3])}).")
        else:
            explanations.append("Suspicious claims detected in content.")
    
    # Add health misinformation explanation if applicable
    if has_health_keywords and (features.get("has_suspicious_claims_headline", False) or features.get("has_suspicious_claims_content", False)):
        explanations.append("Health-related misinformation detected. Be especially cautious about unverified health claims.")
    
    # Image-based explanations
    if features.get("has_metadata_issues", False):
        explanations.append("Image metadata suggests possible editing.")
    
    if features.get("potential_manipulation", False):
        explanations.append("Image shows signs of digital manipulation.")
    
    # If no issues were found
    if not explanations:
        if suspicion_score < 0.2:
            return "No significant issues detected in the content."
        else:
            return "Some minor issues detected, but not enough to classify as fake news."
    
    return " ".join(explanations)

def calculate_avg_word_length(text):
    """
    Calculate the average word length in a text.
    
    Args:
        text (str): Input text
        
    Returns:
        float: Average word length
    """
    words = text.split()
    if not words:
        return 0
    
    total_length = sum(len(word) for word in words)
    return total_length / len(words)

def explain_prediction(headline, content, image_path=None):
    """
    Generate a detailed explanation of the prediction using LIME or SHAP.
    
    Args:
        headline (str): News headline
        content (str): News content
        image_path (str): Path to the image (optional)
        
    Returns:
        dict: Detailed explanation
    """
    # This is a placeholder for a more sophisticated explanation system
    # In a real implementation, we would use LIME or SHAP to explain the model's decision
    
    # Make prediction
    prediction = predict_fake_news(headline, content, image_path)
    
    # Extract top contributing features
    features = prediction["features"]
    
    # Determine top factors
    factors = []
    
    if features.get("is_clickbait", False):
        factors.append({
            "name": "Clickbait headline",
            "impact": "high",
            "examples": features.get("clickbait_patterns", [])[:3]
        })
    
    if features.get("has_suspicious_claims_content", False):
        factors.append({
            "name": "Suspicious claims",
            "impact": "high",
            "examples": features.get("suspicious_patterns_content", [])[:3]
        })
    
    if features.get("potential_manipulation", False):
        factors.append({
            "name": "Image manipulation",
            "impact": "high",
            "details": f"ELA score: {features.get('ela_score', 0):.2f}"
        })
    
    return {
        "prediction": prediction["prediction"],
        "confidence": prediction["confidence"],
        "summary": prediction["explanation"],
        "top_factors": factors
    } 