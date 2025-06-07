from utils.text_utils import detect_clickbait, detect_sensationalist_language, detect_suspicious_claims

def test_fake_news_detection():
    # Test with obvious fake news headline and content
    headline = "SHOCKING: Scientists discover miracle cure that doctors don't want you to know about"
    content = "This incredible breakthrough will change everything. The government is hiding this secret treatment that can cure all diseases overnight. You won't believe what happens next!"
    
    # Print detection results
    print("\nDetection Results:")
    print("Clickbait:", detect_clickbait(headline))
    print("Sensationalist Headline:", detect_sensationalist_language(headline))
    print("Sensationalist Content:", detect_sensationalist_language(content))
    print("Suspicious Claims Headline:", detect_suspicious_claims(headline))
    print("Suspicious Claims Content:", detect_suspicious_claims(content))
    
    # Extract features manually
    features = {}
    is_clickbait, clickbait_patterns = detect_clickbait(headline)
    is_sensationalist_headline, sensationalist_patterns_headline = detect_sensationalist_language(headline)
    is_sensationalist_content, sensationalist_patterns_content = detect_sensationalist_language(content)
    has_suspicious_claims_headline, suspicious_patterns_headline = detect_suspicious_claims(headline)
    has_suspicious_claims_content, suspicious_patterns_content = detect_suspicious_claims(content)
    
    features["is_clickbait"] = is_clickbait
    features["clickbait_patterns"] = clickbait_patterns
    features["is_sensationalist_headline"] = is_sensationalist_headline
    features["sensationalist_patterns_headline"] = sensationalist_patterns_headline
    features["is_sensationalist_content"] = is_sensationalist_content
    features["sensationalist_patterns_content"] = sensationalist_patterns_content
    features["has_suspicious_claims_headline"] = has_suspicious_claims_headline
    features["suspicious_patterns_headline"] = suspicious_patterns_headline
    features["has_suspicious_claims_content"] = has_suspicious_claims_content
    features["suspicious_patterns_content"] = suspicious_patterns_content
    
    # Print features
    print("\nExtracted Features:")
    for key, value in features.items():
        if isinstance(value, (bool, int, float)):
            print(f"  {key}: {value}")
        elif isinstance(value, list) and value:
            print(f"  {key}: {value}")
    
    # Calculate suspicion score manually
    print("\nManual Suspicion Score Calculation:")
    score = 0.0
    
    if features.get("is_clickbait", False):
        score += 0.2
        print("  is_clickbait: +0.2")
    
    if features.get("is_sensationalist_headline", False):
        score += 0.15
        print("  is_sensationalist_headline: +0.15")
    
    if features.get("is_sensationalist_content", False):
        score += 0.1
        print("  is_sensationalist_content: +0.1")
    
    if features.get("has_suspicious_claims_headline", False):
        score += 0.25
        print("  has_suspicious_claims_headline: +0.25")
    
    if features.get("has_suspicious_claims_content", False):
        score += 0.2
        print("  has_suspicious_claims_content: +0.2")
    
    print(f"  Total Score: {score}")
    
    # Determine prediction based on score
    if score > 0.7:
        prediction = "FAKE"
        confidence = min(score, 0.99)
    elif score > 0.4:
        prediction = "SUSPICIOUS"
        confidence = score
    else:
        prediction = "REAL"
        confidence = 1 - score
    
    print(f"\nFinal Prediction: {prediction} (Confidence: {confidence:.2f})")
    
    # Test with real news headline and content
    print("\n\n--- Testing with Real News ---")
    headline_real = "Local community center opens new facility"
    content_real = "The community center has completed construction of its new recreation facility. The building includes a gymnasium, swimming pool, and meeting rooms for local residents."
    
    # Extract features manually for real news
    features_real = {}
    is_clickbait, clickbait_patterns = detect_clickbait(headline_real)
    is_sensationalist_headline, sensationalist_patterns_headline = detect_sensationalist_language(headline_real)
    is_sensationalist_content, sensationalist_patterns_content = detect_sensationalist_language(content_real)
    has_suspicious_claims_headline, suspicious_patterns_headline = detect_suspicious_claims(headline_real)
    has_suspicious_claims_content, suspicious_patterns_content = detect_suspicious_claims(content_real)
    
    features_real["is_clickbait"] = is_clickbait
    features_real["is_sensationalist_headline"] = is_sensationalist_headline
    features_real["is_sensationalist_content"] = is_sensationalist_content
    features_real["has_suspicious_claims_headline"] = has_suspicious_claims_headline
    features_real["has_suspicious_claims_content"] = has_suspicious_claims_content
    
    # Calculate suspicion score manually for real news
    print("\nManual Suspicion Score Calculation (Real News):")
    score_real = 0.0
    
    if features_real.get("is_clickbait", False):
        score_real += 0.2
        print("  is_clickbait: +0.2")
    
    if features_real.get("is_sensationalist_headline", False):
        score_real += 0.15
        print("  is_sensationalist_headline: +0.15")
    
    if features_real.get("is_sensationalist_content", False):
        score_real += 0.1
        print("  is_sensationalist_content: +0.1")
    
    if features_real.get("has_suspicious_claims_headline", False):
        score_real += 0.25
        print("  has_suspicious_claims_headline: +0.25")
    
    if features_real.get("has_suspicious_claims_content", False):
        score_real += 0.2
        print("  has_suspicious_claims_content: +0.2")
    
    print(f"  Total Score: {score_real}")
    
    # Determine prediction based on score
    if score_real > 0.7:
        prediction_real = "FAKE"
        confidence_real = min(score_real, 0.99)
    elif score_real > 0.4:
        prediction_real = "SUSPICIOUS"
        confidence_real = score_real
    else:
        prediction_real = "REAL"
        confidence_real = 1 - score_real
    
    print(f"\nFinal Prediction (Real News): {prediction_real} (Confidence: {confidence_real:.2f})")

if __name__ == "__main__":
    test_fake_news_detection() 